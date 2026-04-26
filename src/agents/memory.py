"""
Memory Stream — ChromaDB-backed retrieval with dynamic reflection threshold.

v5 changes (Phase 3):
- Incremental ChromaDB sync: only last 7 days on cold start, watermark-based after
- Heuristic importance scoring: short content gets a default without an LLM call
- Batch importance scoring: up to 5 memories scored in one LLM call
- Batch touch_memories: single DB UPDATE instead of N sequential calls
- HyDE-lite query builder: structures retrieval query as a first-person memory prompt
- Two-stage focal-point reflection retained from v4 (quality model used)
"""
import logging
import random
from datetime import datetime, timedelta, timezone

from src.agents.llm import (
    build_conversation_reflection_prompt,
    LLMClient,
    build_importance_prompt,
    build_batch_importance_prompt,
    build_focal_questions_prompt,
    build_focal_insights_prompt,
    build_reflection_prompt,
    quick_importance,
)
from src.agents.vector_store import upsert_memory, retrieve_top_memories
from src.utils import db

logger = logging.getLogger(__name__)

BASE_REFLECTION_THRESHOLD = float(__import__("os").getenv("REFLECTION_THRESHOLD", "150"))
VARIANCE_SPIKE_THRESHOLD  = 8.0
CHROMA_COLD_START_DAYS    = 7      # How many days back to sync on bot restart


# ─── Importance scoring ────────────────────────────────────────────────────────

async def score_importance(llm: LLMClient, content: str) -> float:
    """Score importance. Uses heuristic for short content; LLM for longer content."""
    heuristic = quick_importance(content)
    if heuristic is not None:
        return heuristic
    try:
        result = await llm.complete_json(build_importance_prompt(content), max_tokens=100)
        return float(result.get("score", 5.0))
    except Exception:
        return 5.0


async def score_importance_batch(llm: LLMClient, contents: list[str]) -> list[float]:
    """
    Score up to 5 memories in a single LLM call.
    Falls back to per-item heuristic scoring on failure.
    """
    if not contents:
        return []

    # Apply heuristics first — only send items that need full scoring
    results   = [None] * len(contents)
    needs_llm = []  # (original_index, content)

    for i, c in enumerate(contents):
        h = quick_importance(c)
        if h is not None:
            results[i] = h
        else:
            needs_llm.append((i, c))

    if not needs_llm:
        return [r for r in results]

    # Batch up to 5 at a time
    batch_size = 5
    for chunk_start in range(0, len(needs_llm), batch_size):
        chunk = needs_llm[chunk_start:chunk_start + batch_size]
        indices, batch_contents = zip(*chunk)
        try:
            res = await llm.complete_json(
                build_batch_importance_prompt(list(batch_contents)), max_tokens=200
            )
            scores = res.get("scores", [])
            for j, idx in enumerate(indices):
                results[idx] = float(scores[j]) if j < len(scores) else 5.0
        except Exception as e:
            logger.warning(f"Batch importance scoring failed: {e}. Using defaults.")
            for idx in indices:
                results[idx] = 5.0

    return [r if r is not None else 5.0 for r in results]


# ─── Memory addition ───────────────────────────────────────────────────────────

async def add_observation(llm: LLMClient, mee_id: int, content: str,
                           mee_name: str = "") -> int:
    importance = await score_importance(llm, content)
    mem_id     = await db.add_memory(mee_id, content, "observation", importance)
    if mee_name:
        upsert_memory(mee_id, mee_name, mem_id, content, importance, "observation",
                      datetime.now(timezone.utc).isoformat())
    return mem_id


async def add_conversation_memory(llm: LLMClient, mee_id: int, content: str,
                                   mee_name: str = "") -> int:
    importance = await score_importance(llm, content)
    mem_id     = await db.add_memory(mee_id, content, "conversation", importance)
    if mee_name:
        upsert_memory(mee_id, mee_name, mem_id, content, importance, "conversation",
                      datetime.now(timezone.utc).isoformat())
    return mem_id


# ─── Memory retrieval ──────────────────────────────────────────────────────────

async def retrieve_memories(llm: LLMClient, mee_id: int, mee_name: str,
                              query: str, top_k: int = 10,
                              relationships: list[dict] = None) -> list[dict]:
    results = retrieve_top_memories(mee_id, mee_name, query, top_k=top_k,
                                    relationships=relationships)
    if results:
        ids = [m["id"] for m in results]
        await db.touch_memories(ids)   # single batch UPDATE
        return results

    # Fallback: recency-based from SQLite
    all_mems = await db.get_memories(mee_id, limit=50)
    top      = all_mems[:top_k]
    if top:
        await db.touch_memories([m["id"] for m in top])
    return top


# ─── ChromaDB sync ─────────────────────────────────────────────────────────────

async def sync_memories_to_chroma(mee_id: int, mee_name: str,
                                   since_iso: str = None) -> str:
    """
    Sync memories to ChromaDB.
    If since_iso is provided, only syncs memories newer than that timestamp.
    On cold start (since_iso=None), syncs the most recent CHROMA_COLD_START_DAYS days.
    Returns the ISO timestamp of the sync (for the agent to store as watermark).
    """
    if since_iso:
        mems = await db.get_memories_since(mee_id, since_iso, limit=300)
    else:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=CHROMA_COLD_START_DAYS)).isoformat()
        mems   = await db.get_memories_since(mee_id, cutoff, limit=300)

    for m in mems:
        upsert_memory(mee_id, mee_name, m["id"], m["content"],
                      m["importance"], m["memory_type"], m["created_at"])

    if mems:
        logger.info(f"[{mee_name}] Synced {len(mems)} memories to ChromaDB")

    return datetime.now(timezone.utc).isoformat()


# ─── HyDE-lite retrieval query ────────────────────────────────────────────────

def build_retrieval_query(mee_name: str, recent_chat: list[dict],
                           pending_addressed: list[dict] = None,
                           agenda: list[str] = None) -> str:
    """
    Diversified retrieval query using THREE sources to prevent topic fixation:
      1. Last human message / pending addressed (contextual relevance)
      2. A random older message from recent chat (topic diversity)
      3. A random agenda item (character-driven, agenda-first)

    Also has a 20% chance to skip the latest message entirely and query
    only from the agenda — ensuring the character follows their own plans
    rather than endlessly reacting to what was last said.
    """
    # 20% chance: ignore recent chat entirely, query from agenda only
    if (not pending_addressed and agenda
            and random.random() < 0.20):
        plan_item = random.choice(agenda)
        return (
            f"{mee_name} is thinking about their own plans today: "
            f"\"{plan_item[:120]}\" — what memories come to mind?"
        )

    parts = []

    # Source 1: Pending addressed or last human message (contextual)
    if pending_addressed:
        focus = pending_addressed[-1]["content"]
        parts.append(f"responding to: \"{focus[:180]}\"")
    else:
        human_msgs = [m for m in recent_chat[-8:] if not m.get("is_mee")]
        if human_msgs:
            focus = human_msgs[-1]["content"]
            parts.append(f"recent: \"{focus[:150]}\"")

            # Source 2: Random older different message (diversity)
            if len(human_msgs) >= 3:
                older = random.choice(human_msgs[:-1])
                if older["content"] != focus:
                    parts.append(f"also thinking about: \"{older['content'][:100]}\"")
        elif recent_chat:
            parts.append(f"recent: \"{recent_chat[-1]['content'][:150]}\"")

    # Source 3: Agenda-based (character-driven)
    if agenda:
        plan_item = random.choice(agenda)
        parts.append(f"my plans today: \"{plan_item[:100]}\"")

    if parts:
        return (
            f"{mee_name} is thinking about "
            + " | ".join(parts)
            + " — what memories come to mind?"
        )
    return f"{mee_name} thinking about recent events and relationships"


# ─── Reflection ────────────────────────────────────────────────────────────────

async def maybe_reflect(llm: LLMClient, mee_id: int, mee_name: str,
                         all_mee_names: list[str]) -> list[str]:
    """
    Dynamic reflection threshold:
    - Base: REFLECTION_THRESHOLD cumulative importance
    - Spike mode: if recent importance variance > 8.0, threshold drops to 40%

    Two-stage focal-point reflection (quality model):
    1. Generate 3 focal questions from recent memories
    2. Retrieve memories relevant to each question
    3. Generate insights from question-targeted memories
    Falls back to single-stage if stage 1/2 fails.
    """
    cum_importance = await db.sum_recent_importance(mee_id)
    variance       = await db.recent_importance_variance(mee_id, window=15)

    threshold = BASE_REFLECTION_THRESHOLD * 0.4 if variance > VARIANCE_SPIKE_THRESHOLD \
                else BASE_REFLECTION_THRESHOLD

    if cum_importance < threshold:
        return []

    recent = await db.get_memories(mee_id, limit=50)
    if not recent:
        return []

    mem_contents = [m["content"] for m in recent[:50]]

    # Stage 1: focal questions (quality model)
    questions = []
    try:
        q_result  = await llm.complete_json(
            build_focal_questions_prompt(mee_name, mem_contents),
            max_tokens=300, quality=True,
        )
        questions = q_result.get("questions", [])
    except Exception as e:
        logger.warning(f"[{mee_name}] Focal questions failed ({e}), falling back")

    # Stage 2: retrieve per-question, then generate insights (quality model)
    reflections = []
    if questions:
        try:
            retrieved_for_questions = []
            for q in questions:
                q_mems = await retrieve_memories(llm, mee_id, mee_name, q, top_k=5)
                retrieved_for_questions += [m["content"] for m in q_mems]

            seen   = set()
            unique = []
            for c in retrieved_for_questions:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)

            result      = await llm.complete_json(
                build_focal_insights_prompt(mee_name, questions, unique),
                max_tokens=600, quality=True,
            )
            reflections = result.get("reflections", [])
        except Exception as e:
            logger.warning(f"[{mee_name}] Focal insights failed ({e}), falling back")
            reflections = []

    # Fallback: single-stage
    if not reflections:
        try:
            result      = await llm.complete_json(
                build_reflection_prompt(mee_name, mem_contents[:20]),
                max_tokens=600,
            )
            reflections = result.get("reflections", [])
        except Exception:
            return []

    stored = []
    for reflection in reflections:
        importance = await score_importance(llm, reflection)
        importance = max(importance, 7.0)
        mem_id     = await db.add_memory(mee_id, reflection, "reflection", importance)
        upsert_memory(mee_id, mee_name, mem_id, reflection, importance,
                      "reflection", datetime.now(timezone.utc).isoformat())
        stored.append(reflection)
        logger.info(f"[{mee_name}] 💭 Reflection: {reflection[:80]}")

    return stored


async def reflect_on_conversation(
    llm: LLMClient, mee_id: int, mee_name: str,
    partner_name: str, conversation_summary: str,
) -> list[str]:
    """Trigger an immediate reflection after a conversation ends.
    Uses the quality (foreground) model for deeper insight.
    Returns a list of stored reflection strings (typically 1)."""
    try:
        result = await llm.complete_json(
            build_conversation_reflection_prompt(mee_name, partner_name, conversation_summary),
            max_tokens=300, quality=True,
        )
        reflections = result.get("reflections", [])
    except Exception:
        reflections = []

    stored = []
    for reflection in reflections:
        importance = await score_importance(llm, reflection)
        importance = max(importance, 7.0)
        mem_id = await db.add_memory(mee_id, reflection, "reflection", importance)
        upsert_memory(mee_id, mee_name, mem_id, reflection, importance,
                       "reflection", datetime.now(timezone.utc).isoformat())
        stored.append(reflection)
        logger.info(f"[{mee_name}] 💭 Conversation reflection: {reflection[:80]}")
    return stored
