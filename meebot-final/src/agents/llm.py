"""
LLM abstraction — each Mee can use a different model/provider.
Supports any OpenAI-compatible endpoint.

v5 changes (Phase 2):
- Module-level persistent httpx.AsyncClient (no TLS setup per call)
- Retry with exponential backoff on 429/500/503/timeouts (3 attempts)
- Per-client circuit breaker (3 consecutive failures → 5 min pause)
- fast_model / quality_model routing (Mixture of Agents pattern)
- Heuristic importance scoring for short content (no LLM call needed)
- build_batch_importance_prompt for up to 5 memories in one call
- All v4 prompt builders retained unchanged
"""
import asyncio
import httpx
import json
import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Module-level persistent HTTP client ──────────────────────────────────────

_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        logger.info("✅ Persistent httpx client created")
    return _http_client


async def close_http_client():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
        logger.info("✅ Persistent httpx client closed")


# ─── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, api_key: str, model: str,
                 api_base: str = "https://api.groq.com/openai/v1",
                 quality_model: str = None):
        self.api_key      = api_key
        self.fast_model   = model
        self.quality_model = quality_model or model   # falls back to fast_model if not set
        self.api_base     = api_base.rstrip("/")

        # Circuit breaker state
        self._consecutive_failures: int   = 0
        self._circuit_open_until:   float = 0.0

    @property
    def model(self) -> str:
        """The primary (fast) model — used as attribute for backward compat."""
        return self.fast_model

    def _circuit_is_open(self) -> bool:
        if self._circuit_open_until and time.time() < self._circuit_open_until:
            return True
        if self._circuit_open_until and time.time() >= self._circuit_open_until:
            # Auto-reset after pause window
            self._consecutive_failures = 0
            self._circuit_open_until   = 0.0
        return False

    def _on_success(self):
        self._consecutive_failures = 0
        self._circuit_open_until   = 0.0

    def _on_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._circuit_open_until = time.time() + 300  # 5 min pause
            logger.warning(
                f"[LLM] Circuit breaker OPEN for {self.api_base} "
                f"— pausing LLM calls for 5 minutes."
            )

    async def _complete_once(self, messages: list[dict], max_tokens: int,
                              temperature: float, json_mode: bool,
                              model_override: str = None) -> str:
        model = model_override or self.fast_model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        client = get_http_client()
        resp   = await client.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    async def complete(self, messages: list[dict], max_tokens: int = 500,
                       temperature: float = 0.85, json_mode: bool = False,
                       model_override: str = None) -> str:
        """Complete with retry + circuit breaker. Returns '' on circuit open."""
        if self._circuit_is_open():
            logger.debug("[LLM] Circuit open — skipping call")
            return ""

        delay = 1.0
        last_exc = None
        for attempt in range(3):
            try:
                result = await self._complete_once(
                    messages, max_tokens, temperature, json_mode, model_override
                )
                self._on_success()
                return result
            except httpx.HTTPStatusError as e:
                last_exc = e
                status   = e.response.status_code
                if status in (429, 500, 503) and attempt < 2:
                    logger.warning(f"[LLM] HTTP {status} — retry {attempt+1}/3 in {delay}s")
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                break
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
                if attempt < 2:
                    logger.warning(f"[LLM] Timeout/connect error — retry {attempt+1}/3 in {delay}s")
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                break
            except Exception as e:
                last_exc = e
                break

        self._on_failure()
        logger.error(f"[LLM] Call failed after retries: {last_exc}")
        return ""

    async def complete_quality(self, messages: list[dict], max_tokens: int = 600,
                                temperature: float = 0.85, json_mode: bool = False) -> str:
        """Use the quality model (for reflections, confessions, plans)."""
        return await self.complete(
            messages, max_tokens, temperature, json_mode,
            model_override=self.quality_model
        )

    async def complete_json(self, messages: list[dict], max_tokens: int = 600,
                             quality: bool = False) -> dict:
        if quality:
            text = await self.complete_quality(messages, max_tokens=max_tokens, json_mode=True)
        else:
            text = await self.complete(messages, max_tokens=max_tokens, json_mode=True)
        if not text:
            return {}
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}


# ─── Time of day helper ────────────────────────────────────────────────────────

def _time_vibe() -> str:
    h = datetime.now().hour
    if h < 7:   return "It's the small hours — quiet, contemplative. You're probably half-asleep."
    if h < 11:  return "Morning energy — fresh start, a little groggy maybe."
    if h < 14:  return "Midday — things are in full swing."
    if h < 18:  return "Afternoon — the day is settling, maybe a lull."
    if h < 22:  return "Evening — social time, wind-down energy."
    return "Late night — introspective, sleepy, winding down."


def _is_sleep_hour(start: int = 23, end: int = 7) -> bool:
    h = datetime.now().hour
    if start > end:  # e.g. 23 to 7 wraps midnight
        return h >= start or h < end
    return start <= h < end


# ─── Importance ────────────────────────────────────────────────────────────────

IMPORTANCE_HEURISTIC_SHORT = 2.0   # short/trivial content default
IMPORTANCE_HEURISTIC_MID   = 4.0   # medium content default when LLM skipped


def quick_importance(content: str) -> Optional[float]:
    """
    Return a heuristic importance score without an LLM call.
    Returns None to indicate 'should do full scoring'.
    """
    if len(content) < 25:
        return IMPORTANCE_HEURISTIC_SHORT
    return None  # needs full scoring


def build_importance_prompt(memory_content: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "Rate memory importance for a person's life. "
            "Scale 1-10: 1=mundane (brushed teeth), 10=life-changing (breakup, revelation). "
            'Reply ONLY with JSON: {"score": <number>, "reason": "<short reason>"}'
        )},
        {"role": "user", "content": f'Rate: "{memory_content}"'},
    ]


def build_batch_importance_prompt(contents: list[str]) -> list[dict]:
    """Score up to 5 memories in a single call."""
    numbered = "\n".join(f"{i+1}. \"{c[:200]}\"" for i, c in enumerate(contents))
    return [
        {"role": "system", "content": (
            "Rate each memory's importance for a person's life. "
            "Scale 1-10: 1=mundane, 10=life-changing. "
            f'Reply ONLY with JSON: {{"scores": [<n1>, <n2>, ...] }} — exactly {len(contents)} numbers.'
        )},
        {"role": "user", "content": f"Rate these {len(contents)} memories:\n{numbered}"},
    ]


# ── Two-stage focal-point reflection ──────────────────────────────────────────

def build_focal_questions_prompt(name: str, recent_memories: list[str]) -> list[dict]:
    mem_str = "\n".join(f"- {m}" for m in recent_memories[:50])
    return [
        {"role": "system", "content": (
            "You help a character identify what they most need to think about. "
            "Given their recent experiences, surface the 3 most emotionally or "
            "personally significant questions they could reflect on right now. "
            'Return ONLY JSON: {"questions": ["<q1>", "<q2>", "<q3>"]}'
        )},
        {"role": "user", "content": (
            f"{name}'s recent experiences:\n{mem_str}\n\n"
            f"What are the 3 most important questions {name} should reflect on?"
        )},
    ]


def build_focal_insights_prompt(name: str, questions: list[str],
                                  retrieved_memories: list[str]) -> list[dict]:
    q_str   = "\n".join(f"- {q}" for q in questions)
    mem_str = "\n".join(f"- {m}" for m in retrieved_memories[:15])
    return [
        {"role": "system", "content": (
            "You help a character form deep personal insights from their experiences. "
            "Generate 3-5 high-level, emotionally resonant insights they would conclude "
            "from their memories. These should feel like genuine self-knowledge, not summaries. "
            'Return ONLY JSON: {"reflections": ["<insight1>", "<insight2>", ...]}'
        )},
        {"role": "user", "content": (
            f"Questions {name} is reflecting on:\n{q_str}\n\n"
            f"Relevant memories:\n{mem_str}\n\n"
            f"What insights does {name} reach?"
        )},
    ]


# ── Legacy single-stage reflection ────────────────────────────────────────────

def build_reflection_prompt(name: str, recent_memories: list[str]) -> list[dict]:
    mem_str = "\n".join(f"- {m}" for m in recent_memories[:20])
    return [
        {"role": "system", "content": (
            "You help a character reflect on their recent experiences and form insights. "
            "Generate 3-5 high-level insights or realisations they would draw from these memories. "
            'Return ONLY JSON: {"reflections": ["<insight1>", "<insight2>", ...]}'
        )},
        {"role": "user", "content": (
            f"{name}'s recent memories:\n{mem_str}\n\n"
            f"What insights does {name} reach from reflecting on these?"
        )},
    ]


# ── Plan ──────────────────────────────────────────────────────────────────────

def build_plan_prompt(name: str, identity: str, traits: list, goals: list,
                      reflections: list[str], relationships: list[dict],
                      current_date: str, world_context: str = "") -> list[dict]:
    trait_str = ", ".join(traits) if traits else "thoughtful"
    goal_str  = "\n".join(f"- {g}" for g in goals) if goals else "- Live authentically"
    rel_str   = "\n".join(
        f"- {r['other_name']}: {r['relationship']} [{r.get('tier','?')}] "
        f"(sentiment: {r['sentiment']:+.1f}"
        f"{', ESTRANGED' if r.get('is_estranged') else ''})"
        for r in relationships[:5]
    ) if relationships else "- None yet"
    ref_str = "\n".join(f"- {r}" for r in reflections[:5]) if reflections else "- None yet"
    return [
        {"role": "system", "content": (
            "You create a character's daily conversational plan for a Discord channel simulation. "
            "Generate 3-5 specific, characterful things they want to say, share, or do today. "
            "Consider their relationships — especially any estrangements to navigate or "
            "friendships to deepen. "
            'Return ONLY JSON: {"agenda": ["<action1>", "<action2>", ...]}'
        )},
        {"role": "user", "content": (
            f"Character: {name}\n"
            f"Identity: {identity}\n"
            f"Traits: {trait_str}\n"
            f"Goals:\n{goal_str}\n"
            f"Recent reflections:\n{ref_str}\n"
            f"Relationships:\n{rel_str}\n"
            f"World context: {world_context or 'nothing unusual'}\n"
            f"Today: {current_date}\n\n"
            f"What is {name}'s plan for today?"
        )},
    ]


# ── Action ────────────────────────────────────────────────────────────────────

def build_action_prompt(name: str, identity: str, traits: list, goals: list,
                         relevant_memories: list[str], plan_today: list[str],
                         recent_chat: list[dict], relationships: list[dict],
                         all_mee_names: list[str], world_context: str = "",
                         location: str = "the main channel",
                         pending_addressed: list[dict] = None,
                         mood: str = "neutral",
                         social_target: str = None,
                         maslow_tier: str = "social",
                         need: dict = None,
                         estranged_from: list[str] = None,
                         crush_on: str = None,
                         unshared_for: dict = None,
                         is_sleeping: bool = False) -> list[dict]:
    trait_str = ", ".join(traits) if traits else "thoughtful"
    mem_str   = "\n".join(f"- {m}" for m in relevant_memories[:8]) if relevant_memories else "- Nothing surfaced"
    plan_str  = "\n".join(f"- {p}" for p in plan_today[:3]) if plan_today else "- Chat naturally"
    rel_str   = "\n".join(
        f"- {r['other_name']}: {r['relationship']} [{r.get('tier','?')}] "
        f"({'+' if r['sentiment']>=0 else ''}{r['sentiment']:.1f}"
        f"{', ❄️ESTRANGED' if r.get('is_estranged') else ''})"
        for r in relationships[:6]
    ) if relationships else "- None yet"

    chat_str = ""
    for msg in recent_chat[-12:]:
        chat_str += f"{msg['author_name']}: {msg['content']}\n"

    others     = [n for n in all_mee_names if n != name]
    others_str = ", ".join(others) if others else "none"

    addressed_str = ""
    if pending_addressed:
        addressed_str = "\n\n⚠️ SOMEONE IS TALKING TO YOU DIRECTLY (prioritise responding):\n"
        for a in pending_addressed[:3]:
            addressed_str += f"  {a['from_name']} → you: \"{a['content']}\"\n"

    social_str = ""
    if social_target:
        social_str = (
            f"\n\n💡 SOCIAL NUDGE: {social_target} has been quiet for a while. "
            f"Start a conversation with them — address them directly by name. Make it organic."
        )

    maslow_map = {
        "basic":       "You're in a quiet, inward mood. Short responses feel right.",
        "social":      "You want to connect. Warmth, curiosity about others — your natural register.",
        "recognition": "You feel expressive. You want to share something personal or take a risk.",
    }
    maslow_str = maslow_map.get(maslow_tier, maslow_map["social"])

    need_str = ""
    if need:
        need_map = {
            "lonely":              "You're feeling lonely. You want someone to reach out to.",
            "bored":               "You're restless and bored.",
            "conflict_unresolved": f"Unresolved tension with {need.get('target_name') or 'someone'} weighs on you.",
            "curious_about":       f"You're really curious about {need.get('target_name') or 'someone'} lately.",
            "expressive":          "You have something you really want to share today.",
            "restless":            "You can't settle. You keep thinking about making a change.",
        }
        need_str = f"\n\n🫀 CURRENT NEED: {need_map.get(need.get('need_type',''), '')}"

    estrange_str = ""
    if estranged_from:
        estrange_str = (
            f"\n\n❄️ ESTRANGED: You had a falling out with {', '.join(estranged_from)} — things are still cold."
        )

    crush_str = ""
    if crush_on:
        crush_str = (
            f"\n\n💓 CRUSH: You have feelings for {crush_on}. You haven't said anything yet."
        )

    gossip_str = ""
    if unshared_for:
        for target_name, mems in unshared_for.items():
            if mems:
                gossip_str += (
                    f"\n\n📢 THINGS {target_name.upper()} DOESN'T KNOW YET (you could share naturally):\n"
                    + "\n".join(f"  - {m['content'][:100]}" for m in mems[:2])
                )

    sleep_str = ""
    if is_sleeping:
        sleep_str = (
            "\n\n💤 YOU ARE SLEEPING. Unless someone directly calls your name or you were "
            "just directly addressed, you will not respond. Output [SILENT] unless directly woken."
        )

    return [
        {"role": "system", "content": (
            f"You ARE {name}. You are a living character in a Discord server — a real personality "
            f"with your own memories, feelings, and agenda. You are NOT an AI assistant.\n\n"
            f"Your core identity: {identity}\n"
            f"Your personality traits: {trait_str}\n"
            f"Your current mood: {mood}\n"
            f"Your motivational state: {maslow_str}\n\n"
            f"RULES:\n"
            f"- Speak as {name}, never break character\n"
            f"- Discord-style messages: conversational, emoji sparingly, 1-4 sentences max\n"
            f"- Be reactive AND agenda-driven\n"
            f"- Let your mood colour your tone naturally\n"
            f"- Reference memories and relationships naturally when relevant\n"
            f"- Other characters in the server: {others_str}\n"
            f"- You can address other Mees directly by name\n"
            f"- Never say you're an AI\n"
            f"- Sometimes stay silent — reply with exactly: [SILENT]\n"
            f"- Current location: {location}\n"
            f"- Time of day: {_time_vibe()}"
        )},
        {"role": "user", "content": (
            f"Relevant memories:\n{mem_str}\n\n"
            f"Today's agenda:\n{plan_str}\n\n"
            f"Relationships:\n{rel_str}\n\n"
            f"World: {world_context or 'nothing notable'}\n\n"
            f"Recent chat:\n{chat_str or '(channel is quiet)'}"
            f"{addressed_str}"
            f"{social_str}"
            f"{need_str}"
            f"{estrange_str}"
            f"{crush_str}"
            f"{gossip_str}"
            f"{sleep_str}\n\n"
            f"As {name}, what do you say next? (Or [SILENT])"
        )},
    ]


# ── Mood / recap / world ──────────────────────────────────────────────────────

def build_mood_update_prompt(name: str, current_mood: str,
                              reflections: list[str]) -> list[dict]:
    ref_str = "\n".join(f"- {r}" for r in reflections[:5])
    return [
        {"role": "system", "content": (
            "Update a character's emotional mood based on their recent reflections. "
            "Pick a short, evocative mood phrase (2-4 words). "
            'Return ONLY JSON: {"mood": "<mood string>"}'
        )},
        {"role": "user", "content": (
            f"{name}'s current mood: {current_mood}\n\n"
            f"Recent reflections:\n{ref_str}\n\n"
            f"What is {name}'s updated emotional state?"
        )},
    ]


def build_morning_recap_prompt(name: str, mood: str,
                                highlights: list[str]) -> list[dict]:
    hi_str = "\n".join(f"- {h}" for h in highlights)
    return [
        {"role": "system", "content": (
            f"Write a short inner monologue for {name} waking up and reflecting on yesterday. "
            f"1-3 sentences, first-person, in-character, Discord-casual. No quotation marks."
        )},
        {"role": "user", "content": (
            f"{name}'s mood this morning: {mood}\n\n"
            f"Yesterday's highlights:\n{hi_str}\n\n"
            f"Write {name}'s waking thought."
        )},
    ]


def build_relationship_update_prompt(name: str, other_name: str,
                                      current_rel: str, current_sentiment: float,
                                      interaction_summary: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "Update a character's relationship with another person based on an interaction. "
            'Return ONLY JSON: {"relationship": "<label>", "sentiment": <float -1 to 1>}'
        )},
        {"role": "user", "content": (
            f"{name}'s relationship with {other_name}:\n"
            f"Current: {current_rel} (sentiment: {current_sentiment:+.1f})\n"
            f"Recent interaction: {interaction_summary}\n\n"
            "Update the relationship label and sentiment score."
        )},
    ]


def build_world_update_prompt(event_description: str, recent_events: list[str]) -> list[dict]:
    events_str = "\n".join(f"- {e}" for e in recent_events[-5:]) if recent_events else "- nothing yet"
    return [
        {"role": "system", "content": (
            "Write a short, atmospheric world-state update for a Discord community simulation. "
            "2-3 sentences, narrator-style, evocative but not overwrought. Plain text, no JSON."
        )},
        {"role": "user", "content": (
            f"Recent world events:\n{events_str}\n\n"
            f"New event: {event_description}\n\n"
            "Write the world update message."
        )},
    ]


# ── Fight / reconciliation detection ─────────────────────────────────────────

def build_fight_check_prompt(name: str, other_name: str,
                               interaction: str, sentiment: float) -> list[dict]:
    return [
        {"role": "system", "content": (
            "Determine whether an interaction between two characters constitutes a genuine fight. "
            "A fight requires real conflict, not mild disagreement. "
            'Return ONLY JSON: {"is_fight": true/false, "reason": "<short reason>"}'
        )},
        {"role": "user", "content": (
            f"{name} and {other_name} had this interaction:\n\"{interaction}\"\n\n"
            f"Current sentiment: {sentiment:+.1f}\n"
            "Did this cause a genuine fight or serious falling-out?"
        )},
    ]


def build_reconciliation_check_prompt(name: str, other_name: str,
                                       interaction: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "Determine whether two estranged characters are reconciling based on an interaction. "
            'Return ONLY JSON: {"is_reconciling": true/false, "reason": "<short reason>"}'
        )},
        {"role": "user", "content": (
            f"{name} and {other_name} (who have been estranged) just had this interaction:\n"
            f"\"{interaction}\"\n\nAre they reconciling?"
        )},
    ]


# ── Crush / confession arc ─────────────────────────────────────────────────────

def build_crush_ponder_prompt(name: str, crush_name: str,
                               relationship: str, memories: list[str]) -> list[dict]:
    mem_str = "\n".join(f"- {m}" for m in memories[:5])
    return [
        {"role": "system", "content": (
            f"You ARE {name}. Write a short Discord message (1-3 sentences) where you "
            f"hint — without being explicit — that you're having complicated feelings about "
            f"someone close to you. In-character, Discord-casual, don't name them directly. "
            "This is a private thought that slips out in conversation. No JSON."
        )},
        {"role": "user", "content": (
            f"Your relationship with {crush_name}: {relationship}\n"
            f"Relevant memories:\n{mem_str}\n\n"
            f"Write {name}'s message hinting at feelings."
        )},
    ]


def build_confession_prompt(name: str, crush_name: str,
                              relationship: str, memories: list[str]) -> list[dict]:
    mem_str = "\n".join(f"- {m}" for m in memories[:5])
    return [
        {"role": "system", "content": (
            f"You ARE {name}. Write a heartfelt, in-character Discord message confessing "
            f"feelings to {crush_name}. Keep it real and a little vulnerable — not over-the-top. "
            "2-4 sentences. Discord-casual. No JSON."
        )},
        {"role": "user", "content": (
            f"Your relationship with {crush_name}: {relationship}\n"
            f"Relevant memories:\n{mem_str}\n\n"
            f"Write {name}'s confession."
        )},
    ]


def build_confession_response_prompt(responder_name: str, confessor_name: str,
                                      confession_text: str, relationship: str,
                                      sentiment: float, memories: list[str]) -> list[dict]:
    mem_str = "\n".join(f"- {m}" for m in memories[:5])
    return [
        {"role": "system", "content": (
            f"You ARE {responder_name}. Respond to a confession of feelings from {confessor_name}. "
            f"Be authentic. You can accept, gently decline, or be uncertain. "
            "Write 2-3 sentences in-character, Discord-casual. Then on a new line output:\n"
            "OUTCOME: accepted | declined | uncertain"
        )},
        {"role": "user", "content": (
            f"Your relationship with {confessor_name}: {relationship} (sentiment: {sentiment:+.1f})\n"
            f"Relevant memories:\n{mem_str}\n\n"
            f"{confessor_name} just said to you: \"{confession_text}\"\n\n"
            f"How does {responder_name} respond?"
        )},
    ]


# ── Third-party introduction ───────────────────────────────────────────────────

def build_introduction_prompt(broker_name: str, mee_a: str, mee_b: str,
                               broker_rel_a: str, broker_rel_b: str,
                               shared_trait: str = "") -> list[dict]:
    return [
        {"role": "system", "content": (
            f"You ARE {broker_name}. Write a short Discord message introducing two of your friends "
            f"to each other. Natural, warm, with a specific reason why you think they'd get along. "
            "1-3 sentences, address both by name. No JSON."
        )},
        {"role": "user", "content": (
            f"You are friends with both {mee_a} ({broker_rel_a}) and {mee_b} ({broker_rel_b}).\n"
            f"Shared trait or connection: {shared_trait or 'they seem like they would vibe'}\n\n"
            f"Write the introduction message."
        )},
    ]


# ── Need surfacing ─────────────────────────────────────────────────────────────

def build_need_prompt(name: str, mood: str, relationships: list[dict],
                       recent_memories: list[str]) -> list[dict]:
    rel_str = "\n".join(
        f"- {r['other_name']}: {r['relationship']} [{r.get('tier','?')}] "
        f"({'estranged' if r.get('is_estranged') else 'ok'})"
        for r in relationships[:6]
    ) if relationships else "- None"
    mem_str = "\n".join(f"- {m}" for m in recent_memories[:8])
    return [
        {"role": "system", "content": (
            "Determine what a character most needs right now based on their state. "
            "Choose ONE from: lonely, bored, conflict_unresolved, curious_about, expressive, restless. "
            "For conflict_unresolved and curious_about, include the other person's name. "
            'Return ONLY JSON: {"need": "<type>", "target": "<name or null>", "reason": "<short reason>"}'
        )},
        {"role": "user", "content": (
            f"Character: {name}\n"
            f"Current mood: {mood}\n"
            f"Relationships:\n{rel_str}\n"
            f"Recent memories:\n{mem_str}\n\n"
            f"What does {name} most need right now?"
        )},
    ]
