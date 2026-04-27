"""MeeAgent v5 — Tomodachi Life × Smallville × AgentSociety generative agent."""

import asyncio
import logging
import os
import random
import re
import time
from datetime import datetime, timezone, date
from typing import Optional

from src.agents.llm import (
    LLMClient,
    GEMINI_API_BASE,
    build_stop_words,
    build_plan_prompt,
    build_action_prompt,
    build_relationship_update_prompt,
    build_mood_update_prompt,
    build_morning_recap_prompt,
    build_fight_check_prompt,
    build_reconciliation_check_prompt,
    build_crush_ponder_prompt,
    build_confession_prompt,
    build_confession_response_prompt,
    build_introduction_prompt,
    build_need_prompt,
    _is_sleep_hour,
)
from src.agents.memory import (
    retrieve_memories,
    add_observation,
    add_conversation_memory,
    maybe_reflect,
    reflect_on_conversation,
    sync_memories_to_chroma,
    build_retrieval_query,
)
from src.utils import db

logger = logging.getLogger(__name__)

MAX_CALLS_PER_TICK         = int(os.getenv("MAX_CALLS_PER_TICK", "14"))
SLEEP_HOUR_START           = int(os.getenv("SLEEP_HOUR_START", "23"))
SLEEP_HOUR_END             = int(os.getenv("SLEEP_HOUR_END", "7"))
WANDER_COOLDOWN_MINUTES    = int(os.getenv("WANDER_COOLDOWN_MINUTES", "30"))
EXHAUSTION_WAKE_MINUTES    = int(os.getenv("EXHAUSTION_WAKE_MINUTES", "8"))
QUALITY_MODEL              = os.getenv("QUALITY_MODEL", "llama-3.3-70b-versatile").strip()

DEFAULT_LOCATIONS = ["the café", "the park", "the library", "the rooftop", "their room",
                     "the garden", "the couch", "the balcony", "the kitchen", "the town square"]

WANDER_CHANCE             = 0.06
SOCIAL_INITIATIVE_MINUTES = 12
INTRO_CHANCE_PER_TICK     = 0.04
NEED_CHANCE_PER_TICK      = 0.10
CONFESSION_CHANCE         = 0.18
PONDER_CHANCE             = 0.35

# --- Activity System ---
# Agents perform short, flavourful activities between conversations or wander ticks.
# Each entry: (description, emoji, min_ticks, max_ticks)
ACTIVITIES: list[tuple[str, str, int, int]] = [
    ("reading a book",           "📖", 2, 4),
    ("daydreaming by the window","🤔", 2, 3),
    ("writing in their journal", "✍️", 2, 4),
    ("sketching in a notebook",  "🎨", 3, 5),
    ("practicing a tune",        "🎵", 2, 4),
    ("tending to some plants",   "🌱", 2, 3),
    ("stargazing",               "🌟", 2, 3),
    ("playing a handheld game",  "🎮", 2, 4),
    ("sipping tea and watching", "☕", 2, 3),
    ("organising their space",   "📦", 3, 5),
]
ACTIVITY_CHANCE             = 0.40  # Probability of starting an activity when idle
ACTIVITY_POST_COOLDOWN_SECS = 900   # Min seconds between activity world-updates per agent (15 min)

# ─── Conversation State Machine ─────────────────────────────────────────────────
# Formal Mee-to-Mee conversation lifecycle with max messages and timeout,
# adapted from AI Town's Conversation class.
CONVERSATION_MAX_MINUTES  = 10  # Max duration of a Mee-to-Mee conversation
CONVERSATION_MAX_MESSAGES = 8   # Max messages per conversation pair


_SENTIMENT_KEYWORDS = frozenset({
    "love", "hate", "angry", "miss", "sorry", "afraid", "happy", "sad",
    "hurt", "wonderful", "terrible", "wonderful", "heart", "cry", "crying",
    "laugh", "laughing", "scared", "jealous", "proud", "trust", "upset",
    "worried", "excited", "annoyed", "grateful", "thankful",
})


def _build_fg_client(mee_data: dict, bg_client: LLMClient) -> LLMClient:
    """Build foreground (quality) client.

    Priority:
    1. Gemini key (AIza prefix) configured → use Gemini with the Gemini model.
    2. Custom key (non-AIza, e.g. a separate Groq key) → use that key with
       the stored model on the bg_client's API base (Groq).
    3. No key configured → use the same Groq API key as the background with
       QUALITY_MODEL (default: llama-3.3-70b-versatile).
       This eliminates the need for a second API key entirely.
    """
    fg_key   = mee_data.get("gemini_api_key") or ""
    fg_model = mee_data.get("gemini_model") or ""
    if fg_key:
        # AIza → Gemini (backward compat)
        if fg_key.startswith("AIza"):
            return LLMClient(
                api_key=fg_key,
                model=fg_model or "gemini-2.0-flash",
                api_base=GEMINI_API_BASE,
            )
        # Non-AIza → treat as a Groq/OpenAI-compat key override
        return LLMClient(
            api_key=fg_key,
            model=fg_model or QUALITY_MODEL,
            api_base=bg_client.api_base,
            quality_model=fg_model or QUALITY_MODEL,
        )
    # All-Groq: foreground uses the quality model on the same Groq key
    return LLMClient(
        api_key=bg_client.api_key,
        model=QUALITY_MODEL,
        api_base=bg_client.api_base,
        quality_model=QUALITY_MODEL,
    )


def _day_rhythm_factor(hour: int) -> float:
    """Tick-interval multiplier by time-of-day. Sleep hours handled upstream."""
    if _is_sleep_hour(SLEEP_HOUR_START, SLEEP_HOUR_END):
        return 8.0
    if 7  <= hour < 9:   return 1.1
    if 9  <= hour < 12:  return 0.95
    if 12 <= hour < 14:  return 0.85
    if 14 <= hour < 17:  return 1.25
    if 17 <= hour < 19:  return 0.90
    if 19 <= hour < 22:  return 0.70
    return 1.50


def _maslow_tier(mood: str, need: Optional[dict] = None) -> str:
    if need:
        need_type = need.get("need_type", "")
        if need_type in ("expressive", "restless"):
            return "recognition"
        if need_type in ("lonely", "curious_about", "bored"):
            return "social"
        if need_type == "conflict_unresolved":
            return "basic"

    mood_lower = mood.lower()
    if any(w in mood_lower for w in ("bold", "expressive", "energis", "confident", "excite")):
        return "recognition"
    if any(w in mood_lower for w in ("quiet", "contemplat", "melanchol", "withdrawn", "tired")):
        return "basic"
    return "social"


class MeeAgent:
    """Generative agent with LLM-driven autonomy, social arcs, memory, and daily rhythm."""

    def __init__(self, mee_data: dict) -> None:
        self.id:          int  = mee_data["id"]
        self.name:        str  = mee_data["name"]
        self.identity:    str  = mee_data["identity"]
        self.traits:      list = mee_data["traits"]
        self.goals:       list = mee_data["goals"]
        self.image_url:   str  = mee_data["image_url"]
        self.channel_id:  str  = mee_data["channel_id"]
        self.webhook_url: str  = mee_data.get("webhook_url") or ""
        self.location:    str  = mee_data.get("location") or "the main channel"
        self.mood:        str  = mee_data.get("mood") or "neutral"

        # ── PAD emotional state (Pleasure, Arousal, Dominance) ──────────────────
        self.pleasure   = float(mee_data.get("pleasure") or 0.0)
        self.arousal    = float(mee_data.get("arousal") or 0.0)
        self.dominance  = float(mee_data.get("dominance") or 0.0)

        # ── Interrupt-driven drafting tracker ──────────────────────────────────
        self._drafting_task: asyncio.Task | None = None

        self.llm = LLMClient(api_key=mee_data["api_key"], model=mee_data["model"],
                             api_base=mee_data.get("api_base", "https://api.groq.com/openai/v1"),
                             quality_model=mee_data.get("quality_model"))
        self.llm_fg = _build_fg_client(mee_data, self.llm)

        self._today_plan:       list[str]  = []
        self._plan_date:        str        = ""
        self._chroma_watermark: str | None = None
        self._calls_this_tick:  int        = 0
        self._last_wander_time: float      = 0.0
        self._exhausted:        bool       = False
        self._exhausted_at:     float      = 0.0
        self._silent_ticks:     int        = 0
        self._activity: Optional[dict] = None  # {"desc": str, "emoji": str, "ticks_left": int}
        self._last_activity_post_time: float = 0.0
        self._conversation_id: Optional[int] = None  # Active conversation ID, if any
        self._v6_agent_id: str | None = None  # UUID in the v6 agents table


    @property
    def is_drafting(self) -> bool:
        """True while this agent is in the middle of an LLM generation call."""
        return self._drafting_task is not None and not self._drafting_task.done()

    def _budget_ok(self) -> bool:
        """True if within the per-tick LLM call budget."""
        return self._calls_this_tick < MAX_CALLS_PER_TICK

    def _charge_budget(self, n: int = 1):
        self._calls_this_tick += n

    def _pick_activity(self) -> dict:
        """Pick a random activity with random duration in ticks."""
        desc, emoji, min_t, max_t = random.choice(ACTIVITIES)
        return {
            "desc":      desc,
            "emoji":     emoji,
            "ticks_left": random.randint(min_t, max_t),
        }

    def _activity_event(self) -> str:
        """World-update string for the current activity."""
        if not self._activity:
            return ""
        return f"{self._activity['emoji']} {self.name} is {self._activity['desc']} {self.location}."



    async def _ensure_chroma_synced(self) -> None:
        """Incremental ChromaDB sync via watermark for startup."""
        watermark = await sync_memories_to_chroma(
            self.id, self.name, since_iso=self._chroma_watermark
        )
        self._chroma_watermark = watermark

    async def reload_pad(self) -> None:
        """Reload PAD emotional state from the database (called before generation)."""
        try:
            pad = await db.get_pad(self.id)
            self.pleasure  = float(pad.get("pleasure", 0.0))
            self.arousal   = float(pad.get("arousal", 0.0))
            self.dominance = float(pad.get("dominance", 0.0))
        except Exception:
            pass  # Keep current values on error

    async def _reconcile_v6_agent(self) -> None:
        """Ensure a v6 `agents` record exists for this agent.
        
        Creates one if missing (matched by name), stores the UUID in
        self._v6_agent_id for later prompt-state injection.
        """
        try:
            existing = await db.fetch_agent_by_name(self.name)
            if existing:
                self._v6_agent_id = existing["id"]
            else:
                self._v6_agent_id = await db.create_agent(
                    name=self.name,
                    base_persona=self.identity,
                )
                logger.info(
                    f"[{self.name}] Created v6 agent record ({self._v6_agent_id})"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] v6 agent reconcile failed: {e}")
            self._v6_agent_id = None

    async def reload(self) -> None:
        data = await db.get_mee_by_id(self.id)
        if not data:
            return
        self.name        = data["name"]
        self.identity    = data["identity"]
        self.traits      = data["traits"]
        self.goals       = data["goals"]

        # Reconcile v6 agent record on reload (name may have changed)
        await self._reconcile_v6_agent()
        self.image_url   = data["image_url"]
        self.channel_id  = data["channel_id"]
        self.webhook_url = data.get("webhook_url") or ""
        self.location    = data.get("location") or "the main channel"
        self.mood        = data.get("mood") or "neutral"
        self.pleasure    = float(data.get("pleasure") or 0.0)
        self.arousal     = float(data.get("arousal") or 0.0)
        self.dominance   = float(data.get("dominance") or 0.0)
        self.llm = LLMClient(api_key=data["api_key"], model=data["model"],
                             api_base=data.get("api_base", "https://api.groq.com/openai/v1"),
                             quality_model=data.get("quality_model"))
        self.llm_fg = _build_fg_client(data, self.llm)

    async def observe(self, event: str) -> None:
        await add_observation(self.llm, self.id, event, mee_name=self.name)

    async def observe_conversation(self, author: str, content: str) -> None:
        await add_conversation_memory(self.llm, self.id,
                                      f"{author} said: \"{content}\"", mee_name=self.name)

    def _should_update_relationship(self, content: str, other_name: str) -> bool:
        """Gate: only fire LLM relationship updates on emotionally meaningful messages."""
        lowered = content.lower()
        if other_name.lower() in lowered or self.name.lower() in lowered:
            return True
        if any(w in lowered for w in _SENTIMENT_KEYWORDS):
            return True
        return random.random() < 0.15

    async def ensure_plan(self, all_mee_names: list[str]) -> list[str]:
        today = date.today().isoformat()
        if self._plan_date == today and self._today_plan:
            return self._today_plan

        existing = await db.get_plan(self.id, today)
        if existing:
            self._today_plan = existing
            self._plan_date  = today
            return self._today_plan

        relationships = await db.get_relationships(self.id)
        world_events  = await db.get_recent_world_events(limit=5)
        world_context = "; ".join(e["content"] for e in world_events) if world_events else ""
        reflections   = [
            m["content"] for m in (await db.get_memories(self.id, limit=50))
            if m["memory_type"] == "reflection"
        ][:5]

        try:
            result = await self.llm.complete_json(
                build_plan_prompt(
                    self.name, self.identity, self.traits, self.goals,
                    reflections, relationships, today, world_context,
                ),
                max_tokens=400, quality=True,
            )
            self._charge_budget(1)
            agenda = result.get("agenda", ["Chat naturally in the channel"])
        except Exception as e:
            logger.warning(f"[{self.name}] Plan generation failed: {e}")
            agenda = ["Chat naturally and be myself today"]

        await db.save_plan(self.id, today, agenda)
        self._today_plan = agenda
        self._plan_date  = today
        return agenda

    async def maybe_reflect(self, all_mee_names: list[str]) -> list[str]:
        """Two-stage reflection (~3-5 LLM calls); charge budget accordingly."""
        reflections = await maybe_reflect(self.llm, self.id, self.name, all_mee_names)
        if reflections:
            self._charge_budget(min(3, len(reflections) + 1))
        return reflections

    async def check_tier_transition(self, other_name: str) -> str | None:
        rel = await db.get_relationship_with(self.id, other_name)
        if not rel:
            return None
        new_tier = db.compute_tier_from_sentiment(rel["sentiment"], rel["tier"],
                                                  bool(rel.get("is_estranged")))
        if not new_tier:
            return None
        await db.upsert_relationship(self.id, other_name, rel["relationship"], rel["sentiment"],
                                     tier=new_tier)
        tier_labels = {
            "acquaintance": f"{self.name} and {other_name} have gotten to know each other.",
            "friend":       f"{self.name} and {other_name} have become friends! 🌱",
            "close_friend": f"{self.name} and {other_name} have grown really close. 💚",
            "best_friend":  f"{self.name} and {other_name} are now best friends! 💛",
        }
        event_text = tier_labels.get(new_tier,
                     f"{self.name} and {other_name}'s relationship has deepened.")
        await db.add_world_event("relationship", event_text)
        await db.add_memory(self.id,
            f"I feel like {other_name} and I have become {new_tier.replace('_', ' ')}s. Something shifted.",
            "reflection", importance=8.0)
        other_mee = await db.get_mee(other_name)
        if other_mee:
            await db.add_memory(other_mee["id"],
                f"I think {self.name} and I have genuinely become {new_tier.replace('_', ' ')}s.",
                "reflection", importance=8.0)
        logger.info(f"[{self.name}] 💞 Tier → {new_tier} with {other_name}")
        return event_text

    async def check_fight(self, other_name: str, interaction: str) -> bool:
        rel = await db.get_relationship_with(self.id, other_name)
        if not rel or rel["sentiment"] > db.FIGHT_SENTIMENT_THRESHOLD:
            return False
        if rel.get("is_estranged"):
            return False
        try:
            result   = await self.llm.complete_json(
                build_fight_check_prompt(self.name, other_name, interaction, rel["sentiment"]),
                max_tokens=120)
            self._charge_budget()
            is_fight = result.get("is_fight", False)
        except Exception:
            is_fight = False
        if not is_fight:
            return False
        await db.upsert_relationship(self.id, other_name, rel["relationship"], rel["sentiment"],
                                     is_estranged=True)
        other_mee = await db.get_mee(other_name)
        if other_mee:
            other_rel = await db.get_relationship_with(other_mee["id"], self.name)
            if other_rel:
                await db.upsert_relationship(other_mee["id"], self.name, other_rel["relationship"],
                                             other_rel["sentiment"], is_estranged=True)
        event = f"⚡ {self.name} and {other_name} had a falling out. Things are cold between them."
        await db.add_world_event("fight", event)
        await db.add_memory(self.id,
            f"I had a real fight with {other_name}. Things feel cold now.",
            "observation", importance=9.0)
        if other_mee:
            await db.add_memory(other_mee["id"],
                f"I had a falling out with {self.name}. I don't really want to talk to them right now.",
                "observation", importance=9.0)

        logger.info(f"[{self.name}] ⚡ Fight with {other_name} → estranged")
        return True

    async def check_reconciliation(self, other_name: str, interaction: str) -> bool:
        rel = await db.get_relationship_with(self.id, other_name)
        if not rel or not rel.get("is_estranged"):
            return False

        try:
            result         = await self.llm.complete_json(
                build_reconciliation_check_prompt(self.name, other_name, interaction),
                max_tokens=120,
            )
            self._charge_budget()
            is_reconciling = result.get("is_reconciling", False)
        except Exception:
            is_reconciling = False

        if not is_reconciling:
            return False

        new_sentiment = min(1.0, rel["sentiment"] + 0.3)
        await db.upsert_relationship(
            self.id, other_name, rel["relationship"], new_sentiment, is_estranged=False
        )
        other_mee = await db.get_mee(other_name)
        if other_mee:
            other_rel = await db.get_relationship_with(other_mee["id"], self.name)
            if other_rel:
                await db.upsert_relationship(
                    other_mee["id"], self.name, other_rel["relationship"],
                    min(1.0, other_rel["sentiment"] + 0.3), is_estranged=False
                )

        event = f"💚 {self.name} and {other_name} seem to have made up. The air between them is warmer."
        await db.add_world_event("reconciliation", event)
        await db.add_memory(self.id,
            f"I think {other_name} and I are okay again. It feels like a weight lifted.",
            "reflection", importance=8.5)
        if other_mee:
            await db.add_memory(other_mee["id"],
                f"Things feel better between me and {self.name}. I'm glad we worked it out.",
                "reflection", importance=8.5)

        logger.info(f"[{self.name}] 💚 Reconciled with {other_name}")
        return True

    async def update_relationship(self, other_name: str, interaction: str) -> str | None:
        relationships = await db.get_relationships(self.id)
        existing      = next((r for r in relationships if r["other_name"] == other_name), None)
        current_rel   = existing["relationship"] if existing else "stranger"
        current_sent  = existing["sentiment"]    if existing else 0.0
        is_estranged  = bool(existing.get("is_estranged")) if existing else False

        try:
            result = await self.llm.complete_json(
                build_relationship_update_prompt(
                    self.name, other_name, current_rel, current_sent, interaction
                ),
                max_tokens=100,
            )
            self._charge_budget()
            rel  = result.get("relationship", current_rel)
            sent = float(result.get("sentiment", current_sent))
            sent = max(-1.0, min(1.0, sent))
        except Exception as e:
            logger.warning(f"[{self.name}] Relationship update failed: {e}")
            return None

        existing_full = await db.get_relationship_with(self.id, other_name)
        current_tier  = existing_full["tier"] if existing_full else "stranger"

        await db.upsert_relationship(
            self.id, other_name, rel, sent, tier=current_tier, is_estranged=is_estranged
        )

        world_event = None

        tier_event = await self.check_tier_transition(other_name)
        if tier_event:
            world_event = tier_event

        if is_estranged:
            rec = await self.check_reconciliation(other_name, interaction)
            if rec:
                world_event = world_event or f"💚 {self.name} and {other_name} made up."
        elif sent < db.FIGHT_SENTIMENT_THRESHOLD and self._budget_ok():
            fight = await self.check_fight(other_name, interaction)
            if fight:
                world_event = world_event or f"⚡ {self.name} and {other_name} fell out."

        return world_event

    async def maybe_develop_crush(self, all_mee_names: list[str]) -> str | None:
        if not self._budget_ok():
            return None
        eligible = await db.get_crush_eligible(self.id)
        if not eligible:
            return None
        crush_name = random.choice(eligible)
        if random.random() > db.CRUSH_CHANCE_PER_TICK:
            return None

        rel = await db.get_relationship_with(self.id, crush_name)
        await db.upsert_relationship(
            self.id, crush_name,
            rel["relationship"] if rel else "friend",
            rel["sentiment"]    if rel else 0.7,
            crush_on=crush_name, confession_state="pondering"
        )

        mems     = await db.get_memories(self.id, limit=30)
        relevant = [m["content"] for m in mems if crush_name in m["content"]][:5]
        try:
            msg = await self.llm.complete_quality(
                build_crush_ponder_prompt(
                    self.name, crush_name,
                    rel["relationship"] if rel else "friend",
                    relevant
                ),
                max_tokens=150, temperature=0.92,
            )
            self._charge_budget()
            msg = msg.strip().strip('"') if msg else None
        except Exception:
            msg = None

        event = f"💓 {self.name} seems to be developing feelings for someone..."
        await db.add_world_event("crush", event)
        await db.add_memory(self.id,
            f"I realised I might have feelings for {crush_name}. I'm not sure what to do.",
            "reflection", importance=8.0)

        logger.info(f"[{self.name}] 💓 Developed crush on {crush_name}")
        return msg

    async def maybe_confess(self, channel_id: str, all_agents: list) -> tuple[Optional[str], Optional[str]]:
        if not self._budget_ok():
            return None, None
        active_crushes = await db.get_active_crushes(self.id)
        if not active_crushes:
            return None, None

        crush_rel  = active_crushes[0]
        crush_name = crush_rel.get("crush_on")
        if not crush_name:
            return None, None

        conf_state = crush_rel.get("confession_state", "none")
        if conf_state not in ("pondering", "none"):
            return None, None
        if random.random() > CONFESSION_CHANCE:
            return None, None

        rel  = await db.get_relationship_with(self.id, crush_name)
        mems = await db.get_memories(self.id, limit=30)
        relevant = [m["content"] for m in mems if crush_name in m["content"]][:5]

        try:
            # Confession is direct Mee-to-Mee → use foreground LLM (Gemini)
            confession = await self.llm_fg.complete_quality(
                build_confession_prompt(
                    self.name, crush_name,
                    rel["relationship"] if rel else "friend",
                    relevant
                ),
                max_tokens=400, temperature=0.9,
            )
            self._charge_budget()
            confession = confession.strip().strip('"') if confession else None
        except Exception:
            return None, None

        if not confession:
            return None, None

        await db.upsert_relationship(
            self.id, crush_name,
            rel["relationship"] if rel else "friend",
            rel["sentiment"]    if rel else 0.7,
            confession_state="confessed"
        )

        other_agent  = next((a for a in all_agents if a.name == crush_name), None)
        response_msg = None
        outcome      = "uncertain"

        if other_agent and other_agent._budget_ok():
            other_rel  = await db.get_relationship_with(other_agent.id, self.name)
            other_mems = await db.get_memories(other_agent.id, limit=20)
            o_relevant = [m["content"] for m in other_mems if self.name in m["content"]][:5]

            try:
                # Confession response is also a direct targeted Mee-to-Mee reply → foreground
                raw_response = await other_agent.llm_fg.complete_quality(
                    build_confession_response_prompt(
                        other_agent.name, self.name, confession,
                        other_rel["relationship"] if other_rel else "friend",
                        other_rel["sentiment"]    if other_rel else 0.5,
                        o_relevant
                    ),
                    max_tokens=500, temperature=0.88,
                )
                other_agent._charge_budget()
                if raw_response:
                    lines        = raw_response.strip().splitlines()
                    outcome_line = next((l for l in reversed(lines) if l.startswith("OUTCOME:")), None)
                    if outcome_line:
                        outcome      = outcome_line.replace("OUTCOME:", "").strip()
                        response_msg = "\n".join(
                            l for l in lines if not l.startswith("OUTCOME:")
                        ).strip().strip('"')
                    else:
                        response_msg = raw_response.strip().strip('"')
            except Exception as e:
                logger.warning(f"[{other_agent.name}] Confession response failed: {e}")

            if outcome == "accepted":
                event_text = f"💕 {self.name} confessed their feelings to {crush_name} — and it was mutual!"
                await db.upsert_relationship(
                    self.id, crush_name, "sweetheart",
                    min(1.0, (rel["sentiment"] if rel else 0.7) + 0.2),
                    tier="sweetheart", confession_state="accepted"
                )
                if other_rel:
                    await db.upsert_relationship(
                        other_agent.id, self.name, "sweetheart",
                        min(1.0, (other_rel["sentiment"] or 0.5) + 0.2),
                        tier="sweetheart"
                    )
                await db.add_memory(self.id,
                    f"I told {crush_name} how I feel and they feel the same way. My heart is full.",
                    "reflection", importance=10.0)
                await db.add_memory(other_agent.id,
                    f"{self.name} told me they have feelings for me. I feel the same.",
                    "reflection", importance=10.0)
            elif outcome == "declined":
                event_text = f"💔 {self.name} confessed to {crush_name}... but it wasn't reciprocated."
                await db.upsert_relationship(
                    self.id, crush_name,
                    rel["relationship"] if rel else "friend",
                    max(-1.0, (rel["sentiment"] if rel else 0.5) - 0.25),
                    confession_state="declined"
                )
                await db.add_memory(self.id,
                    f"I told {crush_name} how I felt. They didn't feel the same. That hurts.",
                    "reflection", importance=9.0)
            else:
                event_text = f"💬 {self.name} opened up to {crush_name} about their feelings..."
                await db.upsert_relationship(
                    self.id, crush_name,
                    rel["relationship"] if rel else "friend",
                    rel["sentiment"] if rel else 0.7,
                    confession_state="uncertain"
                )

            await db.add_world_event("confession", event_text)
            logger.info(f"[{self.name}] 💌 Confession to {crush_name} → {outcome}")

        return confession, response_msg

    async def maybe_introduce(self, all_agents: list) -> str | None:
        if not self._budget_ok():
            return None
        if len(all_agents) < 3 or random.random() > INTRO_CHANCE_PER_TICK:
            return None

        my_rels = await db.get_relationships(self.id)
        friends = [r for r in my_rels
                   if r["tier"] in ("friend", "close_friend", "best_friend")
                   and not r.get("is_estranged")]
        if len(friends) < 2:
            return None

        a, b     = random.sample(friends, 2)
        mee_a    = await db.get_mee(a["other_name"])
        mee_b    = await db.get_mee(b["other_name"])
        if not mee_a or not mee_b:
            return None

        ab_rel = await db.get_relationship_with(mee_a["id"], b["other_name"])
        if ab_rel and ab_rel["tier"] in ("friend", "close_friend", "best_friend"):
            return None

        try:
            msg = await self.llm.complete(
                build_introduction_prompt(
                    self.name, a["other_name"], b["other_name"],
                    a["relationship"], b["relationship"],
                ),
                max_tokens=180, temperature=0.88,
            )
            self._charge_budget()
            msg = msg.strip().strip('"') if msg else None
        except Exception:
            return None

        if not msg:
            return None

        await db.enqueue_addressed(mee_a["id"], self.name, msg)
        await db.enqueue_addressed(mee_b["id"], self.name, msg)
        await db.add_world_event(
            "introduction",
            f"🤝 {self.name} introduced {a['other_name']} and {b['other_name']} to each other."
        )
        logger.info(f"[{self.name}] 🤝 Introduced {a['other_name']} and {b['other_name']}")
        return msg

    async def maybe_surface_need(self) -> dict | None:
        if not self._budget_ok():
            return None
        existing = await db.get_todays_need(self.id)
        if existing:
            return existing
        if random.random() > NEED_CHANCE_PER_TICK:
            return None

        relationships = await db.get_relationships(self.id)
        recent        = await db.get_memories(self.id, limit=20)
        mem_contents  = [m["content"] for m in recent]

        try:
            result      = await self.llm.complete_json(
                build_need_prompt(self.name, self.mood, relationships, mem_contents),
                max_tokens=150,
            )
            self._charge_budget()
            need_type   = result.get("need", "lonely")
            target_name = result.get("target") or None
        except Exception:
            need_type, target_name = "lonely", None

        await db.set_mee_need(self.id, need_type, target_name)
        need = await db.get_todays_need(self.id)
        logger.info(f"[{self.name}] 🫀 Need → {need_type} (target: {target_name})")
        return need

    async def should_react(self, recent_chat: list[dict],
                            pending_addressed: list[dict],
                            is_sleeping: bool = False) -> bool:
        if pending_addressed:
            return True
        if is_sleeping:
            return random.random() < 0.05   # 5% chance even while sleeping
        human_recent = any(not m.get("is_mee") for m in recent_chat[-5:])
        if human_recent:
            return random.random() < 0.80   # 80% chance to react when humans are present
        return random.random() < 0.50       # 50% chance for spontaneous speech

    async def decide_action(self, channel_id: str, all_mee_names: list[str],
                             forced: bool = False,
                             social_target: Optional[str] = None) -> Optional[str]:
        await self._ensure_chroma_synced()

        # Reload latest PAD emotional state (decayed by background loop)
        await self.reload_pad()

        recent_chat       = await db.get_recent_conversations(channel_id, limit=15)
        pending_addressed = await db.pop_addressed(self.id)
        relationships     = await db.get_relationships(self.id)
        world_events      = await db.get_recent_world_events(limit=5)
        world_context     = "; ".join(e["content"] for e in world_events) if world_events else ""
        is_sleeping       = _is_sleep_hour(SLEEP_HOUR_START, SLEEP_HOUR_END)

        if not forced and not await self.should_react(recent_chat, pending_addressed, is_sleeping):
            return None

        plan = await self.ensure_plan(all_mee_names)

        query         = build_retrieval_query(
            self.name, recent_chat, pending_addressed, agenda=plan
        )
        relevant_mems = await retrieve_memories(
            self.llm, self.id, self.name, query,
            top_k=8, relationships=relationships,
        )
        mem_contents = [m["content"] for m in relevant_mems]

        need           = await db.get_todays_need(self.id)
        maslow_tier    = _maslow_tier(self.mood, need)
        estranged_rels = await db.get_estranged_relationships(self.id)
        estranged_from = [r["other_name"] for r in estranged_rels]
        crushes        = await db.get_active_crushes(self.id)
        crush_on       = crushes[0]["crush_on"] if crushes else None

        if not forced and not pending_addressed and recent_chat:
            human_msgs = [m for m in recent_chat[-5:] if not m.get("is_mee")]
            if human_msgs:
                last_human_words = set(human_msgs[-1]["content"].lower().split()[:10])
                my_recent        = [
                    m["content"] for m in recent_chat[-5:]
                    if m.get("is_mee") and m.get("author_name") == self.name
                ]
                overlap_count = sum(
                    1 for m in my_recent
                    if len(set(m.lower().split()[:10]) & last_human_words) >= 3
                )
                if overlap_count >= 2 and random.random() < 0.50:
                    logger.debug(
                        f"[{self.name}] Topic fatigue: staying silent to break fixation loop"
                    )
                    return None

        others       = [n for n in all_mee_names if n != self.name]
        unshared_for = {}
        for other_name in others[:3]:
            unshared = await db.get_unshared_highlights(self.id, other_name, limit=2)
            if unshared:
                unshared_for[other_name] = unshared

        if forced or social_target:
            _max_tokens = 500
        elif pending_addressed:
            _max_tokens = 400
        else:
            _max_tokens = 250

        is_foreground = bool(pending_addressed) or forced or bool(social_target)
        _action_llm   = self.llm_fg if is_foreground else self.llm
        logger.debug(
            f"[{self.name}] LLM tier: {'FOREGROUND' if is_foreground else 'BACKGROUND'} "
            f"({_action_llm.api_base.split('/')[2]})"
        )

        try:
            # Build stop words to prevent LLM from generating other characters' dialogue
            _stop_words = build_stop_words(self.name, all_mee_names) if all_mee_names else None
            pad_state = (self.pleasure, self.arousal, self.dominance)

            # ── Fetch v6 private state for prompt injection ───────────────────
            v6_private_state = None
            if self._v6_agent_id:
                try:
                    v6_private_state = await db.fetch_agent(self._v6_agent_id)
                except Exception:
                    pass  # Non-fatal — prompt injection is additive

            # ── Interruptible LLM drafting ────────────────────────────────────
            # Register current task so on_message can cancel it if a new
            # relevant message arrives mid-generation.
            self._drafting_task = asyncio.current_task()
            try:
                response = await _action_llm.complete(
                    build_action_prompt(
                        self.name, self.identity, self.traits, self.goals,
                        mem_contents, plan, recent_chat, relationships,
                        all_mee_names, world_context, self.location,
                        pending_addressed=pending_addressed,
                        mood=self.mood,
                        social_target=social_target,
                        maslow_tier=maslow_tier,
                        need=need,
                        estranged_from=estranged_from if estranged_from else None,
                        crush_on=crush_on,
                        unshared_for=unshared_for if unshared_for else None,
                        is_sleeping=is_sleeping,
                        pad_state=pad_state,
                        v6_private_state=v6_private_state,
                    ),
                    max_tokens=_max_tokens,
                    temperature=0.9,
                    stop=_stop_words,
                )
            except asyncio.CancelledError:
                logger.info(f"[{self.name}] Draft cancelled by interrupt")
                raise   # Propagate so _guarded_tick can re-draft
            finally:
                self._drafting_task = None

            self._charge_budget()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Action generation failed: {e}")
            return None

        if not response:
            return None

        if "[SILENT]" in response and not forced:
            return None

        response = response.replace("[SILENT]", "").strip().strip('"')
        if response.lower().startswith(f"{self.name.lower()}:"):
            response = response[len(self.name)+1:].strip()

        if not response:
            return None

        await self._record_shared_info(response, others)
        await self._maybe_queue_addressed(response, all_mee_names)

        return response

    async def _record_shared_info(self, message: str, others: list[str]):
        mems = await db.get_memories(self.id, limit=50)
        for other_name in others:
            if other_name.lower() in message.lower():
                for m in mems[:5]:
                    if m["importance"] >= 6.0 and len(m["content"]) > 20:
                        overlap = set(m["content"].lower().split()) & set(message.lower().split())
                        if len(overlap) > 3:
                            await db.add_shared_info(self.id, other_name, m["content"], m["id"])
                            break

    async def _maybe_queue_addressed(self, message: str, all_mee_names: list[str]):
        other_mees = [n for n in all_mee_names if n != self.name]
        msg_lower  = message.lower()
        for other_name in other_mees:
            patterns = [
                rf"\b{re.escape(other_name.lower())},",
                rf"@{re.escape(other_name.lower())}",
                rf"^{re.escape(other_name.lower())} ",
                rf"hey {re.escape(other_name.lower())}",
            ]
            if any(re.search(p, msg_lower) for p in patterns):
                other_mee = await db.get_mee(other_name)
                if other_mee:
                    await db.enqueue_addressed(other_mee["id"], self.name, message)

    async def move_to(self, new_location: str) -> str:
        old_location  = self.location
        self.location = new_location
        await db.update_mee(self.id, location=new_location)
        event = f"{self.name} moved from {old_location} to {new_location}."
        await db.add_world_event("movement", event)
        await self.observe(event)
        return event

    async def maybe_wander(self, guild_id: Optional[str] = None) -> Optional[str]:
        if self._exhausted:
            return None
        if time.time() - self._last_wander_time < WANDER_COOLDOWN_MINUTES * 60:
            return None
        if random.random() > WANDER_CHANCE:
            return None
        locations = list(DEFAULT_LOCATIONS)
        if guild_id:
            custom = await db.get_server_locations(guild_id)
            if custom:
                locations = custom
        choices = [loc for loc in locations if loc != self.location]
        if not choices:
            return None
        self._last_wander_time = time.time()
        return await self.move_to(random.choice(choices))

    async def check_social_initiative(self, all_agents: list,
                                       channel_id: str,
                                       in_conversation_ids: set[int] = None) -> Optional[str]:
        if len(all_agents) < 2:
            return None
        last_spoke = await db.get_mees_last_spoke(channel_id)
        now        = datetime.now(timezone.utc)
        candidates = []
        for other in all_agents:
            if other.id == self.id:
                continue
            if in_conversation_ids and other.id in in_conversation_ids:
                continue  # Skip agents already in a conversation
            spoke_at = last_spoke.get(other.id)
            if spoke_at:
                try:
                    dt = datetime.fromisoformat(spoke_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    silent_min = (now - dt).total_seconds() / 60.0
                    if silent_min > SOCIAL_INITIATIVE_MINUTES:
                        candidates.append((other, silent_min))
                except Exception:
                    pass
            else:
                candidates.append((other, float("inf")))

        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0].name

    async def morning_recap(self) -> tuple[bool, Optional[str]]:
        """(did_recap, recap_text). recap_text is None when already done today."""
        today = date.today().isoformat()
        if await db.get_morning_recap_done(self.id, today):
            return False, None

        highlights = await db.get_yesterday_highlights(self.id)
        if not highlights:
            recap = f"{self.name} woke up feeling {self.mood}. A new day."
        else:
            try:
                recap = await self.llm.complete(
                    build_morning_recap_prompt(self.name, self.mood, highlights),
                    max_tokens=200, temperature=0.85,
                )
                self._charge_budget()
                recap = recap.strip().strip('"') if recap else \
                        f"{self.name} woke up thinking about yesterday."
            except Exception:
                recap = f"{self.name} woke up thinking about what happened yesterday."

        await db.add_memory(self.id, recap, "morning_recap", importance=6.0)
        logger.info(f"[{self.name}] 🌅 Morning recap: {recap[:80]}")
        return True, recap

    async def update_mood(self, reflections: list[str]) -> None:
        if not reflections:
            return
        try:
            result   = await self.llm.complete_json(
                build_mood_update_prompt(self.name, self.mood, reflections),
                max_tokens=80,
            )
            self._charge_budget()
            new_mood = result.get("mood", self.mood)
            if new_mood and isinstance(new_mood, str):
                self.mood = new_mood[:60]
                await db.update_mee(self.id, mood=self.mood)
                logger.info(f"[{self.name}] 😶 Mood → {self.mood}")
        except Exception as e:
            logger.warning(f"[{self.name}] Mood update failed: {e}")

    async def tick(self, channel_id: str, all_mee_names: list[str],
                   all_agents: Optional[list] = None, forced: bool = False,
                   guild_id: Optional[str] = None) -> tuple[Optional[str], Optional[str], list]:
        """Returns (action, wander_event, extra_events). Tick split into ALWAYS and GATED phases."""
        try:
            await self.reload()
            self._calls_this_tick = 0   # reset budget for this tick

            data = await db.get_mee_by_id(self.id)
            if not data or not data.get("active"):
                return None, None, []

            extra_events = []

            did_recap, recap_text = await self.morning_recap()
            if did_recap and recap_text:
                extra_events.append(("morning_recap", recap_text, self))

            reflections = await self.maybe_reflect(all_mee_names)
            for r in reflections:
                logger.info(f"[{self.name}] 💭 {r[:80]}")
            if reflections:
                await self.update_mood(reflections)

            if self._budget_ok():
                await self.maybe_surface_need()

            # ── Activity gate ───────────────────────────────────────────────
            # If the agent is mid-activity, count ticks but do NOT spam
            # world-updates. The activity world-update was already posted once
            # when the activity started (see activity-start block below).
            # Activities are cosmetic overlays, NOT tick replacements — the
            # agent can still wander, socialise, and generate actions.
            if not forced and self._activity is not None:
                self._activity["ticks_left"] -= 1
                if self._activity["ticks_left"] > 0:
                    self._silent_ticks = 0  # activity counts as 'doing something'
                else:
                    # Activity expired — clear it and proceed to gated phase
                    self._activity = None


            # ── Conversation State Machine ───────────────────────────────────
            # Check if agent is mid-conversation. If so, the agent continues
            # the exchange rather than wandering or starting new social.
            conversation_partner = None
            if not forced:
                conv = await db.get_active_conversation(self.id)
                if conv:
                    now_dt = datetime.now(timezone.utc)
                    last_msg = datetime.fromisoformat(conv["last_message_at"])
                    if last_msg.tzinfo is None:
                        last_msg = last_msg.replace(tzinfo=timezone.utc)
                    elapsed = (now_dt - last_msg).total_seconds() / 60.0
                    if elapsed > CONVERSATION_MAX_MINUTES or conv["message_count"] >= CONVERSATION_MAX_MESSAGES:
                        # Conversation expired — end it
                        partner_name = conv["agent2_name"] if conv["agent1_id"] == self.id else conv["agent1_name"]
                        await db.end_conversation(conv["id"])
                        extra_events.append((
                            "conversation_end",
                            f"💬 {self.name}'s conversation with {partner_name} has come to a natural end.",
                            self
                        ))
                        self._conversation_id = None
                        logger.info(f"[{self.name}] 💬 Conversation with {partner_name} ended (expired)")
                        # ── Conversation-triggered reflection ──────────────────
                        # Reflect immediately after a conversation ends using quality model
                        if self._budget_ok():
                            try:
                                conv_mems = await db.get_memories(self.id, limit=5)
                                conv_summary = " ".join(m["content"] for m in conv_mems[:3]) if conv_mems else ""
                                await reflect_on_conversation(
                                    self.llm, self.id, self.name,
                                    partner_name, conv_summary,
                                )
                                self._charge_budget(2)
                            except Exception as refl_e:
                                logger.warning(
                                    f"[{self.name}] Conversation reflection failed: {refl_e}"
                                )
                    else:
                        # Active conversation
                        self._conversation_id = conv["id"]
                        partner_name = conv["agent2_name"] if conv["agent1_id"] == self.id else conv["agent1_name"]
                        # ── Typing-aware coordination ──────────────────────────────
                        # If this agent spoke last, wait for the other agent to
                        # respond before speaking again (prevents talking over).
                        if conv.get("last_spoke_by") == self.id:
                            logger.info(
                                f"[{self.name}] 💬 Waiting for {partner_name} to respond "
                                f"(msg #{conv['message_count'] + 1})"
                            )
                        else:
                            conversation_partner = partner_name
                            logger.info(
                                f"[{self.name}] 💬 Continuing conversation with "
                                f"{conversation_partner} (msg #{conv['message_count'] + 1})"
                            )

            if not forced and self._budget_ok():
                crush_ponder = await self.maybe_develop_crush(all_mee_names)
                if crush_ponder:
                    extra_events.append(("crush_ponder", crush_ponder, self))

            confession_msg, confession_response = None, None
            if not forced and all_agents and self._budget_ok():
                confession_msg, confession_response = await self.maybe_confess(
                    channel_id, all_agents
                )

            if not forced and all_agents and self._budget_ok():
                intro_msg = await self.maybe_introduce(all_agents)
                if intro_msg:
                    extra_events.append(("intro", intro_msg, self))

            # ── Rate-limit / circuit-breaker gate ────────────────────────────
            # If the LLM client's circuit breaker is open (rate-limited, out of
            # tokens), don't wander, socialize, generate actions, or start
            # activities. Instead, post a single world-update and immediately
            # mark the agent as exhausted. They'll auto-wake once the circuit
            # heals (default 5 min circuit breaker + 8 min exhaustion sleep).
            if not forced:
                try:
                    _llm_circuit_open  = self.llm._circuit_is_open
                    _fg_circuit_open   = self.llm_fg._circuit_is_open
                except Exception:
                    _llm_circuit_open = _fg_circuit_open = False

                if _llm_circuit_open or _fg_circuit_open:
                    if not self._exhausted:
                        self._exhausted    = True
                        self._exhausted_at = time.time()
                        logger.info(
                            f"[{self.name}] 🔇 Circuit breaker open — "
                            f"exhausted until rate limits reset"
                        )
                        extra_events.append((
                            "exhausted",
                            f"💤 {self.name} has run out of tokens and fallen asleep. "
                            f"They'll be back once their rate limits reset!",
                            self
                        ))
                    # Return early: no wander, no social initiative, no action,
                    # no activities — agent is fully asleep until limits recover.
                    return None, None, extra_events

            # ── Gated phase: wander, social, action ──────────────────────────
            # If mid-conversation: skip wander, use partner as social target.
            # Otherwise: normal wander + social initiative (filter out conversing agents).
            wander_event = None
            if not forced and not conversation_partner:
                wander_event = await self.maybe_wander(guild_id)

            social_target = conversation_partner  # None if not in conversation
            if not forced and all_agents and not conversation_partner:
                in_conv_ids = await db.get_in_conversation_ids()
                social_target = await self.check_social_initiative(
                    all_agents, channel_id, in_conversation_ids=in_conv_ids,
                )

            action = await self.decide_action(
                channel_id, all_mee_names, forced=forced, social_target=social_target
            )

            # ── Conversation lifecycle (start / update) ─────────────────────
            if action and social_target:
                if not conversation_partner and all_agents:
                    # Agent initiated a new conversation — create it
                    target_agent = next((a for a in all_agents if a.name == social_target), None)
                    if target_agent:
                        conv_id = await db.start_conversation(
                            self.id, target_agent.id,
                            self.name, target_agent.name,
                            channel_id,
                            last_spoke_by=self.id,
                        )
                        self._conversation_id = conv_id
                        logger.info(f"[{self.name}] 💬 Started conversation with {social_target} (conv #{conv_id})")
                elif self._conversation_id:
                    # Continuing an existing conversation — update activity + last speaker
                    await db.update_conversation_activity(self._conversation_id)
                    await db.update_conversation_last_spoke(self._conversation_id, self.id)

            if action:
                await db.log_conversation(channel_id, self.name, action,
                                           is_mee=True, mee_id=self.id)
                await add_observation(self.llm, self.id, f"I said: \"{action}\"",
                                       mee_name=self.name)
                await db.update_mee(self.id, last_tick=datetime.now(timezone.utc).isoformat())

            now_ts = time.time()
            # Natural wake: agent produced action — announce it
            if action:
                self._silent_ticks = 0
                if self._exhausted:
                    self._exhausted = False
                    extra_events.append(
                        ("wake_up", f"☀️ {self.name} has woken up and is back on their feet!", self)
                    )
            else:
                self._silent_ticks += 1
                if self._silent_ticks >= 10 and not self._exhausted:
                    self._exhausted    = True
                    self._exhausted_at = now_ts
                    extra_events.append(
                        ("exhausted", f"💤 {self.name} has collapsed from exhaustion and fallen asleep.", self)
                    )

            # Silent auto-wake: time-based reset with NO world update.
            # Prevents oscillation where auto-wake posts "woken up" every 5 min
            # but agent is still rate-limited and immediately falls back asleep.
            if self._exhausted and (now_ts - self._exhausted_at) >= EXHAUSTION_WAKE_MINUTES * 60:
                self._exhausted    = False
                self._silent_ticks = 0

            # ── Activity start (idle ticks only) ────────────────────────────
            # If the agent produced no action, didn't wander, and isn't exhausted,
            # they may start a short flavourful activity for the next tick.
            # The world-update is posted ONCE when the activity starts, not on
            # every subsequent tick (the activity gate skips further posts).
            # A per-agent cooldown (ACTIVITY_POST_COOLDOWN_SECS) prevents flooding
            # the world channel — the activity still suppresses exhaustion ticks
            # internally, but the broadcast is throttled.
            if not forced and not action and not wander_event and not self._exhausted and self._activity is None:
                if random.random() < ACTIVITY_CHANCE:
                    self._activity = self._pick_activity()
                    logger.info(f"[{self.name}] 🎭 Started activity: {self._activity['emoji']} {self._activity['desc']} ({self._activity['ticks_left']} ticks)")
                    # Throttle world-update posting — don't broadcast if we posted
                    # one recently (prevents activity-spam without affecting the
                    # internal activity state machine).
                    now_ts_local = time.time()
                    if now_ts_local - self._last_activity_post_time >= ACTIVITY_POST_COOLDOWN_SECS:
                        activity_event = self._activity_event()
                        if activity_event:
                            extra_events.append(("activity", activity_event, self))
                            self._last_activity_post_time = now_ts_local


            if confession_msg:
                crush_rels  = await db.get_active_crushes(self.id)
                crush_name  = crush_rels[0]["crush_on"] if crush_rels else None
                crush_agent = next((a for a in (all_agents or []) if a.name == crush_name), None)
                extra_events.append(("confession", confession_msg, self))
                if confession_response and crush_agent:
                    extra_events.append(("confession_response", confession_response, crush_agent))

            return action, wander_event, extra_events

        except Exception as e:
            logger.error(f"[{self.name}] Tick error: {e}", exc_info=True)
            return None, None, []
