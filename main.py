"""
MeeBot v5 — Tomodachi Life meets Smallsville meets AgentSociety in Discord.

v5 additions:
- asyncio.Semaphore(3): max 3 concurrent agent ticks — prevents thundering herd
- Bot-lifetime aiohttp.ClientSession reused across all webhook posts
- close() override: tears down HTTP clients cleanly
- Day rhythm factor: tick interval scales by time of day (Tomodachi/Smallsville pacing)
  Peak evening: ~2.1 min effective interval. Sleep hours: ~24 min. Afternoon lull: ~3.75 min.
- morning_recap events from tick() are now posted to the channel (waking-up message)
- Tick now returns (action, wander_event, extra_events) — handles all extra event types
- _update_rel_and_events now gated: only fires on emotionally meaningful messages
- Multi-provider event icon map
"""
import asyncio
import logging
import os
import random
from datetime import datetime, timezone

import aiohttp
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from src.utils import db
from src.utils.embeds import world_event_embed, EVENT_ICONS
from src.utils.webhook import post_as_mee, get_or_create_webhook
from src.agents.agent import MeeAgent, _day_rhythm_factor, SLEEP_HOUR_START, SLEEP_HOUR_END
from src.agents.memory import add_observation
from src.agents import llm as llm_module

load_dotenv()

TOKEN              = os.getenv("DISCORD_TOKEN", "")
OWNER_ID           = int(os.getenv("OWNER_ID", "258778043944796161"))
TICK_INTERVAL_MIN  = int(os.getenv("TICK_INTERVAL_MIN", "3"))
SPOKE_COOLDOWN_MIN = int(os.getenv("SPOKE_COOLDOWN_MIN", "4"))
LOG_LEVEL          = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("meebot")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Keywords used to gate relationship updates (same set as agent.py)
_SENTIMENT_KEYWORDS = frozenset({
    "love", "hate", "angry", "miss", "sorry", "afraid", "happy", "sad",
    "hurt", "wonderful", "terrible", "heart", "cry", "crying",
    "laugh", "laughing", "scared", "jealous", "proud", "trust", "upset",
    "worried", "excited", "annoyed", "grateful", "thankful",
})


class MeeBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents, help_command=None)
        self.agents: dict[int, MeeAgent] = {}
        self._lock   = asyncio.Lock()
        # Per-channel excitement meter (0.0–1.0)
        self.excitement: dict[str, float] = {}
        # Concurrency limiter: max 3 agent ticks running simultaneously
        self._tick_semaphore = asyncio.Semaphore(3)
        # Bot-lifetime HTTP session (reused across all webhook posts)
        self._aio_session: aiohttp.ClientSession | None = None
        # Interrupt queue: agent_id → pending messages that cancelled a draft
        self._interrupt_queue: dict[int, list[dict]] = {}

    async def setup_hook(self):
        os.makedirs(os.path.dirname(db.DB_PATH), exist_ok=True)
        await db.init_db()
        logger.info("✅ Database initialised")

        # Create the bot-lifetime aiohttp session
        self._aio_session = aiohttp.ClientSession()
        logger.info("✅ Shared aiohttp session created")

        await self.load_extension("src.commands.manage")
        logger.info("✅ Commands loaded")

        await self.tree.sync()
        logger.info("✅ Slash commands synced")

        await self.reload_agents()
        self.agent_tick_loop.start()
        self.pad_decay_loop.start()
        self.salience_decay_loop.start()

    async def close(self):
        """Clean shutdown — close HTTP clients before disconnecting."""
        self.agent_tick_loop.cancel()
        self.pad_decay_loop.cancel()
        self.salience_decay_loop.cancel()

        if self._aio_session and not self._aio_session.closed:
            await self._aio_session.close()
            logger.info("✅ Shared aiohttp session closed")

        await llm_module.close_http_client()

        await super().close()

    async def reload_agents(self):
        async with self._lock:
            mees     = await db.list_mees()
            existing = set(self.agents.keys())
            db_ids   = {m["id"] for m in mees}

            for gone_id in existing - db_ids:
                del self.agents[gone_id]

            for mee in mees:
                if mee["id"] not in self.agents:
                    self.agents[mee["id"]] = MeeAgent(mee)
                    logger.info(f"🌸 Loaded Mee: {mee['name']}")

    async def on_ready(self):
        logger.info(f"🤖 MeeBot online as {self.user} ({self.user.id})")
        await self.change_presence(
            activity=discord.Activity(type=discord.ActivityType.watching, name="the Mees ✨")
        )

    # ─── Post helpers ──────────────────────────────────────────────────────────

    async def post_mee_message(self, agent: MeeAgent, content: str,
                                channel: discord.TextChannel):
        """Post a Mee message via webhook, reusing the bot's shared aiohttp session."""
        used = await post_as_mee(
            channel=channel,
            name=agent.name,
            content=content,
            image_url=agent.image_url,
            webhook_url=agent.webhook_url,
            session=self._aio_session,
        )
        if not used:
            logger.debug(f"[{agent.name}] posted via fallback (no webhook)")

    async def post_world_update(self, content: str, channel: discord.TextChannel,
                                 event_type: str = "update"):
        """Post a world state update as a subtle embed from the main bot."""
        embed = world_event_embed(content, event_type)
        try:
            await channel.send(embed=embed)
        except Exception as e:
            logger.error(f"World update post failed: {e}")

    # ─── PAD decay loop (v5 + v6) ───────────────────────────────────────────

    @tasks.loop(minutes=2)
    async def pad_decay_loop(self):
        """Slowly decay all agents' PAD emotional state toward 0.0 over time.
        
        Decays both the v5 `mees` table and v6 `agents` table.
        Also processes pending interaction events.
        """
        try:
            await db.decay_all_pad(factor=0.95)
        except Exception as e:
            logger.warning(f"v5 PAD decay error: {e}")

        try:
            await db.decay_v6_agent_pad(factor=0.95)
        except Exception as e:
            logger.warning(f"v6 PAD decay error: {e}")

        try:
            processed, total = await db.process_pending_events(limit=10)
            if processed:
                logger.debug(f"Event processor: marked {len(processed)}/{total} events done")
        except Exception as e:
            logger.warning(f"Event processor error: {e}")

    @pad_decay_loop.before_loop
    async def _before_pad_decay(self):
        await self.wait_until_ready()

    # ─── Salience decay loop ─────────────────────────────────────────────────

    @tasks.loop(minutes=5)
    async def salience_decay_loop(self):
        """Decay memory salience in the v6 memory_streams table.
        
        Runs every 5 minutes (slower than PAD because salience changes
        are less granular). Core memories (base_salience >= 9) decay
        at a much slower rate.
        """
        try:
            await db.decay_all_salience(factor=0.98, core_factor=0.995)
        except Exception as e:
            logger.warning(f"Salience decay error: {e}")

    @salience_decay_loop.before_loop
    async def _before_salience_decay(self):
        await self.wait_until_ready()

    # ─── Agent tick loop ──────────────────────────────────────────────────────

    @tasks.loop(minutes=1)
    async def agent_tick_loop(self):
        if not self.agents:
            return
        all_agents    = list(self.agents.values())
        all_mee_names = [a.name for a in all_agents]

        # Decay excitement 10% per loop tick
        for ch in list(self.excitement):
            self.excitement[ch] = max(0.0, self.excitement[ch] * 0.9)

        hour = datetime.now().hour

        for agent in all_agents:
            data = await db.get_mee_by_id(agent.id)
            if not data or not data.get("active"):
                continue
            channel_id = data.get("channel_id")
            if not channel_id:
                continue

            # Effective interval = base × excitement factor × day rhythm factor
            excitement     = self.excitement.get(channel_id, 0.0)
            excite_factor  = 1.0 - (excitement * 0.5)
            rhythm_factor  = _day_rhythm_factor(hour)
            effective_int  = max(
                SPOKE_COOLDOWN_MIN,
                (TICK_INTERVAL_MIN + (agent.id % 3)) * excite_factor * rhythm_factor,
            )

            last_tick = data.get("last_tick")
            if last_tick:
                try:
                    last_dt = datetime.fromisoformat(last_tick)
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60.0
                    if elapsed < effective_int:
                        continue
                except Exception:
                    pass

            asyncio.create_task(
                self._guarded_tick(agent, channel_id, all_mee_names, all_agents)
            )

    async def _guarded_tick(self, agent: MeeAgent, channel_id: str,
                             all_mee_names: list[str], all_agents: list):
        """Acquire semaphore before running the tick — limits concurrent ticks to 3."""
        async with self._tick_semaphore:
            try:
                await self._run_agent_tick(agent, channel_id, all_mee_names, all_agents)
            except asyncio.CancelledError:
                # Agent's draft was cancelled by a new incoming message.
                # Re-draft with the interrupt context within the same semaphore.
                await self._handle_interrupt_redraft(agent, channel_id, all_mee_names, all_agents)

    async def _handle_interrupt_redraft(self, agent: MeeAgent, channel_id: str,
                                         all_mee_names: list[str], all_agents: list):
        """Re-draft a response after an interrupt cancelled the in-flight LLM call."""
        interrupt_msgs = self._interrupt_queue.pop(agent.id, [])
        if not interrupt_msgs:
            logger.debug(f"[{agent.name}] Interrupt cancelled but no queued messages — skipping redraft")
            return

        logger.info(
            f"[{agent.name}] 🔄 Re-drafting after interrupt "
            f"({len(interrupt_msgs)} pending message{'s' if len(interrupt_msgs)>1 else ''})"
        )
        try:
            channel = self.get_channel(int(channel_id))
            if not channel:
                return

            # Run a forced decide_action so the agent responds with full context
            action = await agent.decide_action(channel_id, all_mee_names, forced=True)
            if action:
                await self.post_mee_message(agent, action, channel)
                await db.log_conversation(channel_id, agent.name, action,
                                           is_mee=True, mee_id=agent.id)
                await add_observation(agent.llm, agent.id, f"I said: \"{action}\"",
                                       mee_name=agent.name)
                await db.update_mee(agent.id, last_tick=datetime.now(timezone.utc).isoformat())
        except asyncio.CancelledError:
            logger.debug(f"[{agent.name}] Redraft was also cancelled — draining queue")
            self._interrupt_queue.pop(agent.id, None)
        except Exception as e:
            logger.error(f"[{agent.name}] Interrupt redraft error: {e}")

    async def _run_agent_tick(self, agent: MeeAgent, channel_id: str,
                               all_mee_names: list[str], all_agents: list = None):
        try:
            channel = self.get_channel(int(channel_id))
            if not channel:
                return

            guild_id = str(channel.guild.id) if hasattr(channel, "guild") and channel.guild else None

            await asyncio.sleep(random.uniform(0, 20))

            action, wander_event, extra_events = await agent.tick(
                channel_id, all_mee_names,
                all_agents=all_agents or [],
                guild_id=guild_id,
            )

            # ── Wander world update ────────────────────────────────────────────
            if wander_event:
                await self.post_world_update(wander_event, channel, event_type="movement")
                self.excitement[channel_id] = min(1.0, self.excitement.get(channel_id, 0.0) + 0.2)

            # ── Main action ────────────────────────────────────────────────────
            if action:
                await self.post_mee_message(agent, action, channel)
                logger.info(f"[{agent.name}] 💬 {action[:80]}")
                if any(
                    other.name.lower() in action.lower()
                    for other in (all_agents or [])
                    if other.id != agent.id
                ):
                    self.excitement[channel_id] = min(
                        1.0, self.excitement.get(channel_id, 0.0) + 0.3
                    )

            # ── Extra events: morning recap, tier transitions, crushes, confessions, intros ──
            for event_type, content, poster_agent in (extra_events or []):
                if event_type == "morning_recap":
                    # Post the waking-up message as the Mee (not an embed)
                    try:
                        await asyncio.sleep(random.uniform(1, 5))
                        await self.post_mee_message(poster_agent, content, channel)
                        logger.info(f"[{poster_agent.name}] 🌅 Woke up: {content[:60]}")
                    except Exception as e:
                        logger.error(f"Morning recap post failed: {e}")

                elif event_type in ("exhausted", "wake_up"):
                    # Narrative sleep/wake announced as world update (not Mee message)
                    await self.post_world_update(content, channel, event_type="exhaustion")

                elif event_type == "activity":
                    # Flavourful idle activity posted as world update (not Mee message)
                    await self.post_world_update(content, channel, event_type="activity")

                elif event_type == "conversation_end":
                    # Conversation expiry posted as world update
                    await self.post_world_update(content, channel, event_type="conversation")

                elif event_type in ("crush_ponder", "intro", "confession", "confession_response"):
                    try:
                        await asyncio.sleep(random.uniform(2, 8))
                        await self.post_mee_message(poster_agent, content, channel)
                        await db.log_conversation(
                            channel_id, poster_agent.name, content,
                            is_mee=True, mee_id=poster_agent.id
                        )
                        await add_observation(
                            poster_agent.llm, poster_agent.id,
                            f"I said: \"{content}\"", mee_name=poster_agent.name
                        )
                        # High-drama events bump excitement significantly
                        self.excitement[channel_id] = min(
                            1.0, self.excitement.get(channel_id, 0.0) + 0.5
                        )
                    except Exception as e:
                        logger.error(f"Extra event post failed ({event_type}): {e}")

        except Exception as e:
            logger.error(f"Tick error for {agent.name}: {e}", exc_info=True)

    @agent_tick_loop.before_loop
    async def before_tick(self):
        await self.wait_until_ready()
        logger.info(f"⏰ Tick loop started (base ~{TICK_INTERVAL_MIN} min, day-rhythm scaled)")

    # ─── Custom events ────────────────────────────────────────────────────────

    async def on_mee_created(self, mee_id: int):
        await self.reload_agents()
        agent = self.agents.get(mee_id)
        if agent:
            await add_observation(
                agent.llm, mee_id,
                f"{agent.name} came into existence in this Discord server, ready to meet people.",
                mee_name=agent.name,
            )
            logger.info(f"🌸 New Mee: {agent.name}")

    async def on_mee_removed(self, mee_id: int):
        async with self._lock:
            if mee_id in self.agents:
                name = self.agents[mee_id].name
                del self.agents[mee_id]
                logger.info(f"🗑️ Mee removed: {name}")

    async def on_mee_force_speak(self, mee_id: int, channel_id: int):
        agent = self.agents.get(mee_id)
        if not agent:
            return
        all_agents    = list(self.agents.values())
        all_mee_names = [a.name for a in all_agents]
        channel       = self.get_channel(channel_id)
        if not channel:
            return
        guild_id = str(channel.guild.id) if hasattr(channel, "guild") and channel.guild else None

        action, wander_event, extra_events = await agent.tick(
            str(channel_id), all_mee_names,
            all_agents=all_agents, forced=True, guild_id=guild_id,
        )
        if wander_event:
            await self.post_world_update(wander_event, channel, event_type="movement")
        if action:
            await self.post_mee_message(agent, action, channel)
        for event_type, content, poster_agent in (extra_events or []):
            if event_type in ("confession", "confession_response", "crush_ponder", "intro"):
                try:
                    await self.post_mee_message(poster_agent, content, channel)
                except Exception:
                    pass

    # ─── Message observation ──────────────────────────────────────────────────

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        channel_id = str(message.channel.id)
        author     = message.author.display_name
        content    = message.content

        await db.log_conversation(channel_id, author, content, is_mee=False)
        self.excitement[channel_id] = min(1.0, self.excitement.get(channel_id, 0.0) + 0.1)

        all_mee_names = [a.name for a in self.agents.values()]

        for agent in list(self.agents.values()):
            data = await db.get_mee_by_id(agent.id)
            if not data or not data.get("active"):
                continue
            if data.get("channel_id") != channel_id:
                continue

            # Sample-gate: don't obsessively store trivial chat as memories.
            # Always observe if the Mee is addressed; otherwise 40% for short msgs.
            is_addressed_here = agent.name.lower() in content.lower()
            if is_addressed_here or len(content) > 20 or random.random() < 0.40:
                asyncio.create_task(agent.observe_conversation(author, content))

            # Relationship update gate: only fire LLM when message is emotionally meaningful
            if len(content) > 10 and agent._should_update_relationship(content, author):
                asyncio.create_task(
                    self._update_rel_and_events(agent, author, content, channel_id)
                )

            is_addressed = agent.name.lower() in content.lower()

            # ── Interrupt-driven drafting ────────────────────────────────────
            # If the agent is currently mid-LLM-call and the new message
            # addresses them by name, cancel the in-flight draft and queue a
            # re-draft that includes the new message context.
            if is_addressed and agent.is_drafting:
                logger.info(
                    f"[{agent.name}] ⚡ Interrupting draft — new message from {author}"
                )
                self._interrupt_queue.setdefault(agent.id, []).append({
                    "author": author,
                    "content": content,
                })
                # Cancel the in-flight LLM call. The CancelledError propagates
                # to _guarded_tick → _handle_interrupt_redraft, which re-drafts.
                agent._drafting_task.cancel()
                # Skip the reactive respond — the re-draft handles it.
                continue

            # If not drafting, proceed with normal reactive response logic
            react_chance = 0.85 if is_addressed else 0.35
            if random.random() < react_chance:
                asyncio.create_task(
                    self._maybe_reactive_respond(agent, channel_id, all_mee_names)
                )

        await self.process_commands(message)

    async def _update_rel_and_events(self, agent: MeeAgent, author: str,
                                      content: str, channel_id: str):
        """Update relationship and post any world events that result."""
        channel     = self.get_channel(int(channel_id))
        world_event = await agent.update_relationship(
            author, f"{author} said: \"{content[:100]}\""
        )
        if world_event and channel:
            etype = "fight"          if "⚡" in world_event else \
                    "reconciliation" if "💚" in world_event else \
                    "relationship"
            await self.post_world_update(world_event, channel, event_type=etype)
            self.excitement[channel_id] = min(
                1.0, self.excitement.get(channel_id, 0.0) + 0.4
            )

    async def _maybe_reactive_respond(self, agent: MeeAgent, channel_id: str,
                                       all_mee_names: list[str]):
        try:
            await asyncio.sleep(random.uniform(3, 12))
            channel = self.get_channel(int(channel_id))
            if not channel:
                return
            action = await agent.decide_action(channel_id, all_mee_names)
            if action:
                await self.post_mee_message(agent, action, channel)
                await db.log_conversation(channel_id, agent.name, action,
                                           is_mee=True, mee_id=agent.id)
                await add_observation(agent.llm, agent.id, f"I said: \"{action}\"",
                                       mee_name=agent.name)
                await db.update_mee(agent.id, last_tick=datetime.now(timezone.utc).isoformat())
        except Exception as e:
            logger.error(f"Reactive response error for {agent.name}: {e}")


bot = MeeBot()


def main():
    if not TOKEN:
        logger.error("❌ DISCORD_TOKEN is not set. Check your .env file.")
        raise SystemExit(1)
    bot.run(TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
