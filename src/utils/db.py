"""
Database layer — all persistence for Mees and their memory streams.

v5 changes (Phases 1 + 5):
- Optional API key encryption via Fernet (SECRET_KEY env var)
- owner_discord_id column on mees (multi-user support)
- quality_model column on mees (Mixture-of-Agents routing)
- touch_memories(ids) batch UPDATE instead of N sequential calls
- get_memories_since(mee_id, since_iso, limit) for incremental Chroma sync
- shared_info and mee_needs tables (carried from v4)
- All v4 relationship tier / fight / crush / confession helpers retained
"""
import aiosqlite
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "/data/meebot.db")

TIER_ORDER = ["stranger", "acquaintance", "friend", "close_friend", "best_friend", "sweetheart"]
TIER_SENTIMENT_THRESHOLDS = {
    "acquaintance": 0.0,
    "friend":       0.30,
    "close_friend": 0.58,
    "best_friend":  0.78,
}
FIGHT_SENTIMENT_THRESHOLD = -0.30
CRUSH_SENTIMENT_THRESHOLD = 0.72
CRUSH_CHANCE_PER_TICK     = 0.06
VALID_NEEDS = ("lonely", "bored", "conflict_unresolved", "curious_about", "expressive", "restless")


# ─── Encryption helpers ────────────────────────────────────────────────────────

def _secret_key() -> Optional[str]:
    return os.getenv("SECRET_KEY", "").strip() or None


def encrypt_key(api_key: str) -> str:
    """Encrypt an API key if SECRET_KEY is configured; otherwise return as-is."""
    sk = _secret_key()
    if not api_key or not sk:
        return api_key
    try:
        from cryptography.fernet import Fernet
        return Fernet(sk.encode()).encrypt(api_key.encode()).decode()
    except Exception as e:
        logger.warning(f"Key encryption failed: {e}. Storing unencrypted.")
        return api_key


def decrypt_key(stored: str) -> str:
    """Decrypt an API key. Gracefully handles un-encrypted keys (returns as-is)."""
    sk = _secret_key()
    if not stored or not sk:
        return stored
    try:
        from cryptography.fernet import Fernet
        return Fernet(sk.encode()).decrypt(stored.encode()).decode()
    except Exception:
        return stored  # not encrypted or wrong key — return as-is (backward compat)


# ─── Schema ────────────────────────────────────────────────────────────────────

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")

        await db.execute("""
            CREATE TABLE IF NOT EXISTS mees (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                name              TEXT    NOT NULL UNIQUE,
                identity          TEXT    NOT NULL,
                traits            TEXT    NOT NULL DEFAULT '[]',
                goals             TEXT    NOT NULL DEFAULT '[]',
                model             TEXT    NOT NULL DEFAULT 'llama-3.1-8b-instant',
                quality_model     TEXT    DEFAULT NULL,
                api_key           TEXT    NOT NULL,
                api_base          TEXT    NOT NULL DEFAULT 'https://api.groq.com/openai/v1',
                gemini_api_key    TEXT    DEFAULT NULL,
                gemini_model      TEXT    DEFAULT NULL,
                image_url         TEXT    NOT NULL DEFAULT '',
                channel_id        TEXT,
                webhook_url       TEXT,
                location          TEXT    NOT NULL DEFAULT 'the main channel',
                mood              TEXT    NOT NULL DEFAULT 'neutral',
                owner_discord_id  TEXT    NOT NULL DEFAULT '0',
                active            INTEGER NOT NULL DEFAULT 1,
                created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
                last_tick         TEXT,
                pleasure          REAL    NOT NULL DEFAULT 0.0,
                arousal           REAL    NOT NULL DEFAULT 0.0,
                dominance         REAL    NOT NULL DEFAULT 0.0
            )
        """)

        # ── PAD emotional state columns (v5 addition) ──────────────────────────
        for col in ("pleasure", "arousal", "dominance"):
            try:
                await db.execute(f"ALTER TABLE mees ADD COLUMN {col} REAL NOT NULL DEFAULT 0.0")
            except Exception:
                pass  # Column already exists — safe on re-launch

        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                mee_id        INTEGER NOT NULL REFERENCES mees(id) ON DELETE CASCADE,
                content       TEXT    NOT NULL,
                memory_type   TEXT    NOT NULL DEFAULT 'observation',
                importance    REAL    NOT NULL DEFAULT 0.0,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                last_accessed TEXT    NOT NULL DEFAULT (datetime('now')),
                access_count  INTEGER NOT NULL DEFAULT 0,
                keywords      TEXT    NOT NULL DEFAULT '[]'
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                mee_id     INTEGER NOT NULL REFERENCES mees(id) ON DELETE CASCADE,
                plan_date  TEXT    NOT NULL,
                agenda     TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id  TEXT    NOT NULL,
                author_name TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                is_mee      INTEGER NOT NULL DEFAULT 0,
                mee_id      INTEGER REFERENCES mees(id) ON DELETE SET NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                mee_id           INTEGER NOT NULL REFERENCES mees(id) ON DELETE CASCADE,
                other_name       TEXT    NOT NULL,
                relationship     TEXT    NOT NULL DEFAULT 'stranger',
                sentiment        REAL    NOT NULL DEFAULT 0.0,
                tier             TEXT    NOT NULL DEFAULT 'stranger',
                is_estranged     INTEGER NOT NULL DEFAULT 0,
                crush_on         TEXT    DEFAULT NULL,
                confession_state TEXT    NOT NULL DEFAULT 'none',
                updated_at       TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(mee_id, other_name)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS world_state (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT    NOT NULL DEFAULT 'update',
                content     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS addressed_queue (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                mee_id      INTEGER NOT NULL REFERENCES mees(id) ON DELETE CASCADE,
                from_name   TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                handled     INTEGER NOT NULL DEFAULT 0
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS server_config (
                guild_id    TEXT PRIMARY KEY,
                locations   TEXT NOT NULL DEFAULT '[]',
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS shared_info (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                from_mee_id  INTEGER NOT NULL REFERENCES mees(id) ON DELETE CASCADE,
                to_name      TEXT    NOT NULL,
                info_snippet TEXT    NOT NULL,
                memory_id    INTEGER REFERENCES memories(id) ON DELETE SET NULL,
                shared_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS mee_needs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                mee_id       INTEGER NOT NULL REFERENCES mees(id) ON DELETE CASCADE,
                need_type    TEXT    NOT NULL,
                target_name  TEXT    DEFAULT NULL,
                need_date    TEXT    NOT NULL DEFAULT (date('now')),
                surfaced     INTEGER NOT NULL DEFAULT 0,
                resolved     INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(mee_id, need_date)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS active_conversations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                agent1_id       INTEGER NOT NULL REFERENCES mees(id) ON DELETE CASCADE,
                agent2_id       INTEGER NOT NULL REFERENCES mees(id) ON DELETE CASCADE,
                agent1_name     TEXT    NOT NULL,
                agent2_name     TEXT    NOT NULL,
                channel_id      TEXT    NOT NULL,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                last_message_at TEXT    NOT NULL DEFAULT (datetime('now')),
                message_count   INTEGER NOT NULL DEFAULT 0,
                max_messages    INTEGER NOT NULL DEFAULT 8,
                last_spoke_by   INTEGER DEFAULT NULL
            )
        """)

        # ──────────────────────────────────────────────────────────────────────────
        # v6 Schema: Emotionally persistent, event-driven multi-agent layer
        # Adapted for SQLite (UUIDs stored as TEXT, JSONB as TEXT, REAL for decimals)
        #───────────────────────────────────────────────────────────────────────────

        # 1. Core Agent State
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id                TEXT    PRIMARY KEY,
                name              TEXT    NOT NULL,
                base_persona      TEXT    NOT NULL,
                pleasure          REAL    DEFAULT 0.00,
                arousal           REAL    DEFAULT 0.00,
                dominance         REAL    DEFAULT 0.00,
                current_activity  TEXT    DEFAULT 'Idle',
                created_at        TEXT    DEFAULT (datetime('now')),
                updated_at        TEXT    DEFAULT (datetime('now'))
            )
        """)

        # 2. Agent Relationships (Social Influence)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agent_relationships (
                id                 TEXT PRIMARY KEY,
                source_agent_id    TEXT REFERENCES agents(id) ON DELETE CASCADE,
                target_agent_id    TEXT REFERENCES agents(id) ON DELETE CASCADE,
                trust              REAL DEFAULT 0.00,
                fear               REAL DEFAULT 0.00,
                admiration         REAL DEFAULT 0.00,
                rivalry            REAL DEFAULT 0.00,
                last_interaction_at TEXT,
                UNIQUE(source_agent_id, target_agent_id)
            )
        """)

        # 3. Memory Streams (Reflection & Emotional Scarring)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memory_streams (
                id                 TEXT PRIMARY KEY,
                agent_id           TEXT REFERENCES agents(id) ON DELETE CASCADE,
                memory_type        TEXT NOT NULL,
                content            TEXT NOT NULL,
                base_salience      INTEGER CHECK (base_salience BETWEEN 1 AND 10),
                current_salience   REAL,
                pad_impact_snapshot TEXT,
                created_at         TEXT DEFAULT (datetime('now')),
                last_accessed_at   TEXT DEFAULT (datetime('now'))
            )
        """)

        # 4. Event Ledger (Async Triggers)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS interaction_events (
                id              TEXT PRIMARY KEY,
                initiator_id    TEXT,
                target_id       TEXT REFERENCES agents(id),
                event_type      TEXT NOT NULL,
                event_payload   TEXT,
                processed       INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            )
        """)

        # v6 indexes
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_relationships_source
            ON agent_relationships(source_agent_id)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_streams_agent_type
            ON memory_streams(agent_id, memory_type)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_interaction_events_unprocessed
            ON interaction_events(processed) WHERE processed = 0
        """)

# Migrations for existing databases
        for table, col, coldef in [
            ("mees",                  "mood",             "TEXT NOT NULL DEFAULT 'neutral'"),
            ("mees",                  "owner_discord_id",  "TEXT NOT NULL DEFAULT '0'"),
            ("mees",                  "quality_model",    "TEXT DEFAULT NULL"),
            ("mees",                  "gemini_api_key",   "TEXT DEFAULT NULL"),
            ("mees",                  "gemini_model",     "TEXT DEFAULT NULL"),
            ("relationships",         "tier",             "TEXT NOT NULL DEFAULT 'stranger'"),
            ("relationships",         "is_estranged",     "INTEGER NOT NULL DEFAULT 0"),
            ("relationships",         "crush_on",         "TEXT DEFAULT NULL"),
            ("relationships",         "confession_state", "TEXT NOT NULL DEFAULT 'none'"),
            ("active_conversations",  "last_spoke_by",    "INTEGER DEFAULT NULL"),
        ]:
            try:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coldef}")
            except Exception:
                pass

        await db.commit()


# ─── Mee CRUD ──────────────────────────────────────────────────────────────────

async def create_mee(name, identity, traits, goals, model, api_key, api_base,
                     image_url, channel_id, webhook_url=None,
                     location="the main channel", owner_discord_id="0",
                     quality_model=None, gemini_api_key=None, gemini_model=None):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            INSERT INTO mees
                (name, identity, traits, goals, model, quality_model,
                 api_key, api_base, gemini_api_key, gemini_model,
                 image_url, channel_id, webhook_url,
                 location, owner_discord_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, identity, json.dumps(traits), json.dumps(goals),
            model, quality_model,
            encrypt_key(api_key), api_base,
            encrypt_key(gemini_api_key) if gemini_api_key else None,
            gemini_model,
            image_url, channel_id, webhook_url, location, str(owner_discord_id),
        ))
        await db.commit()
        return cur.lastrowid


def _parse_mee(row) -> dict:
    d = dict(row)
    d["traits"]   = json.loads(d["traits"])
    d["goals"]    = json.loads(d["goals"])
    d["api_key"]  = decrypt_key(d["api_key"])
    if d.get("gemini_api_key"):
        d["gemini_api_key"] = decrypt_key(d["gemini_api_key"])
    return d


async def get_mee(name: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM mees WHERE name = ? AND active = 1", (name,))
        row = await cur.fetchone()
        return _parse_mee(row) if row else None


async def get_mee_by_id(mee_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM mees WHERE id = ?", (mee_id,))
        row = await cur.fetchone()
        return _parse_mee(row) if row else None


async def list_mees(include_inactive: bool = False) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        q = "SELECT * FROM mees" if include_inactive else "SELECT * FROM mees WHERE active = 1"
        cur = await db.execute(q + " ORDER BY name")
        return [_parse_mee(r) for r in await cur.fetchall()]


async def list_mees_by_owner(owner_discord_id: str) -> list[dict]:
    """Return all active Mees owned by a specific Discord user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM mees WHERE owner_discord_id = ? AND active = 1 ORDER BY name",
            (str(owner_discord_id),)
        )
        return [_parse_mee(r) for r in await cur.fetchall()]


async def count_mees_by_owner(owner_discord_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM mees WHERE owner_discord_id = ? AND active = 1",
            (str(owner_discord_id),)
        )
        row = await cur.fetchone()
        return row[0] if row else 0


async def update_mee(mee_id: int, **kwargs):
    if not kwargs:
        return
    for k in ("traits", "goals"):
        if k in kwargs and isinstance(kwargs[k], list):
            kwargs[k] = json.dumps(kwargs[k])
    # Encrypt api keys if being updated
    if "api_key" in kwargs and kwargs["api_key"]:
        kwargs["api_key"] = encrypt_key(kwargs["api_key"])
    if "gemini_api_key" in kwargs and kwargs["gemini_api_key"]:
        kwargs["gemini_api_key"] = encrypt_key(kwargs["gemini_api_key"])
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [mee_id]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE mees SET {fields} WHERE id = ?", values)
        await db.commit()


async def delete_mee(name: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE mees SET active = 0 WHERE name = ?", (name,))
        await db.commit()


async def delete_mee_by_id(mee_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE mees SET active = 0 WHERE id = ?", (mee_id,))
        await db.commit()


# ─── Memory Stream ─────────────────────────────────────────────────────────────

async def add_memory(mee_id: int, content: str, memory_type: str = "observation",
                     importance: float = 0.0, keywords: list = None) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            INSERT INTO memories (mee_id, content, memory_type, importance, keywords)
            VALUES (?, ?, ?, ?, ?)
        """, (mee_id, content, memory_type, importance, json.dumps(keywords or [])))
        await db.commit()
        return cur.lastrowid


async def get_memories(mee_id: int, limit: int = 100) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT * FROM memories WHERE mee_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (mee_id, limit))
        rows = await cur.fetchall()
        return [{**dict(r), "keywords": json.loads(dict(r)["keywords"])} for r in rows]


async def get_memories_since(mee_id: int, since_iso: str, limit: int = 300) -> list[dict]:
    """Fetch memories created after since_iso — used for incremental ChromaDB sync."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT * FROM memories
            WHERE mee_id = ? AND created_at > ?
            ORDER BY created_at DESC LIMIT ?
        """, (mee_id, since_iso, limit))
        rows = await cur.fetchall()
        return [{**dict(r), "keywords": json.loads(dict(r)["keywords"])} for r in rows]


async def count_memories(mee_id: int) -> int:
    """Count total memories for a Mee."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM memories WHERE mee_id = ?", (mee_id,)
        )
        row = await cur.fetchone()
        return row[0] if row else 0


async def delete_all_memories(mee_id: int) -> int:
    """Delete ALL memories for a Mee. Returns count deleted."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("DELETE FROM memories WHERE mee_id = ?", (mee_id,))
        await db.commit()
        return cur.rowcount


async def delete_today_memories(mee_id: int) -> int:
    """Delete memories created today. Returns count deleted."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM memories WHERE mee_id = ? AND date(created_at) = date('now')",
            (mee_id,),
        )
        await db.commit()
        return cur.rowcount


async def delete_memories_about_person(mee_id: int, person_name: str) -> list:
    """
    Delete memories mentioning a specific person (case-insensitive).
    Returns list of deleted memory IDs for ChromaDB cleanup.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        pattern = f"%{person_name}%"
        cur = await db.execute(
            "SELECT id FROM memories WHERE mee_id = ? AND content LIKE ? COLLATE NOCASE",
            (mee_id, pattern),
        )
        rows = await cur.fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            placeholders = ",".join("?" * len(ids))
            await db.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", ids
            )
            await db.commit()
        return ids


async def reset_all_simulation_data() -> dict[str, int]:
    """Delete ALL simulation data across ALL Mees (memories, relationships, plans,
    conversations, world events, queues, needs, shared info).

    Returns a dict of table_name → row_count_deleted.
    Does NOT delete Mee characters themselves (the ``mees`` table is preserved).
    """
    counts: dict[str, int] = {}
    async with aiosqlite.connect(DB_PATH) as db:
        tables = [
            ("memories",        "DELETE FROM memories"),
            ("relationships",   "DELETE FROM relationships"),
            ("plans",           "DELETE FROM plans"),
            ("conversations",   "DELETE FROM conversations"),
            ("world_state",     "DELETE FROM world_state"),
            ("addressed_queue", "DELETE FROM addressed_queue"),
            ("shared_info",     "DELETE FROM shared_info"),
            ("mee_needs",       "DELETE FROM mee_needs"),
        ]
        for name, sql in tables:
            cur = await db.execute(sql)
            counts[name] = cur.rowcount
        await db.commit()
    # Reset each Mee's mood & location to defaults
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE mees SET mood = 'neutral', location = 'the main channel', last_tick = NULL"
        )
        await db.commit()
    return counts


async def touch_memories(memory_ids: list[int]):
    """Batch UPDATE last_accessed + access_count for a list of memory IDs."""
    if not memory_ids:
        return
    placeholders = ",".join("?" * len(memory_ids))
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE memories SET last_accessed = datetime('now'), "
            f"access_count = access_count + 1 WHERE id IN ({placeholders})",
            memory_ids,
        )
        await db.commit()


async def touch_memory(memory_id: int):
    """Single-memory touch (kept for backward compat)."""
    await touch_memories([memory_id])


async def sum_recent_importance(mee_id: int) -> float:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            SELECT COALESCE(SUM(importance), 0) FROM memories
            WHERE mee_id = ? AND memory_type != 'reflection'
            AND created_at > (
                SELECT COALESCE(MAX(created_at), '1970-01-01')
                FROM memories WHERE mee_id = ? AND memory_type = 'reflection'
            )
        """, (mee_id, mee_id))
        row = await cur.fetchone()
        return float(row[0]) if row else 0.0


async def recent_importance_variance(mee_id: int, window: int = 20) -> float:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            SELECT importance FROM memories
            WHERE mee_id = ? AND memory_type != 'reflection'
            ORDER BY created_at DESC LIMIT ?
        """, (mee_id, window))
        rows = await cur.fetchall()
        if not rows:
            return 0.0
        vals = [r[0] for r in rows]
        mean = sum(vals) / len(vals)
        return sum((v - mean) ** 2 for v in vals) / len(vals)


# ─── Plans ─────────────────────────────────────────────────────────────────────

async def save_plan(mee_id: int, plan_date: str, agenda: list):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM plans WHERE mee_id = ? AND plan_date = ?", (mee_id, plan_date))
        await db.execute("INSERT INTO plans (mee_id, plan_date, agenda) VALUES (?, ?, ?)",
                         (mee_id, plan_date, json.dumps(agenda)))
        await db.commit()


async def get_plan(mee_id: int, plan_date: str) -> Optional[list]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT agenda FROM plans WHERE mee_id = ? AND plan_date = ?",
                               (mee_id, plan_date))
        row = await cur.fetchone()
        return json.loads(row[0]) if row else None


# ─── Conversations ─────────────────────────────────────────────────────────────

async def log_conversation(channel_id: str, author_name: str, content: str,
                            is_mee: bool = False, mee_id: int = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO conversations (channel_id, author_name, content, is_mee, mee_id)
            VALUES (?, ?, ?, ?, ?)
        """, (channel_id, author_name, content, 1 if is_mee else 0, mee_id))
        await db.commit()


async def get_recent_conversations(channel_id: str, limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT * FROM conversations WHERE channel_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (channel_id, limit))
        rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]


# ─── Relationships ─────────────────────────────────────────────────────────────

async def upsert_relationship(mee_id: int, other_name: str,
                               relationship: str, sentiment: float,
                               tier: str = None, is_estranged: bool = None,
                               crush_on: str = None, confession_state: str = None):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM relationships WHERE mee_id = ? AND other_name = ?",
            (mee_id, other_name)
        )
        existing = await cur.fetchone()
        if existing:
            e = dict(existing)
            new_tier  = tier             if tier is not None             else e["tier"]
            new_est   = (1 if is_estranged else 0) if is_estranged is not None else e["is_estranged"]
            new_crush = crush_on         if crush_on is not None         else e.get("crush_on")
            new_conf  = confession_state if confession_state is not None else e["confession_state"]
            await db.execute("""
                UPDATE relationships
                SET relationship=?, sentiment=?, tier=?, is_estranged=?,
                    crush_on=?, confession_state=?, updated_at=datetime('now')
                WHERE mee_id=? AND other_name=?
            """, (relationship, sentiment, new_tier, new_est, new_crush, new_conf, mee_id, other_name))
        else:
            await db.execute("""
                INSERT INTO relationships
                    (mee_id, other_name, relationship, sentiment, tier,
                     is_estranged, crush_on, confession_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (mee_id, other_name, relationship, sentiment,
                  tier or "stranger", 1 if is_estranged else 0,
                  crush_on, confession_state or "none"))
        await db.commit()


async def get_relationships(mee_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT * FROM relationships WHERE mee_id = ?
            ORDER BY ABS(sentiment) DESC
        """, (mee_id,))
        return [dict(r) for r in await cur.fetchall()]


async def get_relationship_with(mee_id: int, other_name: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM relationships WHERE mee_id = ? AND other_name = ?",
            (mee_id, other_name)
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def get_estranged_relationships(mee_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM relationships WHERE mee_id = ? AND is_estranged = 1", (mee_id,)
        )
        return [dict(r) for r in await cur.fetchall()]


async def get_crush_eligible(mee_id: int) -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT other_name FROM relationships
            WHERE mee_id = ?
              AND tier IN ('close_friend', 'best_friend')
              AND sentiment >= ?
              AND crush_on IS NULL
              AND is_estranged = 0
              AND confession_state = 'none'
        """, (mee_id, CRUSH_SENTIMENT_THRESHOLD))
        return [r["other_name"] for r in await cur.fetchall()]


async def get_active_crushes(mee_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM relationships WHERE mee_id = ? AND crush_on IS NOT NULL", (mee_id,)
        )
        return [dict(r) for r in await cur.fetchall()]


def compute_tier_from_sentiment(sentiment: float, current_tier: str,
                                 is_estranged: bool) -> Optional[str]:
    if is_estranged:
        return None
    try:
        cur_idx = TIER_ORDER.index(current_tier)
    except ValueError:
        cur_idx = 0

    best_eligible = current_tier
    for tier_name, threshold in TIER_SENTIMENT_THRESHOLDS.items():
        try:
            t_idx = TIER_ORDER.index(tier_name)
        except ValueError:
            continue
        if t_idx > cur_idx and t_idx <= TIER_ORDER.index("best_friend") and sentiment >= threshold:
            if TIER_ORDER.index(best_eligible) < t_idx:
                best_eligible = tier_name

    return best_eligible if best_eligible != current_tier else None


# ─── World State ───────────────────────────────────────────────────────────────

async def add_world_event(event_type: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO world_state (event_type, content) VALUES (?, ?)",
                         (event_type, content))
        await db.commit()


async def get_recent_world_events(limit: int = 10) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM world_state ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in reversed(await cur.fetchall())]


# ─── Addressed queue ───────────────────────────────────────────────────────────

async def enqueue_addressed(mee_id: int, from_name: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO addressed_queue (mee_id, from_name, content) VALUES (?, ?, ?)",
            (mee_id, from_name, content)
        )
        await db.commit()


async def pop_addressed(mee_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT * FROM addressed_queue WHERE mee_id = ? AND handled = 0
            ORDER BY created_at ASC
        """, (mee_id,))
        result = [dict(r) for r in await cur.fetchall()]
        if result:
            ids = [r["id"] for r in result]
            await db.execute(
                f"UPDATE addressed_queue SET handled = 1 WHERE id IN ({','.join('?'*len(ids))})", ids
            )
            await db.commit()
        return result


# ─── Server config ─────────────────────────────────────────────────────────────

async def get_server_locations(guild_id: str) -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT locations FROM server_config WHERE guild_id = ?", (guild_id,))
        row = await cur.fetchone()
        if row:
            locs = json.loads(row[0])
            if locs:
                return locs
    return []


async def set_server_locations(guild_id: str, locations: list[str]):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO server_config (guild_id, locations) VALUES (?, ?)
            ON CONFLICT(guild_id) DO UPDATE SET locations=excluded.locations, updated_at=datetime('now')
        """, (guild_id, json.dumps(locations)))
        await db.commit()


# ─── Shared info (information diffusion) ───────────────────────────────────────

async def add_shared_info(from_mee_id: int, to_name: str, info_snippet: str,
                           memory_id: int = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO shared_info (from_mee_id, to_name, info_snippet, memory_id)
            VALUES (?, ?, ?, ?)
        """, (from_mee_id, to_name, info_snippet[:300], memory_id))
        await db.commit()


async def get_info_shared_with(from_mee_id: int, to_name: str, limit: int = 20) -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT info_snippet FROM shared_info
            WHERE from_mee_id = ? AND to_name = ?
            ORDER BY shared_at DESC LIMIT ?
        """, (from_mee_id, to_name, limit))
        return [r["info_snippet"] for r in await cur.fetchall()]


async def get_unshared_highlights(mee_id: int, to_name: str, limit: int = 3) -> list[dict]:
    already  = set(await get_info_shared_with(mee_id, to_name, limit=100))
    all_mems = await get_memories(mee_id, limit=100)
    unshared = [
        m for m in all_mems
        if m["importance"] >= 6.0
        and m["memory_type"] in ("observation", "reflection", "morning_recap")
        and m["content"] not in already
    ]
    return unshared[:limit]


# ─── Mee Needs ─────────────────────────────────────────────────────────────────

async def set_mee_need(mee_id: int, need_type: str, target_name: str = None):
    if need_type not in VALID_NEEDS:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute("""
                INSERT INTO mee_needs (mee_id, need_type, target_name, need_date)
                VALUES (?, ?, ?, date('now'))
            """, (mee_id, need_type, target_name))
            await db.commit()
        except Exception:
            pass


async def get_todays_need(mee_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT * FROM mee_needs
            WHERE mee_id = ? AND need_date = date('now') AND resolved = 0
            ORDER BY created_at DESC LIMIT 1
        """, (mee_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def mark_need_surfaced(need_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE mee_needs SET surfaced = 1 WHERE id = ?", (need_id,))
        await db.commit()


async def mark_need_resolved(mee_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE mee_needs SET resolved = 1 WHERE mee_id = ? AND need_date = date('now')",
            (mee_id,)
        )
        await db.commit()


# ─── Living-world helpers ───────────────────────────────────────────────────────

async def get_yesterday_highlights(mee_id: int, min_importance: float = 7.0) -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT content FROM memories
            WHERE mee_id = ? AND importance >= ? AND date(created_at) = date('now', '-1 day')
            ORDER BY importance DESC LIMIT 8
        """, (mee_id, min_importance))
        return [r["content"] for r in await cur.fetchall()]


async def get_mees_last_spoke(channel_id: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT mee_id, MAX(created_at) AS last_spoke
            FROM conversations WHERE channel_id = ? AND is_mee = 1 AND mee_id IS NOT NULL
            GROUP BY mee_id
        """, (channel_id,))
        return {r["mee_id"]: r["last_spoke"] for r in await cur.fetchall()}


async def get_morning_recap_done(mee_id: int, today: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            SELECT 1 FROM memories
            WHERE mee_id = ? AND memory_type = 'morning_recap' AND date(created_at) = ?
            LIMIT 1
        """, (mee_id, today))
        return await cur.fetchone() is not None


# ─── Active Conversation State Machine ────────────────────────────────────────────

async def start_conversation(
    agent1_id: int, agent2_id: int,
    agent1_name: str, agent2_name: str,
    channel_id: str, max_messages: int = 8,
    last_spoke_by: Optional[int] = None,
) -> int:
    """Create a new active conversation between two agents. Returns conversation ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            INSERT INTO active_conversations
                (agent1_id, agent2_id, agent1_name, agent2_name, channel_id, max_messages, last_spoke_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (agent1_id, agent2_id, agent1_name, agent2_name, channel_id, max_messages, last_spoke_by))
        await db.commit()
        return cur.lastrowid


async def get_active_conversation(agent_id: int) -> Optional[dict]:
    """Get active conversation for an agent (as either participant). Returns None if none."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT * FROM active_conversations
            WHERE (agent1_id = ? OR agent2_id = ?)
            LIMIT 1
        """, (agent_id, agent_id))
        row = await cur.fetchone()
        return dict(row) if row else None


async def update_conversation_activity(conv_id: int):
    """Increment message_count and update last_message_at."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE active_conversations
            SET message_count = message_count + 1,
                last_message_at = datetime('now')
            WHERE id = ?
        """, (conv_id,))
        await db.commit()


async def update_conversation_last_spoke(conv_id: int, agent_id: int):
    """Update which agent spoke last in a conversation (typing-aware coord)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE active_conversations
            SET last_spoke_by = ?
            WHERE id = ?
        """, (agent_id, conv_id))
        await db.commit()


async def end_conversation(conv_id: int):
    """Delete an active conversation."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM active_conversations WHERE id = ?", (conv_id,))
        await db.commit()


async def get_in_conversation_ids() -> set[int]:
    """Return set of all agent IDs currently in any active conversation."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT agent1_id, agent2_id FROM active_conversations")
        rows = await cur.fetchall()
        ids: set[int] = set()
        for r in rows:
            ids.add(r[0])
            ids.add(r[1])
        return ids


# ═══════════════════════════════════════════════════════════════════════════
# PAD Emotional State Engine (Pleasure, Arousal, Dominance)
# ═══════════════════════════════════════════════════════════════════════════

async def get_pad(mee_id: int) -> dict:
    """Return PAD emotional state for an agent: {pleasure, arousal, dominance}."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT pleasure, arousal, dominance FROM mees WHERE id = ?",
            (mee_id,),
        )
        row = await cur.fetchone()
        if row:
            return dict(row)
        return {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}


async def set_pad(mee_id: int, pleasure: float, arousal: float, dominance: float):
    """Set PAD values for an agent (clamped to -1.0..1.0)."""
    p = max(-1.0, min(1.0, pleasure))
    a = max(-1.0, min(1.0, arousal))
    d = max(-1.0, min(1.0, dominance))
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE mees SET pleasure = ?, arousal = ?, dominance = ? WHERE id = ?",
            (p, a, d, mee_id),
        )
        await db.commit()


async def decay_all_pad(factor: float = 0.95):
    """Exponential decay of every agent's PAD values toward 0.0.
    
    Called periodically by the background decay loop in main.py.
    factor=0.95 → ~50% decay after ~14 loops (~28 min at 2-min intervals).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"""
            UPDATE mees SET
                pleasure  = MAX(-1.0, MIN(1.0, pleasure  * ?)),
                arousal   = MAX(-1.0, MIN(1.0, arousal   * ?)),
                dominance = MAX(-1.0, MIN(1.0, dominance * ?))
        """, (factor, factor, factor))
        await db.commit()


# ═══════════════════════════════════════════════════════════════════════════
# v6 Schema — Emotionally Persistent Multi-Agent Layer
# These tables / functions supplement (not replace) the existing mees/memories
# schema. UUIDs are generated in Python; timestamps stored as ISO-8601 TEXT.
# ═══════════════════════════════════════════════════════════════════════════

# ─── 1. Core Agent State ──────────────────────────────────────────────────

async def create_agent(name: str, base_persona: str) -> str:
    """Create a new v6 agent. Returns the agent's UUID."""
    agent_id = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO agents (id, name, base_persona) VALUES (?, ?, ?)",
            (agent_id, name, base_persona),
        )
        await db.commit()
    return agent_id


async def fetch_agent(agent_id: str) -> Optional[dict]:
    """Fetch a v6 agent by UUID. Returns None if not found."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def fetch_agent_by_name(name: str) -> Optional[dict]:
    """Fetch a v6 agent by name. Returns None if not found."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM agents WHERE name = ?", (name,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def update_agent_pad(agent_id: str, pleasure: float, arousal: float, dominance: float):
    """Set PAD values for a v6 agent (clamped to -1.0..1.0)."""
    p = max(-1.0, min(1.0, pleasure))
    a = max(-1.0, min(1.0, arousal))
    d = max(-1.0, min(1.0, dominance))
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE agents SET pleasure = ?, arousal = ?, dominance = ?, updated_at = datetime('now') WHERE id = ?",
            (p, a, d, agent_id),
        )
        await db.commit()


async def update_agent_activity(agent_id: str, activity: str):
    """Set the current_activity for a v6 agent."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE agents SET current_activity = ?, updated_at = datetime('now') WHERE id = ?",
            (activity, agent_id),
        )
        await db.commit()


async def list_all_agents() -> list[dict]:
    """Return all v6 agents."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM agents ORDER BY created_at")
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


# ─── 2. Agent Relationships (Social Influence) ─────────────────────────────

async def upsert_agent_rel(
    source_agent_id: str, target_agent_id: str,
    trust: float = 0.0, fear: float = 0.0,
    admiration: float = 0.0, rivalry: float = 0.0,
):
    """Insert or update a relationship between two v6 agents."""
    rel_id = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO agent_relationships
               (id, source_agent_id, target_agent_id, trust, fear, admiration, rivalry, last_interaction_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(source_agent_id, target_agent_id) DO UPDATE SET
                   trust             = ?,
                   fear              = ?,
                   admiration        = ?,
                   rivalry           = ?,
                   last_interaction_at = datetime('now')""",
            (rel_id, source_agent_id, target_agent_id,
             trust, fear, admiration, rivalry,
             trust, fear, admiration, rivalry),
        )
        await db.commit()


async def fetch_agent_relationships(agent_id: str) -> list[dict]:
    """Return all relationships for a v6 agent (as source or target)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT * FROM agent_relationships
               WHERE source_agent_id = ? OR target_agent_id = ?
               ORDER BY last_interaction_at DESC""",
            (agent_id, agent_id),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def fetch_agent_relationship(source_agent_id: str, target_agent_id: str) -> Optional[dict]:
    """Return the relationship between two v6 agents, if any."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT * FROM agent_relationships
               WHERE source_agent_id = ? AND target_agent_id = ?""",
            (source_agent_id, target_agent_id),
        )
        row = await cur.fetchone()
        return dict(row) if row else None


# ─── 3. Memory Streams (Reflection & Emotional Scarring) ───────────────────

async def insert_memory_stream(
    agent_id: str, memory_type: str, content: str,
    base_salience: int, pad_snapshot: Optional[dict] = None,
) -> str:
    """Add a memory to a v6 agent's memory stream. Returns the memory UUID."""
    mem_id = str(uuid.uuid4())
    snapshot_json = json.dumps(pad_snapshot) if pad_snapshot else None
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO memory_streams
               (id, agent_id, memory_type, content, base_salience, current_salience, pad_impact_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (mem_id, agent_id, memory_type, content, base_salience, float(base_salience), snapshot_json),
        )
        await db.commit()
    return mem_id


async def fetch_memory_streams(
    agent_id: str, memory_type: Optional[str] = None, limit: int = 50,
) -> list[dict]:
    """Fetch memories for a v6 agent, optionally filtered by type."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if memory_type:
            cur = await db.execute(
                """SELECT * FROM memory_streams
                   WHERE agent_id = ? AND memory_type = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (agent_id, memory_type, limit),
            )
        else:
            cur = await db.execute(
                """SELECT * FROM memory_streams
                   WHERE agent_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (agent_id, limit),
            )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def fetch_high_salience_streams(agent_id: str, min_salience: int = 7) -> list[dict]:
    """Fetch core / scar memories (base_salience >= threshold)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT * FROM memory_streams
               WHERE agent_id = ? AND base_salience >= ?
               ORDER BY base_salience DESC, created_at DESC""",
            (agent_id, min_salience),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def update_stream_salience(stream_id: str, new_salience: float):
    """Update current_salience for a memory stream entry."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE memory_streams SET current_salience = ? WHERE id = ?",
            (new_salience, stream_id),
        )
        await db.commit()


async def touch_memory_stream(stream_id: str):
    """Update last_accessed_at to now for a memory stream entry."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE memory_streams SET last_accessed_at = datetime('now') WHERE id = ?",
            (stream_id,),
        )
        await db.commit()


# ─── 4. Interaction Events (Async Trigger Ledger) ──────────────────────────

async def push_interaction_event(
    initiator_id: str, target_id: Optional[str],
    event_type: str, payload: Optional[dict] = None,
) -> str:
    """Enqueue an interaction event for async processing. Returns the event UUID."""
    event_id = str(uuid.uuid4())
    payload_json = json.dumps(payload) if payload else "{}"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO interaction_events
               (id, initiator_id, target_id, event_type, event_payload)
               VALUES (?, ?, ?, ?, ?)""",
            (event_id, initiator_id, target_id, event_type, payload_json),
        )
        await db.commit()
    return event_id


async def fetch_unprocessed_events(limit: int = 50) -> list[dict]:
    """Return unprocessed interaction events, oldest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT * FROM interaction_events
               WHERE processed = 0
               ORDER BY created_at ASC LIMIT ?""",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def mark_event_done(event_id: str):
    """Mark an interaction event as processed."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE interaction_events SET processed = 1 WHERE id = ?",
            (event_id,),
        )
        await db.commit()


# ─── Phase 2: Decay & Update Loop Functions ────────────────────────────────


async def decay_v6_agent_pad(factor: float = 0.95):
    """Exponential decay of v6 agents table PAD values toward 0.0.
    
    Each cycle: new_value = old_value * factor
    Default factor 0.95 = 5% decay per cycle (runs every 2 minutes).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE agents SET
               pleasure   = ROUND(pleasure   * ?, 4),
               arousal    = ROUND(arousal    * ?, 4),
               dominance  = ROUND(dominance  * ?, 4),
               updated_at = datetime('now')
               WHERE pleasure != 0 OR arousal != 0 OR dominance != 0""",
            (factor, factor, factor),
        )
        await db.commit()


async def decay_all_salience(factor: float = 0.98, core_factor: float = 0.995):
    """Decay current_salience for all memory_streams entries.
    
    Two-tier decay:
    - base_salience >= 9 (core memories/scars): decays much slower (core_factor)
    - base_salience < 9 (ordinary memories): decays at normal rate (factor)
    
    Salience never goes below 0.5 so it stays retrievable.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        # Ordinary memories: decay toward 0.5
        await db.execute(
            """UPDATE memory_streams SET
               current_salience = MAX(0.5, ROUND(current_salience * ?, 4))
               WHERE base_salience < 9 AND current_salience > 0.5""",
            (factor,),
        )
        # Core memories (base_salience 9-10): decay much slower
        await db.execute(
            """UPDATE memory_streams SET
               current_salience = MAX(0.5, ROUND(current_salience * ?, 4))
               WHERE base_salience >= 9 AND current_salience > 0.5""",
            (core_factor,),
        )
        await db.commit()


async def process_pending_events(limit: int = 10):
    """Event processor: handle unprocessed interaction_events.
    
    Currently a placeholder that marks events as processed.
    In future phases, this will trigger side-effects like
    relationship updates, memory formation, or world events.
    """
    events = await fetch_unprocessed_events(limit=limit)
    if not events:
        return [], 0

    processed_ids = []
    for ev in events:
        # ── Placeholder: log and mark done ──────────────────────────────────
        logger = getattr(process_pending_events, "_logger", None)
        if logger is None:
            import logging
            logger = logging.getLogger("event_processor")
            process_pending_events._logger = logger

        logger.debug(
            "Processing event %s — type=%s initiator=%s target=%s",
            ev["id"], ev["event_type"], ev["initiator_id"], ev["target_id"],
        )

        # Future: dispatch to relationship_update / memory_formation / world_event
        # based on ev["event_type"]

        await mark_event_done(ev["id"])
        processed_ids.append(ev["id"])

    return processed_ids, len(events)
