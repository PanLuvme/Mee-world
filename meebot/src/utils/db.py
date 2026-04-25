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
                image_url         TEXT    NOT NULL DEFAULT '',
                channel_id        TEXT,
                webhook_url       TEXT,
                location          TEXT    NOT NULL DEFAULT 'the main channel',
                mood              TEXT    NOT NULL DEFAULT 'neutral',
                owner_discord_id  TEXT    NOT NULL DEFAULT '0',
                active            INTEGER NOT NULL DEFAULT 1,
                created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
                last_tick         TEXT
            )
        """)

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

        # Migrations for existing databases
        for table, col, coldef in [
            ("mees",          "mood",             "TEXT NOT NULL DEFAULT 'neutral'"),
            ("mees",          "owner_discord_id",  "TEXT NOT NULL DEFAULT '0'"),
            ("mees",          "quality_model",    "TEXT DEFAULT NULL"),
            ("relationships", "tier",             "TEXT NOT NULL DEFAULT 'stranger'"),
            ("relationships", "is_estranged",     "INTEGER NOT NULL DEFAULT 0"),
            ("relationships", "crush_on",         "TEXT DEFAULT NULL"),
            ("relationships", "confession_state", "TEXT NOT NULL DEFAULT 'none'"),
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
                     quality_model=None):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            INSERT INTO mees
                (name, identity, traits, goals, model, quality_model,
                 api_key, api_base, image_url, channel_id, webhook_url,
                 location, owner_discord_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, identity, json.dumps(traits), json.dumps(goals),
            model, quality_model,
            encrypt_key(api_key), api_base, image_url,
            channel_id, webhook_url, location, str(owner_discord_id),
        ))
        await db.commit()
        return cur.lastrowid


def _parse_mee(row) -> dict:
    d = dict(row)
    d["traits"]   = json.loads(d["traits"])
    d["goals"]    = json.loads(d["goals"])
    d["api_key"]  = decrypt_key(d["api_key"])
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
    # Encrypt api_key if it's being updated
    if "api_key" in kwargs and kwargs["api_key"]:
        kwargs["api_key"] = encrypt_key(kwargs["api_key"])
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
