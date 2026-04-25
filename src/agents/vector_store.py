"""
Vector store — ChromaDB-backed memory retrieval.

Each Mee gets its own ChromaDB collection.
Uses local ONNX embeddings (no API call needed for embeddings).

v5 changes (Phase 3):
- Weighted scoring formula matching Park et al. (2023) Generative Agents:
    score = 0.40 * relevance + 0.30 * importance + 0.20 * recency + 0.10 * rel_boost
  (previously all added with equal weight 1.0)
- Memory type bonus: reflections and morning_recaps float up, raw conversations float down
- RETRIEVAL_WEIGHTS and TYPE_BONUS are module-level constants — easy to tune
"""
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_chroma_client = None
_collections: dict[int, object] = {}  # mee_id → chroma collection

# ─── Tunable retrieval weights ─────────────────────────────────────────────────
# Must sum to 1.0 for the formula to be interpretable
RETRIEVAL_WEIGHTS = {
    "relevance":  0.40,
    "importance": 0.32,  # +0.02 — meaningful memories should win over fresh chat
    "recency":    0.16,  # -0.04 — reduce recency pressure to prevent topic fixation
    "rel_boost":  0.12,  # +0.02 — relationship-linked memories get a nudge
}

# Added on top of the weighted sum — keeps values bounded since these are small
TYPE_BONUS: dict[str, float] = {
    "reflection":     0.12,   # deliberately formed insight → surface more
    "morning_recap":  0.08,   # start-of-day context → valuable
    "observation":    0.00,   # neutral
    "plan":           0.02,
    "conversation":  -0.05,   # raw chat line → lower signal, de-prioritise
}


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        data_dir = os.path.dirname(os.getenv("DB_PATH", "/data/meebot.db"))
        _chroma_client = chromadb.PersistentClient(path=os.path.join(data_dir, "chroma"))
        logger.info("✅ ChromaDB initialised")
    return _chroma_client


def _get_collection(mee_id: int, mee_name: str):
    if mee_id not in _collections:
        client    = _get_client()
        safe_name = "".join(c if c.isalnum() else "_" for c in mee_name)
        col       = client.get_or_create_collection(
            name=f"mee_{mee_id}_{safe_name}",
            metadata={"hnsw:space": "cosine"},
        )
        _collections[mee_id] = col
    return _collections[mee_id]


def upsert_memory(mee_id: int, mee_name: str, memory_id: int,
                   content: str, importance: float, memory_type: str,
                   created_at: str):
    """Add or update a memory in the vector store."""
    try:
        col = _get_collection(mee_id, mee_name)
        col.upsert(
            ids=[str(memory_id)],
            documents=[content],
            metadatas=[{
                "importance":   importance,
                "memory_type":  memory_type,
                "created_at":   created_at,
            }],
        )
    except Exception as e:
        logger.warning(f"ChromaDB upsert failed for mee {mee_id}: {e}")


def delete_memories_from_chroma(mee_id: int, mee_name: str, memory_ids: list):
    """Remove specific memories from ChromaDB by their SQLite IDs."""
    if not memory_ids:
        return
    try:
        col = _get_collection(mee_id, mee_name)
        col.delete(ids=[str(mid) for mid in memory_ids])
        logger.info(f"Deleted {len(memory_ids)} memories from ChromaDB for {mee_name}")
    except Exception as e:
        logger.warning(f"ChromaDB delete failed for mee {mee_id}: {e}")


def delete_collection(mee_id: int, mee_name: str):
    """Delete the entire ChromaDB collection for a Mee (used on full memory wipe)."""
    try:
        client    = _get_client()
        safe_name = "".join(c if c.isalnum() else "_" for c in mee_name)
        client.delete_collection(name=f"mee_{mee_id}_{safe_name}")
        if mee_id in _collections:
            del _collections[mee_id]
        logger.info(f"Deleted ChromaDB collection for {mee_name}")
    except Exception as e:
        logger.warning(f"ChromaDB collection delete failed for {mee_name}: {e}")


def query_memories(mee_id: int, mee_name: str, query: str,
                   n_results: int = 30) -> list[dict]:
    """
    Query the vector store for semantically similar memories.
    Returns list of {id, content, importance, memory_type, created_at, distance}.
    """
    try:
        col   = _get_collection(mee_id, mee_name)
        count = col.count()
        if count == 0:
            return []
        actual_n = min(n_results, count)
        results  = col.query(
            query_texts=[query],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )
        out  = []
        ids  = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for i, mem_id in enumerate(ids):
            out.append({
                "id":          int(mem_id),
                "content":     docs[i],
                "importance":  metas[i].get("importance", 5.0),
                "memory_type": metas[i].get("memory_type", "observation"),
                "created_at":  metas[i].get("created_at", ""),
                "distance":    dists[i],
            })
        return out
    except Exception as e:
        logger.warning(f"ChromaDB query failed for mee {mee_id}: {e}")
        return []


def recency_score(created_at_str: str, decay: float = 0.990) -> float:
    """Exponential decay based on hours since memory was created. Returns 0–1.
    Lower decay (0.990 vs 0.995) means older memories fade faster, giving
    less recency penalty pressure — important memories surface over fresh ones."""
    try:
        dt = datetime.fromisoformat(created_at_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
        return math.pow(decay, hours)
    except Exception:
        return 0.5


def normalize_importance(raw: float) -> float:
    """Map raw importance (1–10) to [0, 1]."""
    return max(0.0, min(1.0, (raw - 1.0) / 9.0))


def retrieve_top_memories(
    mee_id: int,
    mee_name: str,
    query: str,
    top_k: int = 10,
    relationships: list[dict] = None,
) -> list[dict]:
    """
    Full retrieval pipeline with weighted scoring (Park et al. 2023):
        score = W_rel * relevance + W_imp * importance + W_rec * recency
                + W_boost * rel_boost + type_bonus
    All base components are in [0, 1] before weighting.
    """
    candidates = query_memories(mee_id, mee_name, query, n_results=min(50, top_k * 5))
    if not candidates:
        return []

    # Build sentiment lookup for relationship boosting
    rel_map: dict[str, float] = {}
    if relationships:
        for r in relationships:
            rel_map[r["other_name"].lower()] = abs(r["sentiment"])

    W = RETRIEVAL_WEIGHTS
    scored = []
    for mem in candidates:
        recency    = recency_score(mem["created_at"])
        importance = normalize_importance(mem["importance"])
        relevance  = 1.0 - (mem["distance"] / 2.0)   # cosine dist → similarity

        # Relationship boost: any known person mentioned in this memory
        rel_boost = 0.0
        content_lower = mem["content"].lower()
        for name, strength in rel_map.items():
            if name in content_lower:
                rel_boost = max(rel_boost, strength)   # 0–1

        # Memory type bonus
        type_bonus = TYPE_BONUS.get(mem["memory_type"], 0.0)

        total = (
            W["relevance"]  * relevance  +
            W["importance"] * importance +
            W["recency"]    * recency    +
            W["rel_boost"]  * rel_boost  +
            type_bonus
        )
        scored.append((total, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]
