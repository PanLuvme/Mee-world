# AI Town Reference Analysis — Comparison with MeeBot

## Overview

[AI Town](ai-town-reference/ai-town-main) by a16z is a 2D pixel-art virtual town where AI agents and human players coexist. It's built on **Convex** (real-time backend-as-service) with a deterministic game engine, while MeeBot is a **Discord-native** Python bot using SQLite + ChromaDB. Despite different tech stacks, there are significant architectural ideas worth adapting.

---

## What MeeBot Does Better Already

| Area | MeeBot Advantage |
|------|-----------------|
| **Multi-provider per agent** | Each Mee has both background (Groq) and foreground (Gemini) LLMs — AI Town uses a single provider |
| **Discord-native UX** | Webhooks, slash commands, embeds, modal forms, buttons — AI Town is a web game |
| **Durable persistence** | SQLite survives restarts — AI Town relies on Convex's edge platform |
| **Token budget system** | `MAX_CALLS_PER_TICK`, `BUDGET_RATE`, circuit breaker — AI Town has no token economics |
| **Daily rhythm** | Sleep hours, wander cooldown, excitement meter, time-of-day mood — AI Town has no 24h cycle |
| **Social depth** | Relationship tiers, fights, crushes, estrangement, reconciliation — AI Town tracks only whether two players talked |
| **Exhaustion recovery** | Circuit breaker + auto-wake cycle — AI Town has no rate-limit handling |

---

## Architectural Upgrades to Consider

### 1. Typing-Aware Conversation Coordination

**AI Town:** When agent Alice starts typing (`conversation.setIsTyping`), agent Bob waits for Alice to finish before sending his message. This prevents both agents talking over each other in Mee-to-Mee dialogue.

**File:** [`convex/aiTown/agent.ts`](ai-town-reference/ai-town-main/convex/aiTown/agent.ts:163) — `if (conversation.isTyping && conversation.isTyping.playerId !== player.id) { return; }`

**MeeBot currently:** Each agent's `decide_action()` fires independently, so two Mees could attempt to respond at the same time. Mee-to-Mee interactions happen via social initiative checks but without coordination.

**Upgrade value:** Medium — would improve natural conversation flow between Mees.

---

### 2. Embeddings Cache (SHA-256 → Embedding)

**AI Town:** Caches text-to-embedding mappings in a `embeddingsCache` table keyed by SHA-256 hash. When computing an embedding, it first checks the cache. Only misses go to the API.

**File:** [`convex/agent/embeddingsCache.ts`](ai-town-reference/ai-town-main/convex/agent/embeddingsCache.ts:9-12)

```typescript
const result = await fetchBatch(ctx, [text]);
// Batch checks cache first, then fetches only missing ones
```

**MeeBot currently:** `vector_store.upsert_memory()` calls `compute_embedding()` every time without checking if the same text was already embedded. The `_ensure_chroma_synced()` method re-embeds all unsynced memories.

**Upgrade value:** **High** — would eliminate redundant embedding API calls, reducing latency and cost. A simple SQLite table `embedding_cache(text_hash TEXT UNIQUE, embedding BLOB)` would suffice.

**Files to modify:**
- [`src/agents/vector_store.py`](src/agents/vector_store.py) — add `get_cached_embedding()` / `set_cached_embedding()`
- [`src/utils/db.py`](src/utils/db.py) — add `embedding_cache` table
- [`src/agents/memory.py`](src/agents/memory.py) — use cache before API call

---

### 3. Hybrid Memory Ranking (Relevance + Recency + Importance)

**AI Town:** After vector search, AI Town over-fetches `NUM_MEMORIES_TO_SEARCH * 10` candidates, then re-ranks them by combined score:

```typescript
overallScore = normalize(relevance) + normalize(importance) + normalize(recency)
```

**File:** [`convex/agent/memory.ts`](ai-town-reference/ai-town-main/convex/agent/memory.ts:158-174) — `rankAndTouchMemories()`

Each memory gets an `importance` score (0-9, LLM-rated) and `lastAccess` timestamp. The normalization ensures no single dimension dominates.

**MeeBot currently:** [`retrieve_top_memories()`](src/agents/vector_store.py:184-234) uses `recency * relevance` with a simpler scoring approach. It doesn't incorporate importance or normalize dimensions.

**Upgrade value:** **High** — would produce more contextually relevant memory retrieval. Memories that are both recent AND important AND relevant would be surfaced more consistently.

**Files to modify:**
- [`src/agents/vector_store.py`](src/agents/vector_store.py) — implement 3-factor hybrid scoring
- [`src/utils/db.py`](src/utils/db.py) — ensure `importance` column is always populated

---

### 4. Conversation State Machine (Formal Phases)

**AI Town:** Conversations follow strict state machine:
`invited` → `walkingOver` → `participating` → `ended`
with timeouts, max messages (8), max duration (10 min), and cooldowns between pairs (60s).

**File:** [`convex/aiTown/agent.ts`](ai-town-reference/ai-town-main/convex/aiTown/agent.ts:106-234) — full state switch on `member.status.kind`

**MeeBot currently:** Mee-to-Mee interactions are opportunistic — `check_social_initiative()` finds a target and responds directly. There's no formal conversation lifecycle with states, cooldowns, or max-length enforcement.

**Upgrade value:** Medium — would make Mee-to-Mee dialogues more structured and prevent runaway conversations.

---

### 5. LLM Stop Word Truncation

**AI Town:** Passes `stop: stopWords(otherPlayer.name, player.name)` to the LLM API AND implements client-side stop word detection in `ChatCompletionContent` class, which truncates streaming output at the first stop word match.

**File:** [`convex/agent/conversation.ts`](ai-town-reference/ai-town-main/convex/agent/conversation.ts:348-352) — `stopWords()` function

**MeeBot currently:** Doesn't use `stop` in any [`llm.py`](src/agents/llm.py) API calls. The `complete()` method relies solely on `max_tokens` to control output length.

**Upgrade value:** Low-Medium — stop words are a nice-to-have for preventing self-referential output, but MeeBot's current approach works.

---

### 6. Activity System (Doing Things While Idle)

**AI Town:** Agents can be assigned an `activity` with a duration (reading 📖, daydreaming 🤔, gardening 🥕). During this time, the agent is "busy" and won't wander or invite. This makes the world feel alive between conversations.

**File:** [`convex/constants.ts`](ai-town-reference/ai-town-main/convex/constants.ts:67-71) — `ACTIVITIES` array

```typescript
ACTIVITIES = [
  { description: 'reading a book', emoji: '📖', duration: 60_000 },
  { description: 'daydreaming', emoji: '🤔', duration: 60_000 },
  { description: 'gardening', emoji: '🥕', duration: 60_000 },
];
```

**MeeBot currently:** Agents wander between locations or stay idle. There's no "doing an activity" state between ticks.

**Upgrade value:** Medium — would make idle ticks more narratively interesting. Each wandering tick could generate a small world update like "Iris is reading a book at the library."

**Files to modify:**
- [`src/agents/agent.py`](src/agents/agent.py) — add `_activity` state to `MeeAgent.__init__`
- [`src/agents/llm.py`](src/agents/llm.py) — potentially add activity-related prompts

---

### 7. Rate-Limit Queue with Backoff (vs. Circuit Breaker)

**AI Town:** Uses `retryWithBackoff()` with exponential backoff [1s, 10s, 20s] for 429/5xx errors.

**File:** [`convex/util/llm.ts`](ai-town-reference/ai-town-main/convex/util/llm.ts:289-318)

**MeeBot currently:** Uses a binary circuit breaker (open after 3 consecutive failures, closed after 5 minute cooldown). This is more aggressive but also more wasteful — a single 429 shouldn't cut off all service.

**Upgrade value:** Low — circuit breaker is already solving the problem, but adding retry-with-backoff *before* tripping the breaker could reduce false positives.

**Files to modify:**
- [`src/agents/llm.py`](src/agents/llm.py) — add retry-with-backoff to `_complete_once()` before incrementing `_fail_count`

---

### 8. Conversation-Triggered Reflection

**AI Town:** `reflectOnMemories()` is called at the end of every `rememberConversation()` — reflection is conversation-driven.

**File:** [`convex/agent/memory.ts`](ai-town-reference/ai-town-main/convex/agent/memory.ts:84) — `await reflectOnMemories(ctx, worldId, playerId);`

**MeeBot currently:** `maybe_reflect()` runs on a fixed schedule (checking sum of recent importance across all memories) rather than being triggered by specific events.

**Upgrade value:** Low-Medium — conversation-triggered reflection would produce more timely insights, but MeeBot's time-based approach is simpler and already functional.

---

### 9. Cooldown Enforcement Between Specific Pairs

**AI Town:** The `participatedTogether` table tracks when each pair last conversed, with `PLAYER_CONVERSATION_COOLDOWN` (60s) preventing immediate re-engagement.

**File:** [`convex/aiTown/agent.ts`](ai-town-reference/ai-town-main/convex/aiTown/agent.ts:349-361) — `findConversationCandidate` query

**MeeBot currently:** Social initiative targeting doesn't check for per-pair cooldowns. An agent could repeatedly target the same Mee if they're the only available partner.

**Upgrade value:** Low — already mitigated by wander cooldowns and the variety of possible targets.

---

## Prioritized Upgrade Recommendations

### Tier 1 (High Value, Low Risk)

| # | Upgrade | Why |
|---|---------|-----|
| 1 | **Embeddings Cache** | Reduces API costs immediately. Simple SQLite table + 10 lines of Python. Every memory write hits the embedding API once — caching eliminates duplicates. |
| 2 | **Hybrid Memory Ranking** | Better context retrieval = more coherent agent behavior. Combines the three dimensions (relevance, recency, importance) that MeeBot already stores but doesn't normalize together. |

### Tier 2 (Medium Value, Moderate Effort)

| # | Upgrade | Why |
|---|---------|-----|
| 3 | **Activity System** | Makes idle ticks narratively richer. Agents read, daydream, garden between conversations rather than just standing still. |
| 4 | **Conversation State Machine** | Formalizes Mee-to-Mee dialogue with phases, timeouts, and max-message limits. Prevents runaway conversations. |
| 5 | **Typing-Aware Coordination** | Prevents two Mees from talking over each other. Better realism in direct dialogue. |

### Tier 3 (Lower Value or Already Addressed)

| # | Upgrade | Why |
|---|---------|-----|
| 6 | **Retry-with-Backoff before Circuit Breaker** | Reduces false-positive breaker trips. Low effort but the current system works. |
| 7 | **Conversation-Triggered Reflection** | Better timing but MeeBot's scheduled approach is fine. |
| 8 | **Stop Word Truncation** | Client-side safety net, but MeeBot's output is short enough already. |
| 9 | **Per-Pair Cooldowns** | Already mitigated by other mechanisms. |
