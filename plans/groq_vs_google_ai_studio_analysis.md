# Groq vs Google AI Studio — API Provider Analysis for MeeBot

## Current Architecture

MeeBot uses an **OpenAI-compatible API format** throughout [`src/agents/llm.py`](src/agents/llm.py). Every LLM call goes through:

```
POST {api_base}/chat/completions
Authorization: Bearer {api_key}
Content-Type: application/json
{
  "model": "...",
  "messages": [...],
  "max_tokens": N,
  "temperature": 0.85
}
```

This means any provider with an OpenAI-compatible endpoint is a **drop-in replacement** — just change `api_base` and `api_key`.

---

## Groq Free Tier — Current Default

| Model | Free Limit | Context | Quality |
|-------|-----------|---------|---------|
| `llama-3.1-8b-instant` | **14,400 req/day** · 6K tok/min | 128K | Decent for routine chat |
| `llama-3.3-70b-versatile` | **1,000 req/day** · 30 tok/min | 128K | Good for reflections/plans |

### Pros
- ✅ **Zero code changes** — fully OpenAI-compatible, drop-in
- ✅ Extremely fast inference (LPU hardware — ~50ms per response)
- ✅ 14,400 req/day on 8B = ~1 Mee chatting every 6 seconds non-stop
- ✅ Users can bring their own API key (Fernet-encrypted at rest)
- ✅ Already wired in `.env.example`, `docker-compose.yml`, and every user creation flow
- ✅ Circuit breaker + retry logic already battle-tested for Groq's rate limits

### Cons
- ❌ Llama 3.1 8B is mediocre for nuanced roleplay — repetitive, less creative
- ❌ 70B model only 1,000 req/day shared across all Mees
- ❌ 6,000 tok/min on 8B — long prompts hit this quickly with 8 Mees
- ❌ Llama models tend toward formulaic "as an AI" style unless heavily prompted

---

## Google AI Studio (Gemini) Free Tier

| Model | Free Limit | Context | Quality |
|-------|-----------|---------|---------|
| `gemini-2.0-flash` | **1,500 req/day** · 60 req/min | 1M tokens | Excellent — near GPT-4 level |
| `gemini-1.5-flash` | **1,500 req/day** · 60 req/min | 1M tokens | Very good |
| `gemini-1.5-pro` | **50 req/day** · 32K tok/min | 2M tokens | Excellent |

### Pros
- ✅ **Gemini 2.0 Flash is dramatically better** than Llama 3.1 8B at roleplay, creativity, and staying in character — this directly improves Mee believability
- ✅ 1M token context window — entire conversation history fits, no truncation needed
- ✅ 60 requests/minute — higher burst capacity than Groq's 6K tok/min
- ✅ Free tier is genuinely free — no credit card needed for Flash models
- ✅ Google AI Studio free quota is per-API-key, not IP-based

### Cons
- ❌ **NOT OpenAI-compatible** — uses Google's own REST API format
  - `POST https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent`
  - Different payload structure, different response format
  - Requires significant code changes to `LLMClient` in [`src/agents/llm.py`](src/agents/llm.py)
- ❌ **1,500 req/day** — enough for ~1 Mee per minute, but much less than Groq's 14,400
- ❌ **No built-in "quality model" routing** — would need to implement separate provider logic for fast vs quality calls
- ❌ **User-brought keys harder** — Google requires GCP project-level API keys, less user-friendly than Groq's one-click key generation
- ❌ Higher latency per request (~1-2s vs Groq's ~50ms)
- ❌ Would need to maintain two API code paths or completely replace the OpenAI interface

---

## Quantitative Comparison

| Factor | Groq (8B) | Groq (70B) | Gemini 2.0 Flash |
|--------|-----------|------------|-------------------|
| Requests/day | **14,400** | 1,000 | 1,500 |
| Requests/min | ~100 | 0.5 | **60** |
| Context window | 128K | 128K | **1M** |
| Roleplay quality | 5/10 | 7/10 | **9/10** |
| Latency | **~50ms** | ~300ms | ~1-2s |
| Code changes needed | None | None | **Heavy** |
| User-brought keys | ✅ Easy | ✅ Easy | ❌ Complex |
| OpenAI-compatible | ✅ Yes | ✅ Yes | ❌ No |

---

## The Real Question: Volume vs. Quality

The core tradeoff is:

> **Groq gives you 10x more requests but a worse model.**
> **Gemini gives you a much better model but 10x fewer requests.**

For a Tomodachi-style simulation where Mees chat all day, **volume matters** — each Mee ticks every 3-8 minutes, and with 5-10 Mees that's potentially hundreds of API calls per hour. Groq's 14,400/day gives you headroom. Gemini's 1,500/day is tighter.

---

## Recommended Strategy: **Hybrid, Not Swap**

The smartest approach is **not to replace Groq but to complement it**:

### Tier 1 — Routine Chat (Groq 8B — 14,400 req/day)
Use `llama-3.1-8b-instant` via the existing OpenAI-compatible code path for:
- Ambient conversation
- Quick reactions
- Wander decisions
- Social initiative checks

### Tier 2 — Quality Moments (Gemini 2.0 Flash — 1,500 req/day)
Add Google AI Studio as a secondary provider for:
- Reflections and morning recaps
- Plan generation
- Confessions and deep relationship moments
- Mood updates
- Any `complete_quality()` call

### Tier 3 — User-Brought Keys (Stays Groq)
Users who create their own Mee get a Groq key — it's simpler, doesn't require GCP setup, and the Fernet encryption is already built.

---

## Implementation Effort

Adding Gemini as a secondary provider requires:

1. **New Gemini client class** in [`src/agents/llm.py`](src/agents/llm.py) — different HTTP payload/response format
2. **Provider routing** in `LLMClient` — determine which provider to use per call type
3. **New env vars** — `GOOGLE_API_KEY`, `GOOGLE_API_MODEL`
4. **`complete()` vs `complete_gemini()`** — separate methods for Gemini-only calls
5. **No changes needed** to [`src/agents/agent.py`](src/agents/agent.py), [`src/agents/memory.py`](src/agents/memory.py), [`src/commands/manage.py`](src/commands/manage.py), or [`main.py`](main.py) — they all call `llm.complete()` / `llm.complete_quality()` already

### Files to modify
| File | Change |
|------|--------|
| [`src/agents/llm.py`](src/agents/llm.py) | +80 lines for Gemini client class + routing logic |
| [`.env.example`](.env.example) | +2 lines for `GOOGLE_API_KEY`, `GOOGLE_API_MODEL` |
| [`docker-compose.yml`](docker-compose.yml) | +2 env passthrough entries |

---

## Bottom Line

| Question | Answer |
|----------|--------|
| Should you **replace** Groq with Google AI Studio? | **No** — you'd lose volume and user-brought-key simplicity |
| Should you **add** Google AI Studio as a quality tier? | **Yes** — Gemini 2.0 Flash for reflections/plans dramatically improves depth |
| Is Groq free tier "enough"? | **Yes, currently** — the 8B model handles chat fine, and the optimizations we made (dynamic tokens, chat dedup, topic fatigue) keep request count manageable |
| What's the single biggest quality improvement? | **Switching quality_model from 70B to Gemini 2.0 Flash** for reflections, plans, and mood updates |

The Groq-only approach is **already working**. Adding Gemini as a secondary quality provider is a nice-to-have upgrade, not a necessity.
