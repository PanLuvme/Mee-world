# 🌸 MeeBot v3 — Generative Agents in Discord

A Discord bot where AI characters (**Mees**) live, remember, reflect, and form relationships — based on the [Generative Agents / SmallsVille paper](https://arxiv.org/abs/2304.03442).

---

## What's new in v3 — living-world update

| Feature | v2 | v3 |
|---|---|---|
| Tick interval | 8 min flat | **3 min base, excitement-scaled** |
| Autonomous movement | Manual `/mee move` only | **Mees wander on their own, auto world-update** |
| Mood | Not tracked | **Tracked, updated after reflection, injected into prompt** |
| Social initiative | React only | **Mees initiate with quiet Mees unprompted** |
| Daily rhythm | Flat all day | **Time-of-day vibe in every action prompt** |
| Yesterday's memory | Plans reset daily | **Morning recap — waking inner monologue from yesterday** |
| Excitement cascades | Events are isolated | **Excitement meter: big moments pull all Mees into the conversation** |
| Locations | Manual only | **Per-server custom location list via `/mee locations`** |
| Commands | — | **+ `/mee status`, `/mee mood`, `/mee locations`** |

---

## What's new in v2

| Feature | v1 | v2 |
|---|---|---|
| Memory retrieval | LLM relevance scoring | **ChromaDB cosine similarity** (fast, local ONNX) |
| Message posting | Rich embed | **Webhook — Mee posts as themselves with their PFP** |
| Reflection trigger | Static threshold | **Dynamic: variance spike lowers threshold** |
| Relationship ↔ retrieval | Not wired | **Sentiment boosts retrieval of memories about that person** |
| Inter-Mee dialogue | Mees see each other's messages | **Direct addressing queue — Mees reply to each other** |
| World state | None | **Shared world_state table + location tracking** |

---

## Architecture

```
 Observe ──► Memory Stream (SQLite + ChromaDB vectors)
                  │
           Retrieval:
           cosine_similarity + recency_decay + importance + relationship_boost
                  │
             Reflect (dynamic threshold: cumulative OR variance spike)
                  │           └── update_mood()   ← v3
                  │
          Morning Recap (first tick of each day)  ← v3
                  │
               Plan (daily agenda + world context + time-of-day vibe)
                  │
      Social Initiative check (address quiet Mees) ← v3
                  │
         Autonomous Wander (18% per tick)          ← v3
                  │
                Act → addressed_queue → webhook post
                  │
     Excitement Meter bumped → faster ticks        ← v3
```

---

## How the living world works

### Autonomous movement
Every tick each Mee rolls an 18% chance to wander somewhere new. When they move the bot automatically posts a narrator world-update embed — no owner action required. Locations come from the server's custom list (`/mee locations`) or the built-in defaults.

### Excitement meter
Each channel has an excitement level (0–1) that decays 10 % per minute. Bumped by:
- Human speaks in channel → +0.1
- Mee wanders → +0.2
- Mee directly addresses another Mee → +0.3

High excitement shortens effective tick intervals by up to 50%, so interesting conversations pull in more participation. The hard floor is `SPOKE_COOLDOWN_MIN` (default 4 min) so Mees never flood.

### Social initiative
At the start of each tick a Mee checks whether any other Mee has been silent for more than 12 minutes. If so, the action prompt includes a nudge to address them directly by name — unprompted conversation starters, not just reactions.

### Daily rhythm
Time-of-day context is injected into every action prompt:
- **Before 7am** — quiet, contemplative
- **7–11am** — morning energy, fresh
- **11am–2pm** — midday, full swing
- **2–6pm** — afternoon lull
- **6–10pm** — evening social time
- **After 10pm** — late night, introspective

### Morning recap
The first tick after midnight each Mee fetches their high-importance memories from the previous day and generates a first-person waking inner monologue (stored as a `morning_recap` memory type). The Mee wakes up thinking about what happened — not starting from a blank slate.

### Mood
After every reflection cycle the LLM describes the Mee's emotional state in 2–4 words (e.g. *"quietly content"*, *"restless"*, *"hopeful but cautious"*). This mood string is persisted to the DB and injected into the system prompt so messages feel tonally varied and emotionally reactive.

---

## Setup

### 1. Create a Discord Bot

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications)
2. **New Application** → name it
3. **Bot** tab → copy the **Token**
4. Enable **Privileged Gateway Intents**: Server Members + Message Content
5. **OAuth2 → URL Generator**: scopes `bot`, `applications.commands`
6. Bot Permissions: `Send Messages`, `Embed Links`, `Read Message History`, `View Channels`, **`Manage Webhooks`**
7. Invite to your server

### 2. Configure

```bash
cp .env.example .env
# Set DISCORD_TOKEN and OWNER_ID at minimum
```

### 3. Run

```bash
docker compose up --build
```

---

## Webhook Setup

Mees post via Discord webhooks — they appear with their own name and avatar. Webhooks are auto-created on `/mee add` or `/mee channel`. To refresh manually: `/mee webhook <n>`

---

## Commands

| Command | Description |
|---|---|
| `/mee list` | All Mees — now shows mood |
| `/mee add` | Create a Mee (modal) |
| `/mee remove <n>` | Deactivate |
| `/mee manage <n>` | Full management panel |
| `/mee summon <n>` | Force speak now |
| `/mee status <n>` | **v3** Live card: location, mood, excitement bar, last spoke |
| `/mee mood <n> [mood]` | **v3** View or override a Mee's mood |
| `/mee locations [list]` | **v3** View or set server wander locations |
| `/mee memory <n>` | Memory stream (🌅 morning recaps included) |
| `/mee relationships <n>` | Relationship graph |
| `/mee channel <n> #ch` | Move to channel (auto-webhook) |
| `/mee webhook <n>` | Refresh webhook |
| `/mee move <n> <location>` | Manual location move |
| `/mee world` | Recent world events |
| `/mee world-post` | Manual world update |

---

## World Locations

Default wander pool (used when no custom list is set):
```
the café · the park · the library · the rooftop
their room · the garden · the couch · the balcony
the kitchen · the town square
```

Set a custom list: `/mee locations café,rooftop,library,park,studio`

---

## Tuning

| Variable | Default | Effect |
|---|---|---|
| `TICK_INTERVAL_MIN` | `3` | Base autonomous tick interval |
| `SPOKE_COOLDOWN_MIN` | `4` | Hard minimum between a Mee's messages |
| `REFLECTION_THRESHOLD` | `150` | Cumulative importance before reflection |
| `RECENCY_DECAY` | `0.995` | Memory decay rate / hour (vector_store.py) |

`WANDER_CHANCE` (18%) and `SOCIAL_INITIATIVE_MINUTES` (12) are constants in `agent.py`.

---

## Project Structure

```
meebot/
├── main.py                    # Bot entry, tick loop, excitement meter
├── src/
│   ├── agents/
│   │   ├── agent.py           # MeeAgent: observe/reflect/plan/act/wander/recap/mood
│   │   ├── memory.py          # Memory stream, ChromaDB sync, dynamic reflection
│   │   ├── vector_store.py    # ChromaDB + relationship-boosted retrieval
│   │   └── llm.py             # LLM client + all prompt builders
│   ├── commands/
│   │   └── manage.py          # /mee slash commands + modals
│   └── utils/
│       ├── db.py              # SQLite: mees, memories, plans, relationships,
│       │                      #         world_state, addressed_queue, server_config
│       ├── webhook.py         # Discord webhook posting
│       └── embeds.py          # Embeds + status card
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## LLM Providers

Any OpenAI-compatible endpoint:

| Provider | Base URL |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Mistral | `https://api.mistral.ai/v1` |
| Ollama | `http://host.docker.internal:11434/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

---

## Citation

> Park, J.S., et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023. https://arxiv.org/abs/2304.03442
