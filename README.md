# AoE2 Coaching Agent

> A real-time AI coaching assistant for Age of Empires II: Definitive Edition.
> It watches your screen. You play the game. It tells you exactly what to do next.

---

## Why This Project Exists

Age of Empires II is one of the most strategically deep real-time strategy games ever made. Fifty civilizations, over a thousand unique units, hundreds of buildings, and a combat system built on layered rock-paper-scissors relationships that take years to fully internalize. A first-time player sits down and is immediately overwhelmed — not because the game is unfair, but because the knowledge gap between a beginner and an intermediate player is enormous.

The questions that lose games for rookies are almost always the same:

- *When do I stop building villagers and start training an army?*
- *My opponent has knights everywhere — what beats knights?*
- *I keep running out of food mid-battle. What am I doing wrong?*
- *Should I advance to Castle Age now or wait?*

This project was built to answer those questions **in real time**, without the player ever needing to pause or alt-tab. The agent reads the game screen, extracts a structured snapshot of the current state, consults a grounded knowledge base of actual game data, and delivers concise, actionable advice — every few seconds, automatically.

The intention is not to play the game for the player. It is to be the experienced friend sitting beside them who says *"you have idle villagers — fix that right now"* or *"their civilization gets stronger cavalry; get spearmen up before they push"*. Over time, hearing that advice repeatedly is how a player learns. The agent is a coach, not a crutch.

---

## What It Does (and Does Not Do)

| The agent IS | The agent is NOT |
|---|---|
| A read-only observer of the screen | A bot that controls the game |
| An advisor whose tips you can take or leave | A replacement for the player's own skill |
| Grounded in structured game data | A source of invented stats or made-up bonuses |
| Always running quietly in the background | Ever capturing keyboard or mouse input |

The system is constitutionally constrained to screenshot-only perception. It has no access to game memory, no process injection, no input automation — it only looks at pixels, and only speaks to the player through text.

---

## How It Helps a First-Time Player

### No memorization required

The agent knows every civilization bonus, every unit's HP, attack, and cost, every counter relationship. When it tells a new player *"build three Spearmen — they cost 35 food and 25 wood each and hard-counter the six Knights your opponent just trained"*, those numbers come from the actual game database, not guesswork. The player gets to focus on playing; the agent handles the encyclopedic knowledge.

### It catches the classic rookie mistakes

- Villagers sitting idle? The alarm fires the instant the icon appears on screen.
- Food below 200 while actively training units? Immediate warning.
- Population cap approaching? Told to build houses before production stalls.
- Enemy units spotted near the base on the minimap? High-threat alert, right now — not after the next Strategist cycle.

### It knows the matchup, not just the player

At session start, the player is asked for **both their own civilization and their opponent's**. The agent immediately understands the matchup dynamics: what the opponent's unique units and bonuses mean for the player's strategy, which advantages to press, and what threats to anticipate. A Teutons player facing Vietnamese archers gets fundamentally different advice than a Mongols player facing Franks.

### It explains the reasoning

Every strategy update includes a rationale — *"you are in Feudal Age with 80 wood and an Archery Range; Britons' extended archer range makes an early archer rush your strongest opening in this matchup"* — so the player understands why, and over time stops needing the coach for that situation.

### It remembers the session

The agent maintains a rolling window of the last 20 game state snapshots. Before each strategy cycle it can look back across that history to understand trends — whether food is steadily declining, whether a threat was already identified, whether its previous advice was relevant. Advice adapts as the game evolves rather than repeating the same opening tips in the Imperial Age.

---

## Architecture

The system runs independent asynchronous loops. They are deliberately decoupled: a slow or temporarily failing Strategist call never delays the Perception Loop, and a heavy VLM call never blocks an alarm from firing. An optional third loop, the [Nimble fast coach](#nimble-fast-coach), gives per-action advice within seconds while the Strategist is still reasoning.

```
┌──────────────────────────────────────────────────────────────────┐
│  PERCEPTION LOOP  (every 3 seconds)                              │
│                                                                  │
│  Screenshot ──► gemma4:latest (Ollama VLM)                       │
│                       │                                          │
│                       ▼                                          │
│               GameStateSnapshot                                  │
│         (age, resources, units, buildings,                       │
│          idle villagers, threat level)                           │
│                       │                                          │
│                       ▼                                          │
│           Deterministic Alarm Checks                             │
│         (pure Python — no LLM involved)                          │
└───────────────────────┬──────────────────────────────────────────┘
                        │  snapshot + rolling history (last 20)
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│  STRATEGIST LOOP  (every 15 seconds)                             │
│                                                                  │
│  Supervisor Agent  (llama3.1:latest)                             │
│    │                                                             │
│    ├── consult_matchup_analyst  ◄─── MatchupAnalyst sub-agent    │
│    ├── consult_civ_analyst      ◄─── CivilizationAnalyst sub-agent│
│    ├── consult_counter_analyst  ◄─── CounterAnalyst sub-agent    │
│    ├── consult_unit_stats       ◄─── UnitStatsAdvisor sub-agent  │
│    └── consult_building_advisor ◄─── BuildingAdvisor sub-agent   │
│                                                                  │
│  Strategy text ──► Console overlay / video panel                 │
└──────────────────────────────────────────────────────────────────┘
```

The rolling snapshot history feeds the Strategist each cycle, enabling trend-aware reasoning across the match rather than treating each update as independent.

---

## Nimble Fast Coach

The Strategist's multi-agent reasoning can take a long time per update on local hardware. The fast coach fills that gap with [Nimble](https://ollama.com/library/nimble), a 9B decision model that doesn't reason step by step. It picks an answer from a fixed set of options and returns the probability of each option.

Every time perception produces a new snapshot, the fast coach asks Nimble:

- **One question per action relevant to the current age:** "train Spearman?", "build Mill?", "research Loom?", "advance to Feudal Age?". Each is answered `now`, `soon` or `skip`, and the answers are ranked into the build/train/research recommendations.
- **A few situation questions:** overall focus, which resource to gather, military urgency, which unit counters the visible enemy (only asked when enemies are seen), and whether the deep Strategist is needed.

Facts the code can work out itself aren't asked. Idle villagers, high threat and near pop cap are handled by fixed rules, and affordability ("short 100 food") is computed and written into each question. When Nimble confidently says a situation needs matchup analysis, it wakes the Strategist early instead of waiting for its next cycle.

| File (`src/aoe2_agent/strategy/nimble/`) | Purpose |
|---|---|
| `catalog.py` | Units, buildings, technologies and age-ups with cost, prerequisite and the ages where each is worth considering |
| `questions.py` | Builds the per-age question plan |
| `state.py` | Converts a `GameStateSnapshot` into the compact state Nimble judges |
| `client.py` | Calls Ollama's `/v1/systemone` endpoint in small batches |
| `advisor.py` | `NimbleAdvisor.advise(snapshot)` → ranked `NimbleAdvice` (Markdown, report or JSON) |
| `loop.py` | Async loop run alongside perception and the Strategist |

Settings live in `config.py` (`NIMBLE_ENABLED`, `NIMBLE_INTERVAL_SECS`, `NIMBLE_BATCH_SIZE`, `NIMBLE_MAX_QUESTIONS`, …). The fast coach can also be switched off on the web setup screen.

```python
from aoe2_agent.strategy.nimble import NimbleAdvisor

advice = NimbleAdvisor("Mongols", "Vietnamese").advise(snapshot)
print(advice.to_markdown())
```

---

## The Deep Agents Supervisor Architecture

The Strategist is built on [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) (`deepagents`) using the **supervisor + subagents-as-tools** pattern. Instead of one monolithic agent that tries to know everything, a Supervisor delegates domain questions to five specialized sub-agents, each an expert in one area.

### How it works

Each sub-agent is built with `create_deep_agent`, given a focused system prompt and access to exactly one knowledge base function. It is then wrapped as a LangChain `@tool`:

```
@tool
def consult_matchup_analyst(query: str) -> str:
    """Consult the Matchup Analyst for player vs opponent civ dynamics."""
    result = matchup_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return extract_last_content(result)
```

The Supervisor receives the full game state and a set of five tool handles. Its instructions require it to **always call `consult_matchup_analyst` first** to establish the civ-vs-civ context, then pull in whatever specific facts it needs from the other sub-agents before synthesizing a final strategy update.

### The five sub-agents

| Sub-agent | Expertise | Knowledge base function |
|---|---|---|
| **MatchupAnalyst** | Player civ vs opponent civ strengths, weaknesses, threats | `get_civilization_info` (called for both civs) |
| **CivilizationAnalyst** | Any single civilization's bonuses, unique units, unique techs | `get_civilization_info` |
| **CounterAnalyst** | Unit counter relationships (what beats what, and why) | `get_unit_counters` |
| **UnitStatsAdvisor** | Unit HP, attack, armour, cost, cost-efficiency comparisons | `get_unit_stats` |
| **BuildingAdvisor** | Building costs, HP, category, production function | `get_building_info` |

Each sub-agent has its own context window. Heavy intermediate reasoning in a sub-agent is isolated from the Supervisor's context, keeping the Supervisor's working memory compact and reducing token usage across the session.

### Why this design matters for a new player

Every factual claim the Supervisor makes is traceable to a specific sub-agent tool call, which is in turn traceable to a structured record in the knowledge base. When the agent says *"Spearmen cost 35F/25W and hard-counter Knights"*, the UnitStatsAdvisor looked that up from a JSON record and the CounterAnalyst confirmed the matchup. The model's training weights are not the source of truth — the knowledge base is.

---

## Memory

The agent uses two complementary memory layers that work together across a session.

### AGENTS.md — Persistent session memory

`AGENTS.md` is loaded by the Deep Agent as a memory file at startup (via the `memory=` parameter in `create_deep_agent`). Unlike skills which are loaded on demand, this file is **always fully loaded** before the first strategy cycle. It contains:

- The agent's role definition: advisor, not bot
- Critical game constants (age advancement costs, population cap, alarm thresholds, rock-paper-scissors rules)
- The required output format for every strategy update
- Reminders about what facts must come from tool lookups

This ensures the Strategist starts every session already calibrated — it does not need to rediscover that food below 200 is dangerous, or that it must use the structured output format, each time it is invoked.

### Rolling snapshot history — In-session memory

The Perception Loop maintains a rolling deque of the last 20 `GameStateSnapshot` readings in memory. This history is passed to the Strategist on every cycle alongside the current snapshot.

This enables trend reasoning that a single snapshot cannot provide:

- Food was 900 three cycles ago and is 280 now — that rapid drain triggers a `RAPID_FOOD_DROP` alarm even if the current food level alone would not
- The threat level has been `high` for the last four cycles — the Strategist can escalate its advice accordingly
- Resources have been steadily climbing for ten cycles — the agent can recognize a mature economy and shift advice toward military production

The combination of `AGENTS.md` (what the agent always knows) and snapshot history (what it has observed this session) gives the Strategist both durable knowledge and situational awareness.

---

## Alarm System

Seven deterministic checks run on every Perception Loop tick. These are pure Python functions with no LLM call — they are designed this way specifically because alarms must be fast, reliable, and independent of model availability.

| Alarm | Trigger condition |
|---|---|
| LOW FOOD | Food below 200 |
| LOW WOOD | Wood below 100 |
| LOW GOLD | Gold below 50 (Castle / Imperial Age only) |
| IDLE VILLAGERS | Idle villager icon visible near population counter |
| HIGH THREAT | Enemy units near player base (minimap signal) |
| POPULATION CAP NEAR | `pop_cap − population ≤ 5` |
| RAPID FOOD DROP | Food decreased by more than 300 across the last 3 snapshots |

Alarms inject directly into the display as soon as they fire, without waiting for the next Strategist cycle. All thresholds are externally configurable in `config.py`.

---

## Models

Both models run entirely on the local machine via [Ollama](https://ollama.com). Nothing leaves the machine.

| Loop | Model | Role |
|---|---|---|
| **Perception** | `gemma4:latest` | Multimodal VLM — reads resource bars, unit counts, minimap threat signal from raw pixels |
| **Strategist** | `llama3.1:latest` | Instruction-tuned reasoning model — multi-step tool use, structured output, civ-aware strategy synthesis |
| **Summarizer** | `llama3.1:latest` (temperature=1) | Condenses the full strategy update into a compact overlay card using a Pydantic output schema |
| **Fast coach** | `nimble` (requires Ollama 0.35+) | Scores now / soon / skip for each unit, building, technology and age-up |

```bash
ollama pull gemma4:latest
ollama pull llama3.1:latest
ollama pull nimble
```

> Nimble is about 10 GB. Running it alongside the local vision and strategy models needs a lot of RAM. If Ollama gets killed for running out of memory under WSL, raise the memory limit in `.wslconfig`, set `OPENROUTER_API_KEY` so vision and strategy run on OpenRouter, or switch off the fast coach.

---

## Knowledge Base

The knowledge base is structured JSON — records designed to be looked up, not free text to be interpreted.

| File | Contents |
|---|---|
| `civilizations/civs.json` | 50 civilizations with bonuses, unique units, and unique technologies |
| `units/units.json` | 1,102 unit records — HP, attack, melee armor, pierce armor, cost, class |
| `counters/counters.json` | Unit-type counter matrix with `strong_vs`, `weak_vs`, and strategic notes |
| `buildings/building.json` | 670 building records — HP, category, garrison capacity, cost |

The knowledge base is loaded once at session start and cached in memory for the duration of the run.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangChain Deep Agents (`deepagents`) |
| LLM runtime | Ollama (local, no cloud) |
| Vision model | `gemma4:latest` |
| Strategy / summarizer model | `llama3.1:latest` |
| Sub-agent orchestration | `langchain-core` `@tool` + `create_deep_agent` |
| Structured output | Pydantic v2 (`GameStateSnapshot`, `StrategySummary`) |
| Screen capture | `mss` + `Pillow` + `imagehash` (perceptual hash deduplication) |
| Video annotation | `cv2.hconcat` — side-by-side panel, game frame untouched |
| Fast decision model | `nimble` via Ollama `/v1/systemone` |
| Web dashboard | `fastapi` + `uvicorn` + Server-Sent Events |
| CLI | `typer` |
| Console display | `rich` (Live layout — alarms / strategy / game state panels) |
| In-session memory | Rolling deque of last 20 `GameStateSnapshot` readings |
| Persistent memory | `AGENTS.md` loaded via Deep Agents `memory=` parameter |
| Async runtime | `asyncio` (dual-loop, fully decoupled) |

---

## CLI

Run everything through `run.sh` from the project root. It sets up the import paths the code needs: it runs from `src/aoe2_agent` and adds `src` to `PYTHONPATH`.

```bash
# Web dashboard — open http://127.0.0.1:8000
./run.sh web
./run.sh web --port 8080

# Interactive console session — asks for both civilizations
./run.sh start

# Provide civilizations directly
./run.sh start --civ Teutons --opponent Vietnamese

# Verify screenshot capture
./run.sh test-capture

# List all 50 available civilizations
./run.sh list-civs

# Smoke-test the knowledge base tools
./run.sh test-kb Mongols Knight
```

Without the script, the equivalent is:

```bash
cd src/aoe2_agent
PYTHONPATH=.. python cli.py web
```

### Web dashboard

`./run.sh web` starts a FastAPI server. In the browser, pick both civilizations, optionally give the game executable path, and choose whether to enable the Nimble fast coach. The dashboard shows:

- **Fast Coach · Nimble:** the priority instruction plus "Do now" and "Soon" lists of units, buildings and technologies, with probabilities and costs
- **Strategist:** the latest full strategy update (stays on screen until a new one arrives)
- **Alarms** and the latest **game state** read from the screen

| Endpoint | Purpose |
|---|---|
| `GET /api/civs` | Civilization list for the dropdowns |
| `POST /api/start` / `POST /api/stop` | Start or stop a coaching session |
| `GET /api/state` / `GET /api/stream` | Current state, one-shot or as Server-Sent Events |
| `POST /api/nimble/advise` | Fast-coach advice for a posted snapshot, without running a session |

---

## Project Structure

```
AOE-Agent/
├── Constitution.md          # Project rules — what the agent must and must not do
├── AGENTS.md                # Deep Agent persistent memory (game constants, output format)
├── requirements.txt
├── run.sh                   # Launcher — ./run.sh web | start | list-civs | …
├── knowledge_base/
│   ├── civilizations/       # 50 civ records
│   ├── units/               # 1,102 unit records
│   ├── counters/            # unit-type counter matrix
│   └── buildings/           # 670 building records
└── src/aoe2_agent/
    ├── config.py            # All thresholds and model names
    ├── schemas.py           # GameStateSnapshot + DetectedUnit (Pydantic)
    ├── cli.py               # Entry point — civ selection + dual-loop start
    ├── perception/
    │   ├── capture.py       # mss screenshot + perceptual-hash dedup
    │   └── loop.py          # Async VLM loop → GameStateSnapshot
    ├── knowledge/
    │   └── tools.py         # 4 KB query functions used by sub-agents
    ├── strategy/
    │   ├── agent.py         # Supervisor + 5 sub-agents architecture
    │   └── nimble/          # Nimble fast coach (catalog, questions, advisor, loop)
    ├── alarms/
    │   └── rules.py         # 7 deterministic alarm checks (no LLM)
    ├── display/
    │   └── overlay.py       # Rich Live: alarms / strategy / state panels
    ├── web/
    │   ├── server.py        # FastAPI app and API endpoints
    │   ├── session.py       # Runs perception, Strategist and Nimble loops
    │   └── static/index.html
    ├── tests/
    │   └── test_nimble.py   # Offline fast-coach tests (no Ollama needed)
    └── notebooks/
        ├── 01_check_screen_capture.ipynb    # Perception loop tests
        ├── 02_test_strategy_agent.ipynb     # Strategy agent tests
        ├── 03_test_strategy_subagents.ipynb # Subagents-as-tools tests
        ├── 04_test_on_video.ipynb           # Frame-by-frame video annotation
        └── 07_test_nimble.ipynb             # Nimble fast-coach scenarios
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) 0.35+ installed and running (on Linux/WSL it usually runs as a systemd service, so there's no need to run `ollama serve`)
- Age of Empires II: Definitive Edition

### Install

```bash
git clone <repo>
cd AOE-Agent-ollama-Nimble
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x run.sh
ollama pull gemma4:latest
ollama pull llama3.1:latest
ollama pull nimble
```

### Verify

```bash
./run.sh test-kb Mongols Knight
./run.sh list-civs
python src/aoe2_agent/tests/test_nimble.py   # offline fast-coach tests
```

### Play

```bash
./run.sh web      # browser dashboard at http://127.0.0.1:8000
# or
./run.sh start    # console overlay
```

Launch AoE2, start a match, and the dashboard or overlay begins updating within seconds of the first screenshot being captured.

---

## License

MIT
