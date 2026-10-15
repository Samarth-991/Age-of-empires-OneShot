# AoE2 Coaching Agent — Project Constitution

**Version**: 2.0.0 · **Ratified**: 2026-06-23 · **Last Amended**: 2026-06-28

This document is the supreme governing artifact for this project. Every spec, technical plan, and task that follows must conform to it. Where a future plan conflicts with an Article below, the plan is wrong and must be revised — not the other way around. Amending an Article requires the explicit process in **Governance** at the end of this file.

---

## Preamble — what this system is, and is not

This project builds a **user-assistance coaching agent** for Age of Empires II: Definitive Edition. It watches a human's match and tells *them* what to do. It does not play the game.

| This system IS | This system is NOT |
|---|---|
| A read-only observer of the screen | An input-automation bot |
| An advisor the player can take or leave | A player-replacement or autopilot |
| Grounded in structured game data | A source of invented stats, costs, or counters |
| Built for single-player vs. AI bots | Built or tested for ranked multiplayer |

Any spec, plan, or line of code that moves the system toward the right-hand column is out of constitutional bounds, regardless of how the feature request is phrased ("just nudge the camera," "auto-click the idle-villager button," etc.). If a future request implies controlling the game rather than observing it, that request requires a constitutional amendment first, not a quiet exception in the code.

---

## Article I — The Assistant, Not the Player

The system's sole output is *information for a human*: text, a visual overlay, or audio. It never wins or loses the game on the player's behalf. Every other Article in this document exists to protect this one.

## Article II — Read-Only Perception, Zero Input Automation

The system MUST NOT send synthetic mouse, keyboard, or window-message input to the game process or OS, under any circumstance — including during an emergency (Article VI). Its only sensory input is the screenshot pixel buffer; its only access to the game is *looking*, never *touching*. This also means no memory-reading, no process injection, and no reverse-engineered save-state access — screenshots only.

## Article III — Dual-Loop Architecture

Perception and strategy are two independent, asynchronously-running loops, not one monolith:

1. **Perception Loop** — `Screenshot → VLM (Ollama gemma4:latest) → GameStateSnapshot`. Fast, cheap, frequent. Its only job is turning pixels into the structured entity list defined in Article IV.
2. **Strategist Loop** — `Screenshot (+ recent GameStateSnapshot history) → Strategist LLM (Azure OpenAI) → Goals + Resource Readings`. Slower, heavier, reasoning-focused. Its job is turning the *trend* of game state into a dynamic plan.

These loops MUST be decoupled in both code and failure domain: a slow, rate-limited, or temporarily failing Strategist call must never block, stall, or degrade the Perception Loop's cadence, and vice versa. They are allowed to run at different intervals and are allowed to fail independently. Neither loop may call the other synchronously in its hot path.

## Article IV — The Entity Contract

`GameStateSnapshot` (and its nested `DetectedUnit`) is the ratified, binding output contract of the Perception Loop:

```python
class DetectedUnit(BaseModel):
    label: str            # "knight", "villager", "skirmisher", ...
    owner: Literal["self", "enemy", "unknown"]
    approx_count: int

class GameStateSnapshot(BaseModel):
    age: Literal["Dark", "Feudal", "Castle", "Imperial"]
    population: int | None
    pop_cap: int | None
    resources: dict[str, int | None]   # {"food": 300, "wood": 150, "gold": 0, "stone": 0}
    visible_units: list[DetectedUnit]
    visible_buildings: list[str]
    idle_villagers_visible: bool
    threat_level: Literal["none", "low", "high"]
    notes: str
```

No component downstream of the Perception Loop may consume raw VLM text — every consumer reads this schema, and only this schema. Fields the VLM cannot confidently read MUST be left `null`/empty rather than guessed (a wrong-but-confident value is worse than an honest unknown, since nothing downstream double-checks it). Changing this contract is a schema migration, not a casual edit — it ripples into the Strategist Loop, the alarm system, and the knowledge-grounding layer below.

## Article V — Deterministic Alarms Are Code, Not Cognition

Resource and economy alarms (e.g. food dropping below a configured floor such as 200, prolonged idle villagers, an unanswered `threat_level: "high"`) MUST be implemented as plain, deterministic, unit-testable functions over the latest `GameStateSnapshot` and recent resource history — never as an LLM call.

Rationale: alarms exist precisely *because* the Strategist Loop is slow and can fail or lag. An alarm that itself depends on a model call inherits the same latency and failure modes it's supposed to be a safety net against. Thresholds (the `200` above, idle-time limits, etc.) MUST be externally configurable, not hardcoded magic numbers — but the *evaluation* of those thresholds stays pure Python, fast and synchronous, runnable every Perception Loop tick regardless of what the Strategist Loop is doing.

## Article VI — The Emergency Injection Channel

There is exactly one delivery surface for urgent information reaching the player, fed by two producers:

1. The **Strategist Loop**, on its own cadence — every time it concludes an analysis cycle and produces a final strategy/goal update, that conclusion is injected into the overlay automatically. The player never has to ask for it.
2. The **Alarm System** (Article V) — the instant a deterministic threshold is breached, it injects directly into the same channel, out of band, without waiting for the Strategist Loop's next tick.

An emergency MUST NOT be silently queued behind whatever is currently displayed. The player must see *something* visibly change within at most one render cycle of any alarm firing. This Article mandates that the channel exists and that both producers can pre-empt it; it deliberately does not mandate the exact visual treatment or tie-breaking rule when both producers fire simultaneously — that level of detail belongs in a technical spec, not the constitution.

## Article VII — Knowledge Grounding Over Improvisation

The Strategist Loop's reasoning MUST be grounded in structured reference data — civilizations, buildings, technologies, and units — not improvised from the model's general training knowledge. Every concrete factual claim in a generated strategy (a bonus, a cost, a counter-unit relationship, a tech prerequisite) must be traceable to an entry in this knowledge base; free-form strategic judgment built on top of those facts is expected and welcome, but the facts themselves are not the model's to invent.

This knowledge base is data, not prose: structured records for civilizations (bonuses, unique units/techs, team bonus), buildings (costs, function, prerequisites), technologies (effects, costs, age-gating), and units (stats, costs, counter relationships) — mirroring the shape of a relational dataset rather than a pile of markdown essays, so the Strategist can look up a fact instead of recalling it. Where this data needs filling in, prefer importing/adapting structured data over hand-writing free text.

## Article VIII — Model Providers: Split by Loop, Behind One Abstraction

The two loops are ratified on different providers, reflecting their different latency/cost/reasoning-depth needs: the **Perception Loop** runs on a local model via Ollama (`gemma4:latest`) — cheap and frequent enough that a hosted vision call per tick would be wasteful; the **Strategist Loop** runs on **Azure OpenAI** — reasoning-heavy and infrequent enough that hosted-model quality is worth the cost. Both MUST still go through a single provider-agnostic interface per loop (one client construction path, one place model/deployment names are configured) rather than being called ad hoc from multiple modules. The point of the abstraction doesn't change just because the providers split: it's insurance against the *next* Article VIII amendment costing a rewrite instead of a config change, whether that amendment swaps the Perception model, the Strategist model, or both.

## Article IX — Minimal Orchestration, Maximal Reasoning-in-Model

The function that periodically updates goals is intentionally minimal: gather the latest screenshot, resource readings, and entity list; call the Strategist model; return its structured output. Strategic reasoning belongs inside the model call, not hand-coded as branching heuristics in Python around it. The deterministic Alarm System (Article V) is the explicit, narrow exception to this principle — and it is an exception precisely because alarms are safety-critical and must not depend on model availability, not because hand-coded heuristics are generally preferred over model reasoning elsewhere in this system.

## Article X — Transparency & Traceability of Advice

Every goal or strategic recommendation the Strategist Loop emits must be traceable back to (a) the `GameStateSnapshot` data that motivated it and (b) the knowledge-base facts it leaned on. "Black-box" advice with no inspectable basis undermines the coaching purpose of the whole system — a player who can't tell *why* the agent suggested something can't learn from it, and can't catch the agent being wrong.

## Article XI — Non-Intrusive, Player-Controlled Delivery

The overlay/notification layer renders advice *over or alongside* the game window without ever capturing input focus, blocking the view of critical UI (resource bar, minimap), or pausing/intercepting gameplay. The player can always ignore it. Audio cues, if added later, must be similarly non-blocking. This Article exists to keep Article I true in practice, not just in architecture: a coaching tool that gets in the way of playing has failed at being a coaching tool.

## Article XII — Phased, Spec-Driven Development

Implementation proceeds in gated phases (perception, then strategist reasoning, then alarms, then the emergency channel, then knowledge-base integration, then the overlay/delivery layer) each with its own spec and plan derived from this constitution. A later phase does not begin until the phase it depends on is demonstrably working against real screenshots — this constitution governs *what must always be true*; the accompanying spec/plan documents govern *what gets built in what order*.

---

## Governance

- This constitution supersedes any prior informal design notes where they conflict with it — most notably, it replaces the earlier per-civilization markdown knowledge-file approach with the structured civs/buildings/techs/units dataset described in Article VII, and replaces an open-ended model-provider stance with the split-provider decision in Article VIII (local Ollama for Perception, Azure OpenAI for the Strategist).
- **Amendment process**: a change to any Article requires (1) a stated reason the current Article is insufficient, (2) an explicit replacement text, and (3) a version bump below. Silent drift — code that violates an Article without an amendment — is a bug, not a precedent.
- **Versioning**: MAJOR for a removed/redefined Article, MINOR for a new Article, PATCH for clarifying wording with no behavioral change.

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-06-23 | Initial ratification: assistant-not-bot scope, dual-loop architecture, entity contract, deterministic alarms, emergency injection channel, structured knowledge grounding, Azure OpenAI provider decision. |
| 2.0.0 | 2026-06-28 | Article VIII redefined: split model providers by loop — Perception Loop moves from Azure OpenAI to a local Ollama vision model (`gemma4:latest`); Strategist Loop remains on Azure OpenAI. |