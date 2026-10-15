"""Strategist Loop — Article III of the Constitution.

Architecture: Subagents-as-Tools
---------------------------------
Instead of giving the Supervisor raw KB functions, we create six *specialized
sub-agents*, each an expert on one knowledge domain, and then wrap each one
as a LangChain @tool.  The Supervisor (main Strategist) calls these tools,
and each tool internally runs its own Deep Agent with focused context.

Pattern (mirrors 05_subagents_as_tool.ipynb):

  sub-agent (create_deep_agent) → wrapped as @tool → Supervisor (create_deep_agent)

Sub-agents:
  StrategicReferenceAdvisor — summary.md: counter matrix, build orders, principles (call FIRST)
  CivilizationAnalyst       — civ bonuses, unique units/techs
  CounterAnalyst            — unit counter relationships (deep detail)
  UnitStatsAdvisor          — unit HP/attack/cost comparisons
  BuildingAdvisor           — building costs, categories, functions
  MatchupAnalyst            — player civ vs opponent civ strengths/weaknesses

Supervisor receives the GameStateSnapshot (+ both civilizations) and delegates
domain questions to the six sub-agents, then synthesises a final strategy update.
"""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Optional

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_scaledown import ScaledownSummarizationMiddleware

from config import (
    AGENTS_MEMORY_FILE,
    AGENT_LOG_FILE,
    OPENROUTER_API_KEY,
    OPENROUTER_STRATEGY_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_STRATEGY_MODEL,
    OLLAMA_STRATEGY_NUM_CTX,
    STRATEGIST_INTERVAL_SECS,
)
from knowledge.tools import (
    get_building_info,
    get_civilization_info,
    get_summary_info,
    get_unit_counters,
    get_unit_stats,
)
from schemas import GameStateSnapshot
from strategy.format import (
    format_strategy,
    parse_strategy_response,
    strategy_to_markdown,
    summarize_raw_strategy,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File logger — writes every strategy update to logs/agent.log
# ---------------------------------------------------------------------------

_LOG_FILE = AGENT_LOG_FILE
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
)

_agent_logger = logging.getLogger("aoe2_agent.strategy")
_agent_logger.setLevel(logging.DEBUG)
_agent_logger.addHandler(_file_handler)
_agent_logger.propagate = True


def _log_strategy(player_civ: str, opponent_civ: str, strategy_text: str) -> None:
    """Write a strategy update to the rotating log file."""
    separator = "─" * 60
    _agent_logger.info(
        "\n%s\n  STRATEGY UPDATE  |  %s vs %s  |  %s\n%s\n%s\n%s",
        separator,
        player_civ,
        opponent_civ,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        separator,
        strategy_text,
        separator,
    )


_RESOLVED_MODEL = None


def _resolve_strategy_model():
    """Return the deepagents model string for the Strategist (cached).

    Priority:
    1. OpenRouter (anthropic/claude-sonnet-4-6) — if OPENROUTER_API_KEY is set
       in config.py or in the environment at call time.
    2. Ollama (llama3.1:latest) — local fallback, no API key required.

    The result is cached after the first call so that building the supervisor
    plus its sub-agents resolves (and logs) the model exactly once per process.
    """
    global _RESOLVED_MODEL
    if _RESOLVED_MODEL is not None:
        return _RESOLVED_MODEL

    import os
    key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        os.environ["OPENROUTER_API_KEY"] = key
        _RESOLVED_MODEL = f"openrouter:{OPENROUTER_STRATEGY_MODEL}"
        log.info("Strategist using OpenRouter model: %s", OPENROUTER_STRATEGY_MODEL)
    else:
        from langchain_ollama import ChatOllama
        _RESOLVED_MODEL = ChatOllama(
            model=OLLAMA_STRATEGY_MODEL,
            base_url=OLLAMA_BASE_URL,
            num_ctx=OLLAMA_STRATEGY_NUM_CTX,
            temperature=0,
        )
        log.info(
        "Strategist using Ollama model: %s (no OPENROUTER_API_KEY found — "
        "set OPENROUTER_API_KEY for much faster strategy updates)",
        OLLAMA_STRATEGY_MODEL,
    )
    return _RESOLVED_MODEL

# ---------------------------------------------------------------------------
# Sub-agent system prompts
# ---------------------------------------------------------------------------

_STRATEGIC_REF_PROMPT = """\
You are the Strategic Reference Advisor for Age of Empires II.
Your sole job: answer questions using get_summary_info, which returns the
master strategic reference covering:
  - Unit counter matrix (all major archetypes, strong_vs and weak_vs)
  - Six universal strategic principles (economy, RPS, upgrades, scouting, etc.)
  - Three common build orders with exact villager assignments and timing gates
  - Age advancement costs and research times for all three ages

Call get_summary_info to retrieve the full reference, then answer the query directly
from its content. Do not invent information not present in the reference.

Use this as the strategic backbone that frames every other piece of advice:
  - Before recommending a build order, confirm it matches the reference
  - Before naming a counter unit, cross-check against the counter matrix
  - Before advising on age-up, confirm the player has the required resources
"""

_CIV_PROMPT = """\
You are a Civilization Analyst for Age of Empires II.
Your sole job: answer questions about a civilization's bonuses, unique units,
and unique technologies using get_civilization_info.
Give concise, factual answers — never invent bonuses not in the KB.
"""

_COUNTER_PROMPT = """\
You are a Unit Counter Analyst for Age of Empires II.
Your sole job: given a unit type, return what it counters and what counters it,
using get_unit_counters.
Always explain the rock-paper-scissors logic briefly.
"""

_UNIT_STATS_PROMPT = """\
You are a Unit Stats Advisor for Age of Empires II.
Your sole job: look up HP, attack, armour, and cost for specific units
using get_unit_stats, and compare cost-effectiveness when asked.
Return numbers exactly as in the KB — never guess stats.
"""

_BUILDING_PROMPT = """\
You are a Building Advisor for Age of Empires II.
Your sole job: answer questions about building costs, HP, and function
using get_building_info.
Focus on actionable advice (e.g. "costs 175 wood, build near your lumber camp").
"""

_MATCHUP_PROMPT = """\
You are a Matchup Analyst for Age of Empires II specializing in the
{player_civ} vs {opponent_civ} matchup.

Your sole job: compare both civilizations head-to-head using get_civilization_info.
When asked, call get_civilization_info for BOTH civilizations and then provide:

1. Player ({player_civ}) key strengths in this matchup
2. Opponent ({opponent_civ}) key threats the player must prepare for
3. What the opponent's unique units/bonuses mean for the player's strategy
4. The single most important exploit or counter the player should focus on

Be concrete and civ-specific — never give generic advice.
Facts must come from get_civilization_info, not from memory.
"""

_SUPERVISOR_PROMPT = """\
You are the Strategist for an Age of Empires II: Definitive Edition coaching assistant.
Player civilization: {player_civ}
Opponent civilization: {opponent_civ}

You have six specialist sub-agents available as tools:
- consult_strategic_reference — CALL THIS FIRST every cycle: counter matrix, build orders, strategic principles, age costs
- consult_matchup_analyst     — {player_civ} vs {opponent_civ} strengths, weaknesses, and threats
- consult_civ_analyst         — look up any civilization's bonuses and unique units
- consult_counter_analyst     — deep unit counter detail
- consult_unit_stats          — unit HP, attack, armour, costs
- consult_building_advisor    — building costs and functions

Your job:
1. Read the GameStateSnapshot provided in the user message.
2. Call consult_strategic_reference FIRST to orient yourself on build orders,
   principles, and the counter matrix. Then call consult_matchup_analyst to
   understand the civ dynamics. Use other tools for specific facts as needed.
   Every factual claim MUST come from a sub-agent tool — do not invent stats.
3. After tool use, output structured JSON (rules below).

Advise the human player only.
"""


# ---------------------------------------------------------------------------
# Sub-agent builders
# ---------------------------------------------------------------------------

def _build_strategic_reference_advisor():
    return create_deep_agent(
        model=_resolve_strategy_model(),
        tools=[get_summary_info],
        system_prompt=_STRATEGIC_REF_PROMPT,
    )


def _build_civ_analyst():
    return create_deep_agent(
        model=_resolve_strategy_model(),
        tools=[get_civilization_info],
        system_prompt=_CIV_PROMPT,
        # middleware=[ScaledownSummarizationMiddleware(
        #     model=_resolve_strategy_model(),
        #     trigger=("fraction", 0.9),
        #     keep=("fraction", 0.7),
        # )]
    )


def _build_counter_analyst():
    return create_deep_agent(
        model=_resolve_strategy_model(),
        tools=[get_unit_counters],
        system_prompt=_COUNTER_PROMPT,
        # middleware=[ScaledownSummarizationMiddleware(
        #     model=_resolve_strategy_model(),
        #     trigger=("fraction", 0.8),
        #     keep=("fraction", 0.5),
        # )]
    )


def _build_unit_stats_advisor():
    return create_deep_agent(
        model=_resolve_strategy_model(),
        tools=[get_unit_stats],
        system_prompt=_UNIT_STATS_PROMPT,
    )


def _build_building_advisor():
    return create_deep_agent(
        model=_resolve_strategy_model(),
        tools=[get_building_info],
        system_prompt=_BUILDING_PROMPT,
    )


def _build_matchup_analyst(player_civ: str, opponent_civ: str):
    """Build the matchup analyst pre-seeded with both civilization names."""
    return create_deep_agent(
        model=_resolve_strategy_model(),
        tools=[get_civilization_info],
        system_prompt=_MATCHUP_PROMPT.format(
            player_civ=player_civ,
            opponent_civ=opponent_civ,
        ),
    )

def build_basic_strategy():
    return create_deep_agent(
        model=_resolve_strategy_model(),
        tools = []
    )

# ---------------------------------------------------------------------------
# Wrap sub-agents as @tool (pattern from 05_subagents_as_tool.ipynb)
# ---------------------------------------------------------------------------

def _extract_last_content(result: dict) -> str:
    """Pull the final assistant message out of a deepagents invoke result."""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", None) or (msg.get("content", "") if isinstance(msg, dict) else "")
        if content:
            return str(content)
    return ""


def build_kb_subagent_tools(player_civ: str, opponent_civ: str):
    """Build the five sub-agents and return them wrapped as LangChain tools.

    Args:
        player_civ: The player's civilization (used to seed the MatchupAnalyst).
        opponent_civ: The opponent's civilization (used to seed the MatchupAnalyst).

    Called once per session so each sub-agent is constructed fresh and the
    closures capture their own agent instance.
    """
    strategic_ref_agent = _build_strategic_reference_advisor()
    civ_agent           = _build_civ_analyst()
    counter_agent       = _build_counter_analyst()
    unit_agent          = _build_unit_stats_advisor()
    building_agent      = _build_building_advisor()
    matchup_agent       = _build_matchup_analyst(player_civ, opponent_civ)

    @tool
    def consult_strategic_reference(query: str) -> str:
        """Consult the Strategic Reference Advisor for the AoE2 master cheat-sheet.

        Call this FIRST every strategy cycle. Returns the complete strategic reference
        containing the unit counter matrix, six key strategic principles, three common
        build orders with exact villager assignments, and age advancement costs.

        Use for:
        - Checking which unit type counters the enemy's army composition
        - Verifying whether the player's build order is on schedule
        - Confirming resources needed before advising an age-up
        - Grounding any general strategic advice in universal principles

        Input: any question about general strategy, counters, build orders, or age costs.
        Example: "What counters cavalry archers?", "Is the player on track for Fast Castle?",
                 "How much does Castle Age cost?"
        """
        result = strategic_ref_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return _extract_last_content(result)

    @tool
    def consult_civ_analyst(query: str) -> str:
        """Consult the Civilization Analyst sub-agent for AoE2 civ bonuses and unique content.

        Use when you need to know a civilization's bonuses, unique units, or unique
        technologies.  Input should be a natural-language question, e.g.
        "What are the Mongols' bonuses?" or "Does Britons have a mounted unit bonus?"
        """
        result = civ_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return _extract_last_content(result)

    @tool
    def consult_counter_analyst(query: str) -> str:
        """Consult the Counter Analyst sub-agent for unit counter relationships.

        Use when you need to know what beats a unit type or what a unit type is
        weak against.  Input examples: "What counters knights?",
        "What is cavalry strong against?"
        """
        result = counter_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return _extract_last_content(result)

    @tool
    def consult_unit_stats(query: str) -> str:
        """Consult the Unit Stats Advisor sub-agent for AoE2 unit HP, attack, armour, and costs.

        Use when you need precise numbers for a unit, e.g. "How much does a Hussar
        cost?", "What is the HP of a Trebuchet?", "Compare Archer vs Crossbowman cost."
        """
        result = unit_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return _extract_last_content(result)

    @tool
    def consult_building_advisor(query: str) -> str:
        """Consult the Building Advisor sub-agent for building costs, HP, and function.

        Use when you need to know what a building costs or does, e.g.
        "How much wood does an Archery Range cost?", "What buildings produce cavalry?"
        """
        result = building_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return _extract_last_content(result)

    @tool
    def consult_matchup_analyst(query: str) -> str:
        """Consult the Matchup Analyst sub-agent for player vs opponent civilization dynamics.

        Use this FIRST before generating any strategy. It compares the player's
        civilization strengths against the opponent's threats and returns actionable
        matchup advice.  Input examples:
        "How does my civ compare to the opponent?",
        "What should I watch for from the opponent's unique units?",
        "What is my biggest advantage in this matchup?"
        """
        result = matchup_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return _extract_last_content(result)

    return [
        consult_strategic_reference,   # always call first — orientation tool
        consult_matchup_analyst,       # always call second — civ dynamics
        consult_civ_analyst,
        consult_counter_analyst,
        consult_unit_stats,
        consult_building_advisor,
    ]


# ---------------------------------------------------------------------------
# Supervisor / Strategist
# ---------------------------------------------------------------------------

_STRATEGY_JSON_RULES = """\
4. Your FINAL message is a list of short, concrete instructions for the player, grouped under
   Economy, Military, Tech & buildings and Scouting. Name exact units, buildings, technologies,
   villager counts and costs from the tools. No generic advice, no description of your process.
"""

_FLAT_SUPERVISOR_PROMPT = """\
You are the Strategist for an Age of Empires II: Definitive Edition coaching assistant.
Player civilization: {player_civ}
Opponent civilization: {opponent_civ}

You have direct knowledge-base tools:
- get_summary_info        — counter matrix, build orders, strategic principles, age costs (call FIRST)
- get_civilization_info   — a civilization's bonuses and unique units (use for both civs to assess the matchup)
- get_unit_counters       — what a unit type beats and is beaten by
- get_unit_stats          — a unit's HP, attack, armour, cost
- get_building_info       — a building's cost, HP, function

Your job:
1. Read the GameStateSnapshot in the user message.
2. Call get_summary_info first for orientation, then look up only the specific
   facts you need (the player's civ, the opponent's civ, relevant counters).
   Keep tool calls minimal — you do NOT need to call every tool every time.
   Every factual claim MUST come from a tool — do not invent stats.
3. After tool use, write the final instructions (rules below).

Advise the human player only.
"""


def build_strategist(
    player_civ: str,
    opponent_civ: str = "Unknown",
    use_subagents: bool = False,
    model_service=None,
):
    """Build the Supervisor Strategist agent.

    Args:
        player_civ: Civilization the player selected at session start.
        opponent_civ: Civilization the opponent is playing (default 'Unknown').
        use_subagents: When True, use the nested subagents-as-tools architecture
            (6 specialist sub-agents). This is thorough but SLOW — each sub-agent
            is a full deep-agent doing its own model round-trips, so one strategy
            cycle can take many minutes on local Ollama. When False (default), the
            supervisor calls the KB functions directly (flat tools), collapsing
            ~7 model loops into 1 and producing a strategy update much faster.

    Returns:
        A compiled deepagents agent ready to invoke.
    """
    agents_md = AGENTS_MEMORY_FILE

    if use_subagents:
        tools = build_kb_subagent_tools(player_civ, opponent_civ)
        system_prompt = (
            _SUPERVISOR_PROMPT.format(player_civ=player_civ, opponent_civ=opponent_civ)
            + "\n"
            + _STRATEGY_JSON_RULES
        )
    else:
        # Flat, fast path — KB functions exposed directly to the supervisor
        tools = [
            get_summary_info,
            get_civilization_info,
            get_unit_counters,
            get_unit_stats,
            get_building_info,
        ]
        system_prompt = (
            _FLAT_SUPERVISOR_PROMPT.format(player_civ=player_civ, opponent_civ=opponent_civ)
            + "\n"
            + _STRATEGY_JSON_RULES
        )

    return create_deep_agent(
        model=_resolve_strategy_model(),
        tools=tools,
        system_prompt=system_prompt,
        memory=[str(agents_md)] if agents_md.exists() else [],
    )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _snapshot_to_prompt(
    snap: GameStateSnapshot,
    history: Deque[GameStateSnapshot],
    player_civ: str,
    opponent_civ: str = "Unknown",
) -> str:
    """Serialize the current snapshot + recent trend + matchup context into a user message."""
    resources = snap.resources
    units_summary = ", ".join(
        f"{u.approx_count}x {u.label} ({u.owner})" for u in snap.visible_units
    ) or "none visible"
    buildings_summary = ", ".join(snap.visible_buildings) or "none visible"

    trend_note = ""
    if len(history) >= 3:
        old = history[-3]
        old_food = old.resources.get("food")
        new_food = resources.get("food")
        if old_food is not None and new_food is not None:
            delta = new_food - old_food
            trend_note = f"\nFood trend (last ~{3 * STRATEGIST_INTERVAL_SECS}s): {delta:+d}"

    return f"""\
Matchup: {player_civ} (you) vs {opponent_civ} (opponent)

Age: {snap.age}
Population: {snap.population}/{snap.pop_cap}
Resources: food={resources.get('food')}, wood={resources.get('wood')}, \
gold={resources.get('gold')}, stone={resources.get('stone')}{trend_note}
Idle villagers: {snap.idle_villagers_visible}
Threat level: {snap.threat_level}
Visible units: {units_summary}
Visible buildings: {buildings_summary}
Notes: {snap.notes or 'none'}

Consult the matchup analyst first, then provide your strategic advice for this situation.
"""


# ---------------------------------------------------------------------------
# Async loop
# ---------------------------------------------------------------------------

class StrategistLoop:
    """Async Strategist Loop.

    Args:
        player_civ: Civilization chosen by the player.
        opponent_civ: Civilization the opponent is playing.
        on_strategy: Callback invoked with strategy text after each analysis.
        interval: Seconds between strategist invocations.
        perception_loop: Reference to the PerceptionLoop for snapshot access.
        gate: When given, each analysis waits until this event is set. The web
            session uses it to hold the strategist back while Nimble is
            answering the latest snapshot.
    """

    def __init__(
        self,
        player_civ: str,
        on_strategy: Callable[[str, Optional[dict]], None],
        opponent_civ: str = "Unknown",
        interval: float = STRATEGIST_INTERVAL_SECS,
        perception_loop=None,
        use_subagents: bool = True,
        on_status: Callable[[str], None] | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self._gate = gate
        self._player_civ = player_civ
        self._opponent_civ = opponent_civ
        self._on_strategy = on_strategy
        self._on_status = on_status
        self._interval = interval
        self._perception = perception_loop
        self._agent = build_strategist(player_civ, opponent_civ, use_subagents=use_subagents)
        self._wake = asyncio.Event()

    def request_analysis(self) -> None:
        """Start the next analysis now instead of waiting out the interval.

        Must be called from the thread running this loop's event loop.
        """
        self._wake.set()

    async def _wait_for_first_snapshot(self, stop_event: asyncio.Event | None) -> bool:
        """Poll briefly until perception produces its first snapshot.

        Returns True once a snapshot is available, False if stopped first.
        """
        while stop_event is None or not stop_event.is_set():
            if self._perception is not None and self._perception.latest_snapshot is not None:
                return True
            await asyncio.sleep(1.0)
        return False

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Run until stop_event is set (or forever if None)."""
        log.info(
            "Strategist loop starting — %s vs %s (interval=%.1fs)",
            self._player_civ, self._opponent_civ, self._interval,
        )
        _agent_logger.info("Strategist loop starting — %s vs %s (interval=%.1fs)",
                           self._player_civ, self._opponent_civ, self._interval)

        # Run the FIRST analysis as soon as perception has a snapshot, rather than
        # blocking for a full interval first (which delayed the first update badly).
        if not await self._wait_for_first_snapshot(stop_event):
            return

        first = True
        while stop_event is None or not stop_event.is_set():
            if not first:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    pass
            self._wake.clear()
            first = False
            if self._gate is not None:
                await self._gate.wait()

            snap: GameStateSnapshot | None = None
            history: Deque[GameStateSnapshot] = deque()

            if self._perception is not None:
                snap = self._perception.latest_snapshot
                history = self._perception.snapshot_history

            if snap is None:
                log.debug("Strategist waiting for first snapshot…")
                continue

            # Feedback that we are thinking — routed to on_status so the real
            # strategy text stays "sticky" and is NOT overwritten by a transient.
            if self._on_status is not None:
                self._on_status(f"Analyzing — {snap.age} Age…")

            prompt = _snapshot_to_prompt(snap, history, self._player_civ, self._opponent_civ)

            try:
                import time as _time
                t0 = _time.perf_counter()
                _agent_logger.info("Strategist invoke started (%s Age)…", snap.age)
                result = await asyncio.to_thread(
                    self._agent.invoke,
                    {"messages": [{"role": "user", "content": prompt}]},
                )
                elapsed = _time.perf_counter() - t0
                raw = _extract_last_content(result)
                if raw:
                    update = parse_strategy_response(raw)
                    if update is None:
                        update = await asyncio.to_thread(format_strategy, prompt, raw, snap)
                    if update is None:
                        update = summarize_raw_strategy(raw, snap.age)
                        log.warning("Strategy formatter unavailable — used keyword fallback")
                    elapsed = _time.perf_counter() - t0
                    strategy_text = strategy_to_markdown(update)
                    structured = update.model_dump()
                    _agent_logger.info("Strategist invoke finished in %.1fs", elapsed)
                    _log_strategy(self._player_civ, self._opponent_civ, strategy_text)
                    self._on_strategy(strategy_text, structured)
                    if self._on_status is not None:
                        self._on_status("idle")
                    log.debug("Strategist update issued (%.1fs)", elapsed)
                else:
                    _agent_logger.warning("Strategist returned empty response for %s vs %s (%.1fs)",
                                          self._player_civ, self._opponent_civ, elapsed)
                    log.warning("Strategist returned empty response")
            except Exception as exc:
                _agent_logger.error("Strategist loop error: %s", exc, exc_info=True)
                log.error("Strategist loop error: %s", exc)
            # Note: the inter-cycle sleep happens at the TOP of the loop so the
            # first analysis runs immediately after the first snapshot arrives.
