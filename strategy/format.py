"""Turn strategist research into a specific, bullet-point StrategyUpdate."""
from __future__ import annotations

import json
import logging
import os
import re

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_STRATEGY_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_STRATEGY_MODEL,
)
from schemas import (
    STRATEGY_MAX_BULLET_WORDS,
    STRATEGY_MAX_BULLETS,
    STRATEGY_MAX_WORDS,
    STRATEGY_SECTIONS,
    GameStateSnapshot,
    StrategyUpdate,
)

log = logging.getLogger(__name__)

_SECTION_LABELS = {
    "economy": "Economy",
    "military": "Military",
    "tech_and_buildings": "Tech & buildings",
    "scouting": "Scouting",
}

_FORMATTER_PROMPT = f"""\
You turn Age of Empires II coaching notes into a short, specific action plan for the player.

Fields:
- priority: a full sentence with the single most urgent action. Check the game state first:
  population within 5 of pop cap -> build houses; idle villagers -> put them to work;
  food under 200 -> more on food; enemy threat high -> defend.
- economy: villagers only — how many to train, which resource to move them to
  (food = sheep/berries/farms near the Mill, wood = Lumber Camp, gold/stone = Mining Camp).
- military: which units to train, how many, and what they counter.
- tech_and_buildings: buildings to place and technologies/ages to research. No villager orders here.
- scouting: what the scout should look for.
- matchup_note: one line on how the two civilizations shape the plan.

Example of the style only (a different game — do not copy its content):
{{"age": "Dark",
 "priority": "Queue 3 villagers at the Town Center — it has been idle",
 "economy": ["Put the next 4 villagers on sheep under the Town Center"],
 "military": [],
 "tech_and_buildings": ["Research Loom before the first enemy scout arrives"],
 "scouting": ["Find the second gold mine near your base"],
 "matchup_note": "Aztecs rush early; keep villagers close to the Town Center"}}

Rules:
- Base every bullet on THIS game's state. Only recommend units the player's civilization can train.
- Keep each fact attached to what it describes in the notes; never move a cost onto a different item.
- Every bullet is ONE concrete instruction that starts with a verb and names the exact unit,
  building, technology or resource, with numbers where possible
  (e.g. "Put 6 villagers on sheep, 4 on wood, 2 on berries", "Train 5 Spearmen at the Barracks").
- Never write generic advice like "build a strong economy" or "focus on military".
- Never mention tools, analysts, or how the notes were produced.
- Use only facts present in the game state and the notes. Do not invent costs or stats.
- 1-{STRATEGY_MAX_BULLETS} bullets per section, each at most {STRATEGY_MAX_BULLET_WORDS} words.
  Use an empty list for a section with nothing useful.
- Whole answer at most {STRATEGY_MAX_WORDS} words.
"""

_formatter_llm = None


def _get_formatter():
    """Schema-constrained chat model; Ollama's `format` guarantees valid JSON."""
    global _formatter_llm
    if _formatter_llm is not None:
        return _formatter_llm
    key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        from langchain_openrouter import ChatOpenRouter
        _formatter_llm = ChatOpenRouter(
            model=OPENROUTER_STRATEGY_MODEL, temperature=0, max_tokens=1024,
        ).with_structured_output(StrategyUpdate)
    else:
        from langchain_ollama import ChatOllama
        _formatter_llm = ChatOllama(
            model=OLLAMA_STRATEGY_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            format=StrategyUpdate.model_json_schema(),
        )
    return _formatter_llm


def urgent_actions(snap: GameStateSnapshot) -> list[str]:
    """Snapshot-driven must-do actions, most urgent first (mirrors alarms/rules.py)."""
    out: list[str] = []
    food = snap.resources.get("food")
    if snap.threat_level == "high":
        out.append("Defend now — pull your army back to the units under attack")
    if snap.population is not None and snap.pop_cap is not None and snap.pop_cap - snap.population <= 5:
        out.append(f"Build 2 houses now — population is {snap.population}/{snap.pop_cap}")
    if snap.idle_villagers_visible:
        out.append("Put the idle villagers back to work")
    if food is not None and food < 200:
        out.append(f"Move villagers onto food — only {food} food left")
    return out


def format_strategy(game_state: str, notes: str, snap: GameStateSnapshot) -> StrategyUpdate | None:
    """Ask the model to rewrite free-form strategist notes as a StrategyUpdate."""
    urgent = urgent_actions(snap)
    must = ""
    if urgent:
        must = "\n\nMust address (most urgent first):\n" + "\n".join(f"- {u}" for u in urgent)
    messages = [
        ("system", _FORMATTER_PROMPT),
        ("user", f"Game state:\n{game_state}\n\nCoaching notes:\n{notes}{must}\n\nCurrent age: {snap.age}"),
    ]
    try:
        result = _get_formatter().invoke(messages)
    except Exception as exc:
        log.warning("Strategy formatter failed: %s", exc)
        return None
    update = result if isinstance(result, StrategyUpdate) else parse_strategy_response(
        getattr(result, "content", "") or ""
    )
    if update is not None and urgent:
        update = StrategyUpdate.model_validate({**update.model_dump(), "priority": urgent[0]})
    return update


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()


def parse_strategy_response(raw: str) -> StrategyUpdate | None:
    """Parse model output into StrategyUpdate; None if not valid JSON."""
    if not raw or not raw.strip():
        return None
    text = _strip_fences(raw)
    if not text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
    try:
        return StrategyUpdate.model_validate(json.loads(text))
    except Exception:
        return None


def strategy_to_markdown(update: StrategyUpdate) -> str:
    """Bullet-point markdown for the CLI overlay and logs."""
    lines = [f"## Strategist — {update.age} Age", "", f"**Priority:** {update.priority}", ""]
    for key, label in _SECTION_LABELS.items():
        bullets = getattr(update, key)
        if bullets:
            lines.append(f"**{label}**")
            lines.extend(f"- {b}" for b in bullets)
            lines.append("")
    if update.matchup_note:
        lines.append(f"**Matchup:** {update.matchup_note}")
    return "\n".join(lines).strip()


_NOISE = ("consult", "tool", "analyst", "strategic reference", "assistant:", "i will", "i'll", "let's")
_KEYWORDS = {
    "economy": ("villager", "economy", "food", "wood", "gold", "stone", "sheep", "berries", "farm"),
    "military": ("army", "military", "defend", "counter", "archer", "spear", "pike", "knight",
                 "skirmisher", "militia", "cavalry", "infantry", "unit"),
    "tech_and_buildings": ("build", "research", "blacksmith", "barracks", "range", "stable",
                           "mill", "camp", "house", "advance", "age"),
    "scouting": ("scout", "vision", "map", "enemy base"),
}


def summarize_raw_strategy(raw: str, age: str) -> StrategyUpdate:
    """Last-resort fallback: sort free-form lines into sections without repeating any line."""
    lines: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).replace("**", "").strip(" .")
        if len(line.split()) >= 4 and not any(n in line.lower() for n in _NOISE) and line not in lines:
            lines.append(line)

    sections: dict[str, list[str]] = {s: [] for s in STRATEGY_SECTIONS}
    for line in lines[1:]:
        low = line.lower()
        for section in STRATEGY_SECTIONS:
            if len(sections[section]) < STRATEGY_MAX_BULLETS and any(k in low for k in _KEYWORDS[section]):
                sections[section].append(line)
                break

    return StrategyUpdate(
        age=age if age in ("Dark", "Feudal", "Castle", "Imperial") else "Dark",
        priority=lines[0] if lines else "Keep the Town Center producing villagers",
        **sections,
    )
