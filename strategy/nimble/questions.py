"""Nimble question plan for one coaching cycle.

Two groups of questions are asked against the same state:

* Situation questions — overall focus, which resource to gather, military
  urgency, which unit counters the visible enemy, and whether the slow KB
  strategist should run.
* Action questions — one ``choice`` per catalog item relevant to the current
  age ("train Spearman?", "build Mill?", "research Loom?", "advance to Feudal?")
  answered with ``now`` / ``soon`` / ``skip``. Ranking these answers gives
  the build/train/research recommendations.

Facts the code already knows (idle villagers, pop headroom, affordability)
are computed here and written into the question text instead of being asked,
so Nimble only makes judgement calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schemas import GameStateSnapshot

from .catalog import CatalogItem, items_for_age, military_units_for_age
from .state import has_enemy_presence

Question = dict[str, Any]

FOCUS = "focus"
GATHER = "gather"
MILITARY_URGENCY = "military_urgency"
COUNTER = "counter"
ESCALATE = "escalate"

NO_COUNTER = "none"
ITEM_PREFIX = "item_"

VERDICTS = ("now", "soon", "skip")
VERDICT_CRITERIA = {
    "now": "Do it this minute: it addresses a current need and the player can afford it",
    "soon": "Worth doing within the next 1-2 minutes, but something else matters more right now",
    "skip": "Not needed in this situation, unaffordable for a while, or its prerequisite is missing",
}

FOCUS_CRITERIA = {
    "economy": "Grow the economy: villagers, houses, drop sites, eco upgrades",
    "advance_age": "Save resources and click the next age",
    "defend": "Enemy pressure: walls, towers, garrison or defensive units",
    "attack": "Safe and ahead: take a fight or raid with the current army",
    "scout": "Opponent's position or army is unknown and should be found first",
}

GATHER_CRITERIA = {
    "food": "Villagers, age-ups and most upgrades need food",
    "wood": "Houses, farms, drop sites and military buildings need wood",
    "gold": "Castle/Imperial age-ups, Knights, archers and Monks need gold",
    "stone": "Castles, Town Centers and towers need stone",
    "balanced": "No single resource is the bottleneck",
}

MILITARY_URGENCY_LEVELS = [
    "None: no enemy army seen, keep investing in economy",
    "Light: a few defensive units would be prudent soon",
    "High: enemy army is visible or threat is rising, produce units now",
    "Emergency: base is under attack right now",
]

ESCALATE_CRITERIA = {
    "fast_enough": "Routine economy, housing or age-up decision",
    "matchup_detail": "Civilization bonuses or opponent unique units change the plan",
    "composition_plan": "An exact army composition or counter plan from the knowledge base is needed",
}


@dataclass(frozen=True)
class QuestionPlan:
    questions: dict[str, Question]
    items: tuple[CatalogItem, ...]
    counter_options: tuple[CatalogItem, ...]

    @staticmethod
    def item_key(item: CatalogItem) -> str:
        return f"{ITEM_PREFIX}{item.id}"


def _item_question(item: CatalogItem, snap: GameStateSnapshot) -> Question:
    details = "; ".join(
        part for part in (item.prereq, item.cost_text(), item.affordability(snap.resources)) if part
    )
    return {
        "type": "choice",
        "instructions": f"Should the player {item.verb} {item.name} ({details})? Purpose: {item.role}.",
        "criteria": VERDICT_CRITERIA,
    }


def _counter_question(options: list[CatalogItem]) -> Question:
    criteria: dict[str, str] = {item.id: f"{item.name}: {item.role}" for item in options}
    criteria[NO_COUNTER] = "No enemy army is visible, so no counter unit is needed"
    return {
        "type": "choice",
        "instructions": "Which unit available in this age best counters the visible enemy units?",
        "criteria": criteria,
    }


def _situation_questions(snap: GameStateSnapshot) -> tuple[dict[str, Question], list[CatalogItem]]:
    questions: dict[str, Question] = {
        FOCUS: {
            "type": "choice",
            "instructions": "What should the player focus on for the next minute?",
            "criteria": FOCUS_CRITERIA,
        },
        GATHER: {
            "type": "choice",
            "instructions": "Which resource should newly trained or idle villagers gather?",
            "criteria": GATHER_CRITERIA,
        },
        MILITARY_URGENCY: {
            "type": "score",
            "instructions": "How urgently does the player need military units?",
            "criteria": MILITARY_URGENCY_LEVELS,
        },
        ESCALATE: {
            "type": "choice",
            "instructions": "Is a fast checklist answer enough, or does this situation need deep matchup analysis?",
            "criteria": ESCALATE_CRITERIA,
        },
    }
    counter_options: list[CatalogItem] = []
    if has_enemy_presence(snap):
        counter_options = military_units_for_age(snap.age)
        if counter_options:
            questions[COUNTER] = _counter_question(counter_options)
    return questions, counter_options


def build_question_plan(snap: GameStateSnapshot, max_questions: int) -> QuestionPlan:
    questions, counter_options = _situation_questions(snap)
    budget = max(0, max_questions - len(questions))
    items = items_for_age(snap.age)[:budget]
    for item in items:
        questions[QuestionPlan.item_key(item)] = _item_question(item, snap)
    return QuestionPlan(questions=questions, items=tuple(items), counter_options=tuple(counter_options))
