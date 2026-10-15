"""Offline tests for the Nimble fast coach (no Ollama required).

Run from src/:
    python -m pytest aoe2_agent/tests/test_nimble.py
    python aoe2_agent/tests/test_nimble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aoe2_agent.schemas import DetectedUnit, GameStateSnapshot
from aoe2_agent.strategy.nimble import (
    DecisionResponse,
    NimbleAdvisor,
    build_question_plan,
    items_for_age,
)
from aoe2_agent.strategy.nimble.questions import COUNTER, QuestionPlan


class FakeClient:
    """Answers every action question with a fixed verdict, overridable per key."""

    def __init__(self, overrides: dict[str, dict] | None = None, default_verdict: str = "skip") -> None:
        self.overrides = overrides or {}
        self.default_verdict = default_verdict
        self.calls: list[dict] = []

    def decide(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        answers = {}
        for key, q in questions.items():
            if key in self.overrides:
                answers[key] = self.overrides[key]
            elif q["type"] == "choice":
                first = self.default_verdict if self.default_verdict in q["criteria"] else next(iter(q["criteria"]))
                answers[key] = {"type": "choice", "choice": first, "probabilities": {first: 0.9}}
            elif q["type"] == "score":
                answers[key] = {"type": "score", "score": 0.0}
        return DecisionResponse(answers=answers, latency_ms=12.0, model="fake")


def _choice(choice: str, p: float = 0.8) -> dict:
    return {"type": "choice", "choice": choice, "probabilities": {choice: p}}


def _dark_opening(**overrides) -> GameStateSnapshot:
    fields = dict(
        age="Dark",
        population=6,
        pop_cap=10,
        resources={"food": 400, "wood": 250, "gold": 0, "stone": 0},
        visible_units=[DetectedUnit(label="villager", owner="self", approx_count=6)],
        visible_buildings=["Town Center", "House"],
        idle_villagers_visible=False,
        threat_level="none",
        notes="Game just started",
    )
    fields.update(overrides)
    return GameStateSnapshot(**fields)


def test_plan_asks_one_question_per_age_relevant_item():
    snap = _dark_opening()
    plan = build_question_plan(snap, max_questions=64)
    for item in items_for_age("Dark"):
        assert QuestionPlan.item_key(item) in plan.questions
    assert "item_knight" not in plan.questions
    assert COUNTER not in plan.questions


def test_question_text_includes_affordability():
    plan = build_question_plan(_dark_opening(), max_questions=64)
    assert "short 50 gold" in plan.questions["item_loom"]["instructions"]
    assert "affordable now" in plan.questions["item_villager"]["instructions"]


def test_counter_question_only_when_enemy_visible():
    snap = _dark_opening(
        age="Feudal",
        threat_level="low",
        visible_units=[DetectedUnit(label="scout cavalry", owner="enemy", approx_count=1)],
    )
    plan = build_question_plan(snap, max_questions=64)
    assert set(plan.questions[COUNTER]["criteria"]) >= {"spearman", "skirmisher", "none"}


def test_question_budget_is_respected():
    plan = build_question_plan(_dark_opening(age="Castle"), max_questions=10)
    assert len(plan.questions) == 10


def test_ranked_recommendations_drive_priority_and_actions():
    client = FakeClient(overrides={
        "item_villager": {"type": "choice", "choice": "now", "probabilities": {"now": 0.9, "soon": 0.1}},
        "item_mill": {"type": "choice", "choice": "now", "probabilities": {"now": 0.6, "soon": 0.3}},
        "item_feudal_age": {"type": "choice", "choice": "soon", "probabilities": {"now": 0.2, "soon": 0.7}},
        "gather": _choice("food"),
    })
    snap = _dark_opening(population=6, pop_cap=15)
    advice = NimbleAdvisor("Mongols", "Vietnamese", client=client).advise(snap)

    assert advice.priority.startswith("Train Villager at Town Center")
    assert [r.item.id for r in advice.by_verdict("now")] == ["villager", "mill"]
    assert advice.actions[0].startswith("Build Mill")
    assert any(a.startswith("Soon: Advance to Feudal Age") for a in advice.actions)
    assert advice.actions[-1] == "Send new villagers to food"
    assert "**Priority:**" in advice.to_markdown()


def test_deterministic_rules_override_model_ranking():
    client = FakeClient(default_verdict="now")
    advisor = NimbleAdvisor("Mongols", "Vietnamese", client=client)

    assert "idle villagers" in advisor.advise(_dark_opening(idle_villagers_visible=True)).priority
    assert advisor.advise(_dark_opening(population=8, pop_cap=10)).priority.startswith("Build a House")


def test_high_threat_uses_counter_answer():
    client = FakeClient(overrides={COUNTER: _choice("spearman")})
    snap = _dark_opening(
        age="Feudal",
        population=20,
        pop_cap=35,
        threat_level="high",
        visible_units=[DetectedUnit(label="scout cavalry", owner="enemy", approx_count=4)],
    )
    advice = NimbleAdvisor("Mongols", "Vietnamese", client=client).advise(snap)
    assert advice.priority.startswith("Defend now: train Spearman at Barracks")


def test_escalation_needs_confident_answer():
    weak = FakeClient(overrides={"escalate": _choice("matchup_detail", p=0.3)})
    strong = FakeClient(overrides={"escalate": _choice("composition_plan", p=0.7)})
    assert not NimbleAdvisor("Mongols", client=weak).advise(_dark_opening()).escalate
    assert NimbleAdvisor("Mongols", client=strong).advise(_dark_opening()).escalate


def test_to_dict_is_json_serializable():
    import json

    advice = NimbleAdvisor("Mongols", "Vietnamese", client=FakeClient()).advise(_dark_opening())
    payload = json.loads(json.dumps(advice.to_dict()))
    assert payload["age"] == "Dark"
    assert payload["recommendations"]


if __name__ == "__main__":
    tests = [obj for name, obj in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"All {len(tests)} Nimble tests passed.")
