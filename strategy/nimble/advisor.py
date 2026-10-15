"""Fast coaching: snapshot -> Nimble answers -> ranked, player-facing advice."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from time_display import now_hms
from typing import Any, Literal

from config import ALARM_FOOD_FLOOR, ALARM_POP_HEADROOM, NIMBLE_MAX_QUESTIONS
from schemas import GameStateSnapshot

from .catalog import CATALOG_BY_ID, CatalogItem
from .client import DecisionClient, DecisionResponse, NimbleClient
from .questions import (
    COUNTER,
    ESCALATE,
    FOCUS,
    FOCUS_CRITERIA,
    GATHER,
    MILITARY_URGENCY,
    MILITARY_URGENCY_LEVELS,
    NO_COUNTER,
    VERDICTS,
    QuestionPlan,
    build_question_plan,
)
from .state import build_state, pop_headroom

Verdict = Literal["now", "soon", "skip"]

ESCALATE_MIN_PROBABILITY = 0.5
MIN_PICK_PROBABILITY = 0.4
MAX_ACTIONS = 5
MAX_POPULATION = 200


@dataclass(frozen=True)
class Pick:
    """The winning option of a ``choice`` question and its probability distribution."""

    choice: str | None
    probabilities: dict[str, float]

    @property
    def probability(self) -> float:
        return self.probabilities.get(self.choice or "", 0.0)

    @classmethod
    def from_answer(cls, answer: dict[str, Any] | None) -> Pick:
        answer = answer or {}
        probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}
        return cls(choice=answer.get("choice"), probabilities=probs)


@dataclass(frozen=True)
class Recommendation:
    item: CatalogItem
    verdict: Verdict
    probabilities: dict[str, float]
    affordability: str

    @property
    def p_now(self) -> float:
        return self.probabilities.get("now", 0.0)

    @property
    def p_soon(self) -> float:
        return self.probabilities.get("soon", 0.0)

    def action_text(self) -> str:
        where = f" {self.item.prereq}" if self.item.kind in ("unit", "tech") else ""
        return f"{self.item.verb.capitalize()} {self.item.name}{where} — {self.item.cost_text()} ({self.affordability})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item.id,
            "name": self.item.name,
            "kind": self.item.kind,
            "verb": self.item.verb,
            "prereq": self.item.prereq,
            "cost": self.item.cost_text(),
            "affordability": self.affordability,
            "verdict": self.verdict,
            "p_now": round(self.p_now, 3),
            "p_soon": round(self.p_soon, 3),
            "text": self.action_text(),
        }


@dataclass
class NimbleAdvice:
    age: str
    player_civ: str
    opponent_civ: str
    priority: str
    actions: list[str]
    rationale: str
    recommendations: list[Recommendation]
    focus: Pick
    gather: Pick
    counter: CatalogItem | None
    military_urgency: int | None
    escalate: bool
    latency_ms: float
    model: str
    question_count: int
    answers: dict[str, dict[str, Any]] = field(repr=False, default_factory=dict)
    generated_at: str = field(default_factory=now_hms)

    def by_verdict(self, verdict: Verdict) -> list[Recommendation]:
        return [r for r in self.recommendations if r.verdict == verdict]

    def to_markdown(self) -> str:
        """Player-facing update in the Strategist output format from AGENTS.md."""
        numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(self.actions, start=1))
        return (
            f"## Strategist Update — {self.age} Age (fast coach)\n\n"
            f"**Priority:** {self.priority}\n\n"
            f"**Next actions:**\n{numbered}\n\n"
            f"**Rationale:** {self.rationale}\n"
        )

    def to_report(self) -> str:
        """Debug view of every Nimble answer, for notebooks."""
        urgency = (
            MILITARY_URGENCY_LEVELS[self.military_urgency] if self.military_urgency is not None else "unknown"
        )
        lines = [
            f"### Nimble — {self.player_civ} vs {self.opponent_civ}, {self.age} Age",
            f"`{self.model}` · {self.question_count} questions · {self.latency_ms:,.0f} ms",
            "",
            "| Situation | Answer | p |",
            "|---|---|---|",
            f"| Focus | `{self.focus.choice}` | {self.focus.probability:.0%} |",
            f"| Gather | `{self.gather.choice}` | {self.gather.probability:.0%} |",
            f"| Military urgency | {urgency} | |",
            f"| Counter unit | {self.counter.name if self.counter else '—'} | |",
            f"| Escalate to strategist | {'yes' if self.escalate else 'no'} | |",
            "",
            "| Action | Kind | Verdict | p(now) | p(soon) | Cost |",
            "|---|---|---|---|---|---|",
        ]
        for rec in self.recommendations:
            lines.append(
                f"| {rec.item.verb} {rec.item.name} | {rec.item.kind} | **{rec.verdict}** | "
                f"{rec.p_now:.0%} | {rec.p_soon:.0%} | {rec.item.cost_text()} ({rec.affordability}) |"
            )
        lines += ["", "---", "", self.to_markdown()]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "age": self.age,
            "priority": self.priority,
            "actions": self.actions,
            "rationale": self.rationale,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "focus": self.focus.choice,
            "gather": self.gather.choice,
            "counter": self.counter.name if self.counter else None,
            "military_urgency": self.military_urgency,
            "military_urgency_label": (
                MILITARY_URGENCY_LEVELS[self.military_urgency] if self.military_urgency is not None else None
            ),
            "escalate": self.escalate,
            "latency_ms": round(self.latency_ms),
            "model": self.model,
            "question_count": self.question_count,
            "generated_at": self.generated_at,
            "markdown": self.to_markdown(),
        }


class NimbleAdvisor:
    """Ask Nimble the per-age question plan and interpret the answers."""

    def __init__(
        self,
        player_civ: str,
        opponent_civ: str = "Unknown",
        client: DecisionClient | None = None,
        max_questions: int = NIMBLE_MAX_QUESTIONS,
    ) -> None:
        self.player_civ = player_civ
        self.opponent_civ = opponent_civ
        self._client = client or NimbleClient()
        self._max_questions = max_questions

    def advise(
        self, snap: GameStateSnapshot, history: Sequence[GameStateSnapshot] = ()
    ) -> NimbleAdvice:
        state = build_state(
            snap, player_civ=self.player_civ, opponent_civ=self.opponent_civ, history=history
        )
        plan = build_question_plan(snap, self._max_questions)
        response = self._client.decide(state, plan.questions)
        return interpret(snap, plan, response, self.player_civ, self.opponent_civ)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def _recommendations(snap: GameStateSnapshot, plan: QuestionPlan, answers: dict[str, Any]) -> list[Recommendation]:
    recs: list[Recommendation] = []
    for item in plan.items:
        pick = Pick.from_answer(answers.get(plan.item_key(item)))
        verdict: Verdict = pick.choice if pick.choice in VERDICTS else "skip"  # type: ignore[assignment]
        recs.append(Recommendation(item, verdict, pick.probabilities, item.affordability(snap.resources)))
    order = {v: i for i, v in enumerate(VERDICTS)}
    recs.sort(key=lambda r: (order[r.verdict], -r.p_now, -r.p_soon))
    return recs


def _score_level(answer: dict[str, Any] | None) -> int | None:
    score = (answer or {}).get("score")
    if score is None:
        return None
    return max(0, min(len(MILITARY_URGENCY_LEVELS) - 1, round(float(score))))


def _counter(answers: dict[str, Any]) -> CatalogItem | None:
    pick = Pick.from_answer(answers.get(COUNTER))
    if pick.choice in (None, NO_COUNTER):
        return None
    return CATALOG_BY_ID.get(pick.choice)


def _priority(
    snap: GameStateSnapshot, recs: list[Recommendation], focus: Pick, counter: CatalogItem | None
) -> tuple[str, Recommendation | None]:
    """Pick the single most important instruction; returns the recommendation it consumed."""
    food = snap.resources.get("food")
    if snap.idle_villagers_visible:
        target = "food" if food is not None and food < ALARM_FOOD_FLOOR else "the scarcest resource"
        return f"Put idle villagers back to work on {target}", None

    if snap.threat_level == "high":
        if counter is not None:
            return f"Defend now: train {counter.name} {counter.prereq} ({counter.cost_text()})", None
        return "Defend now: garrison villagers and pull your army home", None

    headroom = pop_headroom(snap)
    if headroom is not None and headroom <= ALARM_POP_HEADROOM and (snap.pop_cap or 0) < MAX_POPULATION:
        house = next((r for r in recs if r.item.id == "house"), None)
        return (
            f"Build a House now (25W, +5 pop): population is {snap.population}/{snap.pop_cap}",
            house,
        )

    top_now = next((r for r in recs if r.verdict == "now"), None)
    if top_now is not None:
        return top_now.action_text(), top_now

    if focus.choice and focus.probability >= MIN_PICK_PROBABILITY:
        return FOCUS_CRITERIA.get(focus.choice, focus.choice), None
    return "Keep the Town Center producing villagers", None


def _actions(recs: list[Recommendation], used: Recommendation | None, gather: Pick) -> list[str]:
    actions = [r.action_text() for r in recs if r.verdict == "now" and r is not used]
    actions += [f"Soon: {r.action_text()}" for r in recs if r.verdict == "soon" and r is not used]
    actions = actions[: MAX_ACTIONS - 1]
    if gather.choice and gather.probability >= MIN_PICK_PROBABILITY:
        actions.append(f"Send new villagers to {gather.choice}")
    if not actions:
        actions.append("Keep the Town Center producing villagers")
    return actions[:MAX_ACTIONS]


def _rationale(
    snap: GameStateSnapshot, response: DecisionResponse, recs: list[Recommendation],
    player_civ: str, opponent_civ: str, escalate: bool,
) -> str:
    r = snap.resources
    observed = (
        f"pop {snap.population}/{snap.pop_cap}, food {r.get('food')}, wood {r.get('wood')}, "
        f"gold {r.get('gold')}, stone {r.get('stone')}, threat {snap.threat_level}"
    )
    text = (
        f"Nimble ranked {len(recs)} {snap.age} Age options for {player_civ} vs {opponent_civ} "
        f"in {response.latency_ms / 1000:.1f}s from the snapshot ({observed})."
    )
    if escalate:
        text += " Deep strategist analysis recommended for this matchup."
    return text


def interpret(
    snap: GameStateSnapshot,
    plan: QuestionPlan,
    response: DecisionResponse,
    player_civ: str,
    opponent_civ: str,
) -> NimbleAdvice:
    answers = response.answers
    recs = _recommendations(snap, plan, answers)
    focus = Pick.from_answer(answers.get(FOCUS))
    gather = Pick.from_answer(answers.get(GATHER))
    counter = _counter(answers)
    escalate_pick = Pick.from_answer(answers.get(ESCALATE))
    escalate = (
        escalate_pick.choice in ("matchup_detail", "composition_plan")
        and escalate_pick.probability >= ESCALATE_MIN_PROBABILITY
    )

    priority, used = _priority(snap, recs, focus, counter)
    return NimbleAdvice(
        age=snap.age,
        player_civ=player_civ,
        opponent_civ=opponent_civ,
        priority=priority,
        actions=_actions(recs, used, gather),
        rationale=_rationale(snap, response, recs, player_civ, opponent_civ, escalate),
        recommendations=recs,
        focus=focus,
        gather=gather,
        counter=counter,
        military_urgency=_score_level(answers.get(MILITARY_URGENCY)),
        escalate=escalate,
        latency_ms=response.latency_ms,
        model=response.model,
        question_count=len(plan.questions),
        answers=answers,
    )
