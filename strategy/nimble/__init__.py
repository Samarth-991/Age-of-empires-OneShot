"""Fast AoE2 coaching with the Nimble decision model (Ollama ``/v1/systemone``).

Typical use::

    advisor = NimbleAdvisor("Mongols", "Vietnamese")
    advice = advisor.advise(snapshot, history)
    print(advice.to_markdown())
"""
from .advisor import NimbleAdvice, NimbleAdvisor, Pick, Recommendation, interpret
from .catalog import CATALOG, CatalogItem, items_for_age
from .client import DecisionResponse, NimbleClient, NimbleError
from .loop import NimbleLoop
from .questions import QuestionPlan, build_question_plan
from .state import build_state

__all__ = [
    "CATALOG",
    "CatalogItem",
    "DecisionResponse",
    "NimbleAdvice",
    "NimbleAdvisor",
    "NimbleClient",
    "NimbleError",
    "NimbleLoop",
    "Pick",
    "QuestionPlan",
    "Recommendation",
    "build_question_plan",
    "build_state",
    "interpret",
    "items_for_age",
]
