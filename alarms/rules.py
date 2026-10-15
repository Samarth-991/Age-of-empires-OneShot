"""Deterministic alarm functions — Article V of the Constitution.

All alarms are pure Python: no LLM calls, no I/O, no asyncio.
Every function takes the latest GameStateSnapshot (plus optional history)
and returns a list of human-readable alarm strings. An empty list = no alarms.

Thresholds come from config.py so they can be changed without touching this file.
"""
from __future__ import annotations

from collections import deque
from typing import Deque

from config import (
    ALARM_FOOD_FLOOR,
    ALARM_GOLD_FLOOR,
    ALARM_POP_HEADROOM,
    ALARM_WOOD_FLOOR,
)
from schemas import GameStateSnapshot


# ---------------------------------------------------------------------------
# Individual alarm checks — each returns a list (empty = OK)
# ---------------------------------------------------------------------------


def _check_food(snap: GameStateSnapshot) -> list[str]:
    food = snap.resources.get("food")
    if food is not None and food < ALARM_FOOD_FLOOR:
        return [f"LOW FOOD: {food} (floor {ALARM_FOOD_FLOOR})"]
    return []


def _check_wood(snap: GameStateSnapshot) -> list[str]:
    wood = snap.resources.get("wood")
    if wood is not None and wood < ALARM_WOOD_FLOOR:
        return [f"LOW WOOD: {wood} (floor {ALARM_WOOD_FLOOR})"]
    return []


def _check_gold(snap: GameStateSnapshot) -> list[str]:
    gold = snap.resources.get("gold")
    if gold is not None and gold < ALARM_GOLD_FLOOR and snap.age in ("Castle", "Imperial"):
        return [f"LOW GOLD: {gold} (floor {ALARM_GOLD_FLOOR})"]
    return []


def _check_idle_villagers(snap: GameStateSnapshot) -> list[str]:
    if snap.idle_villagers_visible:
        return ["IDLE VILLAGERS: assign them to a resource immediately"]
    return []


def _check_threat(snap: GameStateSnapshot) -> list[str]:
    if snap.threat_level == "high":
        return ["HIGH THREAT: enemy units approaching your base — react now"]
    return []


def _check_population(snap: GameStateSnapshot) -> list[str]:
    if snap.population is not None and snap.pop_cap is not None:
        headroom = snap.pop_cap - snap.population
        if headroom <= ALARM_POP_HEADROOM:
            return [f"POPULATION CAP NEAR: {snap.population}/{snap.pop_cap} — build houses"]
    return []


def _check_resource_drain(
    snap: GameStateSnapshot, history: Deque[GameStateSnapshot]
) -> list[str]:
    """Fire if food has dropped by more than 300 in the last 3 snapshots (sudden drain)."""
    if len(history) < 3:
        return []
    older = history[-3]
    old_food = older.resources.get("food")
    new_food = snap.resources.get("food")
    if old_food is not None and new_food is not None and (old_food - new_food) > 300:
        return [f"RAPID FOOD DROP: {old_food} → {new_food} — check for mass training or attack"]
    return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_alarms(
    snap: GameStateSnapshot,
    history: Deque[GameStateSnapshot] | None = None,
) -> list[str]:
    """Run all deterministic alarm checks and return all triggered messages.

    Args:
        snap: The latest GameStateSnapshot from the Perception Loop.
        history: Recent snapshot history (deque). Optional but enables trend alarms.

    Returns:
        List of alarm strings. Empty list means no alarms.
    """
    _history: Deque[GameStateSnapshot] = history if history is not None else deque()
    alarms: list[str] = []

    alarms.extend(_check_food(snap))
    alarms.extend(_check_wood(snap))
    alarms.extend(_check_gold(snap))
    alarms.extend(_check_idle_villagers(snap))
    alarms.extend(_check_threat(snap))
    alarms.extend(_check_population(snap))
    alarms.extend(_check_resource_drain(snap, _history))

    return alarms
