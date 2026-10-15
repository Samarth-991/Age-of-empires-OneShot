"""Turn a GameStateSnapshot into the compact ``state`` object Nimble judges."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from schemas import GameStateSnapshot

RESOURCES = ("food", "wood", "gold", "stone")


def pop_headroom(snap: GameStateSnapshot) -> int | None:
    if snap.population is None or snap.pop_cap is None:
        return None
    return snap.pop_cap - snap.population


def has_enemy_presence(snap: GameStateSnapshot) -> bool:
    return snap.threat_level != "none" or any(u.owner == "enemy" for u in snap.visible_units)


def resource_trend(
    snap: GameStateSnapshot, history: Sequence[GameStateSnapshot]
) -> dict[str, int] | None:
    """Change per resource versus three snapshots ago, or None without enough history."""
    if len(history) < 3:
        return None
    old = history[-3]
    trend = {
        res: snap.resources[res] - old.resources[res]
        for res in RESOURCES
        if snap.resources.get(res) is not None and old.resources.get(res) is not None
    }
    return trend or None


def _units(snap: GameStateSnapshot, owner: str) -> str:
    parts = [f"{u.approx_count}x {u.label}" for u in snap.visible_units if u.owner == owner]
    return ", ".join(parts) or "none seen"


def build_state(
    snap: GameStateSnapshot,
    *,
    player_civ: str,
    opponent_civ: str = "Unknown",
    history: Sequence[GameStateSnapshot] = (),
) -> dict[str, Any]:
    return {
        "matchup": f"{player_civ} (player) vs {opponent_civ} (opponent)",
        "age": snap.age,
        "population": snap.population,
        "pop_cap": snap.pop_cap,
        "pop_headroom": pop_headroom(snap),
        "resources": {res: snap.resources.get(res) for res in RESOURCES},
        "resource_trend": resource_trend(snap, history),
        "idle_villagers": snap.idle_villagers_visible,
        "threat_level": snap.threat_level,
        "own_units": _units(snap, "self"),
        "enemy_units": _units(snap, "enemy"),
        "buildings": ", ".join(snap.visible_buildings) or "none seen",
        "notes": snap.notes or "none",
    }
