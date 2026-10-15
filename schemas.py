from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_SNAPSHOT_UNITS = 12
MAX_SNAPSHOT_BUILDINGS = 12


class DetectedUnit(BaseModel):
    label: str = Field(description="Unit type name, e.g. 'knight', 'villager', 'skirmisher'")
    owner: Literal["self", "enemy", "unknown"] = Field(
        description="'self' for player units, 'enemy' for opponent, 'unknown' when unclear"
    )
    approx_count: int = Field(description="Approximate number of this unit visible on screen")


class GameStateSnapshot(BaseModel):
    age: Literal["Dark", "Feudal", "Castle", "Imperial"] = Field(
        description="Current in-game age"
    )
    population: int | None = Field(
        default=None,
        description="Current population count; null if unreadable",
    )
    pop_cap: int | None = Field(
        default=None,
        description="Current population cap; null if unreadable",
    )
    resources: dict[str, int | None] = Field(
        default_factory=dict,
        description="Resource counts keyed by name: food, wood, gold, stone. Null if unreadable.",
    )
    visible_units: list[DetectedUnit] = Field(
        default_factory=list,
        max_length=MAX_SNAPSHOT_UNITS,
        description="One entry per unit type and owner (never repeat a pair)",
    )
    visible_buildings: list[str] = Field(
        default_factory=list,
        max_length=MAX_SNAPSHOT_BUILDINGS,
        description="Distinct building names visible in the viewport",
    )
    idle_villagers_visible: bool = Field(
        default=False,
        description="True when the idle-villager icon is visible near the population counter",
    )
    threat_level: Literal["none", "low", "high"] = Field(
        default="none",
        description="Threat assessment based on enemy unit positions on minimap",
    )
    notes: str = Field(
        default="",
        max_length=300,
        description="One short sentence on anything notable not captured by other fields",
    )

    @field_validator("visible_units", mode="before")
    @classmethod
    def _merge_units(cls, v: object) -> object:
        """Collapse repeated (label, owner) pairs, keeping the largest count."""
        if not isinstance(v, list):
            return v
        merged: dict[tuple[str, str], dict] = {}
        for item in v:
            if not isinstance(item, dict) or "label" not in item:
                continue
            key = (str(item["label"]).strip().lower(), str(item.get("owner", "unknown")))
            prev = merged.get(key)
            if prev is None or (item.get("approx_count") or 0) > (prev.get("approx_count") or 0):
                merged[key] = item
        return list(merged.values())[:MAX_SNAPSHOT_UNITS]

    @field_validator("visible_buildings", mode="before")
    @classmethod
    def _dedupe_buildings(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        seen: dict[str, str] = {}
        for b in v:
            seen.setdefault(str(b).strip().lower(), str(b).strip())
        return list(seen.values())[:MAX_SNAPSHOT_BUILDINGS]

    @field_validator("notes", mode="before")
    @classmethod
    def _clip_notes(cls, v: object) -> object:
        return v[:300] if isinstance(v, str) else v


class ResourceBarReading(BaseModel):
    """Numbers read from the enlarged top-left HUD crop."""

    wood: int | None = None
    food: int | None = None
    gold: int | None = None
    stone: int | None = None
    population: int | None = None
    pop_cap: int | None = None


class MatchContext(BaseModel):
    """Metadata provided at session start — never changes mid-game."""

    player_civ: str
    opponent_civ: str | None = None


STRATEGY_MAX_WORDS = 300
STRATEGY_MAX_BULLETS = 3
STRATEGY_MAX_BULLET_WORDS = 25
STRATEGY_SECTIONS = ("economy", "military", "tech_and_buildings", "scouting")


def _clip_words(text: object, max_words: int) -> str:
    words = str(text).split()
    return " ".join(words[:max_words])


class StrategyUpdate(BaseModel):
    """Structured strategist advice — specific bullets by domain, ≤300 words total."""

    age: Literal["Dark", "Feudal", "Castle", "Imperial"]
    priority: str = Field(
        description="The single most important action right now, with numbers (≤25 words)",
    )
    economy: list[str] = Field(
        default_factory=list,
        max_length=STRATEGY_MAX_BULLETS,
        description="1-3 bullets: villager counts per resource, TC production, eco upgrades",
    )
    military: list[str] = Field(
        default_factory=list,
        max_length=STRATEGY_MAX_BULLETS,
        description="1-3 bullets: which units to train and how many, what they counter",
    )
    tech_and_buildings: list[str] = Field(
        default_factory=list,
        max_length=STRATEGY_MAX_BULLETS,
        description="1-3 bullets: buildings to add, researches, age-up timing with costs",
    )
    scouting: list[str] = Field(
        default_factory=list,
        max_length=STRATEGY_MAX_BULLETS,
        description="0-2 bullets: what to scout for and why",
    )
    matchup_note: str = Field(
        default="",
        description="One line on the civ-vs-civ dynamic (≤25 words)",
    )

    @field_validator(*STRATEGY_SECTIONS, mode="before")
    @classmethod
    def _normalize_bullets(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        bullets = [_clip_words(x, STRATEGY_MAX_BULLET_WORDS) for x in v]
        return [b for b in bullets if b][:STRATEGY_MAX_BULLETS]

    @field_validator("priority", "matchup_note", mode="before")
    @classmethod
    def _normalize_text(cls, v: object) -> str:
        return "" if v is None else _clip_words(v, STRATEGY_MAX_BULLET_WORDS)

    def word_count(self) -> int:
        parts = [self.priority, self.matchup_note]
        for section in STRATEGY_SECTIONS:
            parts.extend(getattr(self, section))
        return sum(len(p.split()) for p in parts)

    @model_validator(mode="after")
    def _enforce_word_budget(self) -> StrategyUpdate:
        """Drop trailing bullets (scouting first) until the update fits STRATEGY_MAX_WORDS."""
        for section in reversed(STRATEGY_SECTIONS):
            bullets = getattr(self, section)
            while self.word_count() > STRATEGY_MAX_WORDS and len(bullets) > 1:
                bullets.pop()
        return self
