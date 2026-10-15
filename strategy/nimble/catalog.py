"""Age-gated catalog of the actions Nimble can recommend.

Every entry is one concrete thing the player can do — train a unit, place a
building, research a technology, or advance an age. ``relevant_in`` lists the
ages where the action is a realistic choice; it deliberately excludes ages
where the item is usually already done, because every extra entry adds a
Nimble question and therefore latency.

Unit, building and technology costs are standard AoE2:DE values. Age-up costs
follow the Game Constants table in AGENTS.md.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Age = Literal["Dark", "Feudal", "Castle", "Imperial"]
Kind = Literal["unit", "building", "tech", "age"]

_VERBS: dict[Kind, str] = {
    "unit": "train",
    "building": "build",
    "tech": "research",
    "age": "advance to",
}
_RESOURCE_ABBREV = {"food": "F", "wood": "W", "gold": "G", "stone": "S"}

D, F, C, I = "Dark", "Feudal", "Castle", "Imperial"


@dataclass(frozen=True)
class CatalogItem:
    id: str
    name: str
    kind: Kind
    prereq: str
    cost: Mapping[str, int]
    role: str
    relevant_in: tuple[Age, ...]
    military: bool = False

    @property
    def verb(self) -> str:
        return _VERBS[self.kind]

    def cost_text(self) -> str:
        if not self.cost:
            return "cost varies by civ"
        return " ".join(f"{amount}{_RESOURCE_ABBREV[res]}" for res, amount in self.cost.items())

    def shortfall(self, resources: Mapping[str, int | None]) -> dict[str, int] | None:
        """Missing amount per resource, or None when affordability can't be judged."""
        if not self.cost:
            return None
        missing: dict[str, int] = {}
        for res, amount in self.cost.items():
            have = resources.get(res)
            if have is None:
                return None
            if have < amount:
                missing[res] = amount - have
        return missing

    def affordability(self, resources: Mapping[str, int | None]) -> str:
        gap = self.shortfall(resources)
        if gap is None:
            return "affordability unknown"
        if not gap:
            return "affordable now"
        return "short " + ", ".join(f"{amount} {res}" for res, amount in gap.items())


def _age(id_: str, name: str, prereq: str, cost: dict[str, int], role: str, relevant_in: tuple[Age, ...]) -> CatalogItem:
    return CatalogItem(id_, name, "age", prereq, cost, role, relevant_in)


def _unit(id_: str, name: str, at: str, cost: dict[str, int], role: str, relevant_in: tuple[Age, ...], military: bool = True) -> CatalogItem:
    return CatalogItem(id_, name, "unit", f"at {at}", cost, role, relevant_in, military)


def _building(id_: str, name: str, needs: str, cost: dict[str, int], role: str, relevant_in: tuple[Age, ...]) -> CatalogItem:
    prereq = f"needs {needs}" if needs else ""
    return CatalogItem(id_, name, "building", prereq, cost, role, relevant_in)


def _tech(id_: str, name: str, at: str, cost: dict[str, int], role: str, relevant_in: tuple[Age, ...]) -> CatalogItem:
    return CatalogItem(id_, name, "tech", f"at {at}", cost, role, relevant_in)


# Order matters: when the question budget is exceeded, entries at the end of
# an age's list are dropped first.
CATALOG: tuple[CatalogItem, ...] = (
    # ---- Age advancement ---------------------------------------------------
    _age("feudal_age", "Feudal Age", "needs 2 Dark Age buildings such as Barracks, Mill or Lumber Camp",
         {"food": 500, "wood": 130}, "Unlocks Archery Range, Stable, Blacksmith and Feudal eco upgrades", (D,)),
    _age("castle_age", "Castle Age", "needs 2 Feudal Age buildings such as Blacksmith, Archery Range or Stable",
         {"food": 800, "gold": 200}, "Unlocks Knights, Castles, extra Town Centers and Castle eco upgrades", (F,)),
    _age("imperial_age", "Imperial Age", "needs a Castle or 2 Castle Age buildings such as University or Monastery",
         {"food": 1000, "gold": 800}, "Unlocks Trebuchets, final unit upgrades and Imperial eco upgrades", (C,)),

    # ---- Economy ---------------------------------------------------------------
    _unit("villager", "Villager", "Town Center", {"food": 50},
          "Economy unit; keep producing until about 120 villagers", (D, F, C, I), military=False),
    _building("house", "House", "", {"wood": 25},
              "Adds 5 population; needed whenever population is close to the cap", (D, F, C, I)),
    _building("farm", "Farm", "Mill or Town Center", {"wood": 60},
              "Endless food once sheep, boar and berries run out", (D, F, C, I)),
    _building("lumber_camp", "Lumber Camp", "", {"wood": 100},
              "Drop site next to a forest so wood villagers walk less", (D,)),
    _building("mill", "Mill", "", {"wood": 100},
              "Drop site for berries and farms; counts toward Feudal Age", (D,)),
    _building("mining_camp", "Mining Camp", "", {"wood": 100},
              "Drop site for gold or stone; gold is required from Feudal onward", (F,)),
    _tech("loom", "Loom", "Town Center", {"gold": 50},
          "Villagers gain HP and armor; protects against early scout or militia raids", (D,)),
    _tech("double_bit_axe", "Double-Bit Axe", "Lumber Camp", {"food": 100, "wood": 50},
          "Wood villagers gather 20% faster", (F,)),
    _tech("horse_collar", "Horse Collar", "Mill", {"food": 75, "wood": 75},
          "Farms hold more food", (F,)),
    _tech("wheelbarrow", "Wheelbarrow", "Town Center", {"food": 175, "wood": 50},
          "Villagers move faster and carry more", (F,)),
    _building("town_center", "Town Center", "Castle Age", {"wood": 275, "stone": 100},
              "Second or third Town Center to boom villager production", (C,)),
    _tech("bow_saw", "Bow Saw", "Lumber Camp", {"food": 150, "wood": 100},
          "Wood villagers gather faster again", (C,)),
    _tech("heavy_plow", "Heavy Plow", "Mill", {"food": 125, "wood": 125},
          "Farms hold more food and farmers carry more", (C,)),
    _tech("gold_mining", "Gold Mining", "Mining Camp", {"food": 100, "wood": 75},
          "Gold miners work 15% faster; matters for gold-heavy armies", (C,)),
    _tech("hand_cart", "Hand Cart", "Town Center", {"food": 300, "wood": 200},
          "Villagers move faster and carry more (requires Wheelbarrow)", (I,)),
    _tech("two_man_saw", "Two-Man Saw", "Lumber Camp", {"food": 300, "wood": 200},
          "Final wood gathering upgrade", (I,)),
    _tech("crop_rotation", "Crop Rotation", "Mill", {"food": 250, "wood": 250},
          "Final farm upgrade", (I,)),

    # ---- Military buildings ----------------------------------------------------
    _building("barracks", "Barracks", "", {"wood": 175},
              "Infantry production; needed for Archery Range and Stable", (D,)),
    _building("palisade_walls", "Palisade Walls", "", {"wood": 2},
              "Cheap walls (cost per tile) to block early raids", (D,)),
    _building("archery_range", "Archery Range", "Barracks", {"wood": 175},
              "Archers and Skirmishers", (F,)),
    _building("stable", "Stable", "Barracks", {"wood": 175},
              "Scout Cavalry, Knights and Camels", (F,)),
    _building("blacksmith", "Blacksmith", "Feudal Age", {"wood": 150},
              "Attack and armor upgrades; counts toward Castle Age", (F,)),
    _building("castle", "Castle", "Castle Age", {"stone": 650},
              "Unique units, unique techs and a strong defensive anchor", (C, I)),
    _building("monastery", "Monastery", "Castle Age", {"wood": 175},
              "Monks for healing, relics and converting expensive units", (C,)),
    _building("university", "University", "Castle Age", {"wood": 200},
              "Ballistics and building upgrades; counts toward Imperial Age", (C,)),
    _building("siege_workshop", "Siege Workshop", "Blacksmith", {"wood": 200},
              "Rams against buildings and Mangonels against massed archers", (C,)),

    # ---- Units -----------------------------------------------------------------
    _unit("militia", "Militia", "Barracks", {"food": 60, "gold": 20},
          "Early infantry for a drush or to chase off scouts", (D,)),
    _unit("scout_cavalry", "Scout Cavalry", "Stable", {"food": 80},
          "Fast raider; strong against villagers and lone archers", (F,)),
    _unit("archer", "Archer", "Archery Range", {"wood": 25, "gold": 45},
          "Ranged unit that beats infantry", (F, C)),
    _unit("skirmisher", "Skirmisher", "Archery Range", {"food": 25, "wood": 35},
          "No-gold unit that hard-counters archers", (F, C)),
    _unit("spearman", "Spearman", "Barracks", {"food": 35, "wood": 25},
          "Cheap infantry that hard-counters cavalry", (F, C)),
    _unit("knight", "Knight", "Stable", {"food": 60, "gold": 75},
          "Heavy cavalry that beats archers and siege", (C, I)),
    _unit("camel_rider", "Camel Rider", "Stable", {"food": 55, "gold": 60},
          "Anti-cavalry mounted unit that counters Knights", (C,)),
    _unit("monk", "Monk", "Monastery", {"gold": 100},
          "Converts expensive units such as Knights, elephants and unique units", (C,)),
    _unit("mangonel", "Mangonel", "Siege Workshop", {"wood": 160, "gold": 135},
          "Area damage that breaks massed archers and skirmishers", (C,)),
    _unit("battering_ram", "Battering Ram", "Siege Workshop", {"wood": 160, "gold": 75},
          "Destroys buildings and is resistant to arrows", (C,)),
    _unit("unique_unit", "the civilization's unique unit", "Castle", {},
          "Signature unit of your civilization", (C, I)),
    _unit("trebuchet", "Trebuchet", "Castle", {"wood": 200, "gold": 200},
          "Long-range siege that outranges Castles and towers", (I,)),

    # ---- Military upgrades -------------------------------------------------------
    _tech("fletching", "Fletching", "Blacksmith", {"food": 100, "gold": 50},
          "+1 attack and range for archers and towers", (F,)),
    _tech("crossbowman", "Crossbowman upgrade", "Archery Range", {"food": 125, "gold": 75},
          "Upgrades Archers to Crossbowmen", (C,)),
    _tech("pikeman", "Pikeman upgrade", "Barracks", {"food": 215, "gold": 90},
          "Upgrades Spearmen to Pikemen", (C,)),
    _tech("bodkin_arrow", "Bodkin Arrow", "Blacksmith", {"food": 200, "gold": 100},
          "+1 attack and range for archers", (C,)),
    _tech("halberdier", "Halberdier upgrade", "Barracks", {"food": 300, "gold": 600},
          "Upgrades Pikemen to Halberdiers", (I,)),
    _tech("arbalester", "Arbalester upgrade", "Archery Range", {"food": 350, "gold": 300},
          "Upgrades Crossbowmen to Arbalesters", (I,)),
    _tech("cavalier", "Cavalier upgrade", "Stable", {"food": 300, "gold": 300},
          "Upgrades Knights to Cavaliers", (I,)),
    _tech("hussar", "Hussar upgrade", "Stable", {"food": 500, "gold": 600},
          "Upgrades Light Cavalry to Hussars", (I,)),
    _tech("blast_furnace", "Blast Furnace", "Blacksmith", {"food": 275, "gold": 225},
          "+2 melee attack for infantry and cavalry", (I,)),
    _tech("bracer", "Bracer", "Blacksmith", {"food": 300, "gold": 200},
          "+1 attack and range for archers", (I,)),
    _tech("chemistry", "Chemistry", "University", {"food": 300, "gold": 200},
          "+1 attack for ranged units; required for gunpowder units", (I,)),
)

CATALOG_BY_ID: dict[str, CatalogItem] = {item.id: item for item in CATALOG}


def items_for_age(age: Age) -> list[CatalogItem]:
    return [item for item in CATALOG if age in item.relevant_in]


def military_units_for_age(age: Age) -> list[CatalogItem]:
    return [item for item in items_for_age(age) if item.kind == "unit" and item.military]
