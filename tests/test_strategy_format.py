"""Tests for structured strategist output parsing."""
from schemas import StrategyUpdate
from strategy.format import parse_strategy_response, strategy_to_markdown

SAMPLE = """{
  "age": "Feudal",
  "priority": "Build 2 houses before pop cap",
  "economy": ["Keep TC on villagers"],
  "military": ["Spears if knights spotted"],
  "scouting": ["Scout gold"],
  "tech_and_buildings": ["Barracks when 150W"],
  "matchup_note": "Teutons: knights soon",
  "rationale": "Pop 23/25 from snapshot."
}"""


def test_parse_strategy_json():
    update = parse_strategy_response(SAMPLE)
    assert update is not None
    assert update.age == "Feudal"
    assert len(update.economy) == 1
    assert update.military[0].startswith("Spears")


def test_parse_strips_markdown_fence():
    wrapped = "```json\n" + SAMPLE + "\n```"
    assert parse_strategy_response(wrapped) is not None


def test_strategy_to_markdown_sections():
    update = StrategyUpdate.model_validate_json(SAMPLE)
    md = strategy_to_markdown(update)
    assert "**Economy**" in md
    assert "**Military**" in md
    assert "Build 2 houses" in md


def test_bullets_capped_per_section_and_word_budget():
    long_bullet = " ".join(["word"] * 40)
    update = StrategyUpdate(
        age="Castle",
        priority="Advance to Imperial",
        economy=[long_bullet] * 5,
        military=[long_bullet] * 5,
        tech_and_buildings=[long_bullet] * 5,
        scouting=[long_bullet] * 5,
    )
    assert len(update.economy) <= 3
    assert all(len(b.split()) <= 25 for b in update.economy)
    assert update.word_count() <= 300


def test_fallback_does_not_repeat_lines():
    from strategy.format import summarize_raw_strategy
    raw = (
        "Build 2 houses now to avoid pop cap\n"
        "- Put 4 more villagers on wood for the Archery Range\n"
        "- Train 5 Spearmen to counter Knights\n"
        "- Scout the enemy gold mines with your Scout Cavalry\n"
    )
    update = summarize_raw_strategy(raw, "Feudal")
    all_bullets = update.economy + update.military + update.tech_and_buildings + update.scouting
    assert len(all_bullets) == len(set(all_bullets))
    assert update.priority.startswith("Build 2 houses")
