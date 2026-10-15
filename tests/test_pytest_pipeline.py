"""Pytest-compatible wrappers for the AoE2 Coaching Agent pipeline.

Run from src/aoe2_agent/:
    pytest tests/test_pytest_pipeline.py -v
    pytest tests/test_pytest_pipeline.py -v -m "not integration"   # unit tests only

Two layers:
  * Unit tests (always run, no models needed): schema, alarms, KB tools,
    snapshot→prompt formatting, overlay recording, fake capture forcing VLM.
  * Integration tests (need Ollama / OpenRouter): a single end-to-end tick.
    Automatically skipped when no Ollama server is reachable, so CI without
    models stays green.
"""
from __future__ import annotations

import asyncio
import socket
import sys
from collections import deque
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.parent          # src/aoe2_agent/
sys.path.insert(0, str(HERE.parent))         # src/

from config import OLLAMA_BASE_URL, ALARM_FOOD_FLOOR
from schemas import GameStateSnapshot, DetectedUnit
from alarms.rules import check_alarms
from knowledge.tools import (
    get_summary_info,
    get_civilization_info,
    get_unit_counters,
)
from strategy.agent import _snapshot_to_prompt, _extract_last_content

SCREENSHOT_PATH = HERE / "logs" / "screenshots" / "screenshot_1782755179413.png"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
def _ollama_reachable() -> bool:
    """Return True if an Ollama server is listening on the configured host:port."""
    try:
        host = OLLAMA_BASE_URL.split("://")[-1].split(":")[0]
        port = int(OLLAMA_BASE_URL.rsplit(":", 1)[-1])
    except Exception:
        host, port = "localhost", 11434
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


needs_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=f"Ollama not reachable at {OLLAMA_BASE_URL} — skipping integration test",
)


@pytest.fixture
def dark_age_snapshot() -> GameStateSnapshot:
    return GameStateSnapshot(
        age="Dark",
        population=6, pop_cap=10,
        resources={"food": 150, "wood": 200, "gold": 0, "stone": 0},
        visible_units=[DetectedUnit(label="villager", owner="self", approx_count=6)],
        visible_buildings=["Town Center", "House"],
        idle_villagers_visible=True,
        threat_level="none",
        notes="Test snapshot",
    )


# ---------------------------------------------------------------------------
# Unit tests — no models required
# ---------------------------------------------------------------------------
class TestSchema:
    def test_snapshot_constructs(self, dark_age_snapshot):
        assert dark_age_snapshot.age == "Dark"
        assert dark_age_snapshot.resources["food"] == 150

    def test_detected_unit_owner_literal(self):
        u = DetectedUnit(label="knight", owner="enemy", approx_count=3)
        assert u.owner == "enemy"


class TestAlarms:
    def test_low_food_fires(self, dark_age_snapshot):
        alarms = check_alarms(dark_age_snapshot, deque())
        assert any("FOOD" in a.upper() for a in alarms), f"expected low-food alarm, got {alarms}"

    def test_idle_villager_fires(self, dark_age_snapshot):
        alarms = check_alarms(dark_age_snapshot, deque())
        assert any("IDLE" in a.upper() for a in alarms)

    def test_healthy_snapshot_no_alarms(self):
        healthy = GameStateSnapshot(
            age="Castle", population=80, pop_cap=100,
            resources={"food": 600, "wood": 400, "gold": 300, "stone": 150},
            visible_units=[], visible_buildings=["Town Center"],
            idle_villagers_visible=False, threat_level="none", notes="",
        )
        assert check_alarms(healthy, deque()) == []


class TestKnowledgeBase:
    def test_summary_loads(self):
        text = get_summary_info()
        assert "Counter Matrix" in text or "counter" in text.lower()

    def test_civ_lookup(self):
        import json
        result = json.loads(get_civilization_info("Britons"))
        assert result["name"] == "Britons"
        assert result.get("bonuses")

    def test_counter_lookup(self):
        import json
        result = json.loads(get_unit_counters("archer"))
        assert "archer" in result


class TestPromptFormatting:
    def test_prompt_includes_both_civs(self, dark_age_snapshot):
        prompt = _snapshot_to_prompt(dark_age_snapshot, deque(), "Britons", "Teutons")
        assert "Britons" in prompt
        assert "Teutons" in prompt
        assert "Dark" in prompt

    def test_extract_last_content_dict(self):
        class _Msg:
            def __init__(self, c): self.content = c
        result = {"messages": [_Msg("first"), _Msg("final answer")]}
        assert _extract_last_content(result) == "final answer"

    def test_extract_last_content_empty(self):
        assert _extract_last_content({"messages": []}) == ""


class TestFakeCapture:
    def test_fixed_capture_forces_vlm(self):
        from tests.test_cli_e2e import FixedImageCapture, RecordingOverlay
        img = Image.new("RGB", (320, 240), (50, 80, 50))
        cap = FixedImageCapture(img)
        for _ in range(4):
            path, returned_img, needs_vlm = cap.run_once()
            assert needs_vlm is True   # never deduped
            assert returned_img is img
        assert cap.calls == 4

    def test_recording_overlay_records(self, dark_age_snapshot):
        from tests.test_cli_e2e import RecordingOverlay
        ov = RecordingOverlay()
        ov.set_snapshot(dark_age_snapshot)
        ov.set_strategy("do X")
        ov.clear_alarms()
        assert len(ov.snapshots) == 1
        assert len(ov.strategies) == 1
        assert ov.clears == 1


# ---------------------------------------------------------------------------
# Integration test — needs a live Ollama (and OpenRouter key if configured)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@needs_ollama
def test_single_tick_end_to_end():
    """One VLM + one strategist call through the real pipeline."""
    if not SCREENSHOT_PATH.exists():
        pytest.skip(f"No screenshot at {SCREENSHOT_PATH}")
    from tests.test_cli_e2e import run_once
    assert asyncio.run(run_once()) is True
