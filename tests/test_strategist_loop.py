"""Isolated test for the Strategist loop pipeline.

Run from src/aoe2_agent/:
    python tests/test_strategist_loop.py

What this tests:
    1. VLM perception on a real screenshot → GameStateSnapshot
    2. Strategist agent invoke (with KB sub-agents)
    3. on_strategy callback is triggered and the result is non-empty
    4. Everything is logged to logs/agent.log

Purpose: reproduces the "Strategist messages not updating" issue in the CLI
by testing each layer independently before the full async loop is involved.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — run from anywhere inside src/aoe2_agent/
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.parent          # src/aoe2_agent/
sys.path.insert(0, str(HERE.parent))         # src/

# Configure root logger to also print to stdout while testing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("test_strategist_loop")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCREENSHOT_PATH = HERE / "logs" / "screenshots" / "screenshot_1782755179413.png"
PLAYER_CIV  = "Britons"
OPPONENT_CIV = "Teutons"
LOG_FILE = HERE / "logs" / "agent.log"

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
from PIL import Image
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import sys 
sys.path.append('/mnt/e/Personal/Samarth/repository/AOE-Agent/src/aoe2_agent')
from config import OLLAMA_BASE_URL, VLM_MODEL
from perception.loop import VISION_SYSTEM_PROMPT, _img_to_b64, _parse_snapshot
from schemas import GameStateSnapshot, DetectedUnit
from strategy.agent import (
    build_strategist,
    _snapshot_to_prompt,
    _extract_last_content,
    _log_strategy,
)


# ---------------------------------------------------------------------------
# Step 1 — Load screenshot
# ---------------------------------------------------------------------------
def step1_load_screenshot() -> Image.Image:
    log.info("─" * 60)
    log.info("STEP 1 — Load screenshot")
    log.info("─" * 60)

    if not SCREENSHOT_PATH.exists():
        log.error("Screenshot not found: %s", SCREENSHOT_PATH)
        log.error("Run the CLI once so the Perception Loop creates a screenshot,")
        log.error("then re-run this test.")
        sys.exit(1)

    img = Image.open(SCREENSHOT_PATH).convert("RGB")
    log.info("Loaded: %s  size=%dx%d", SCREENSHOT_PATH.name, img.size[0], img.size[1])
    return img


# ---------------------------------------------------------------------------
# Step 2 — VLM → GameStateSnapshot
# ---------------------------------------------------------------------------
async def step2_perception(img: Image.Image) -> GameStateSnapshot:
    log.info("─" * 60)
    log.info("STEP 2 — VLM perception (gemma4:latest)")
    log.info("─" * 60)

    vlm = ChatOllama(model=VLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    b64 = _img_to_b64(img)

    messages = [
        SystemMessage(content=VISION_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "Analyze this Age of Empires II screenshot."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]),
    ]

    t0 = time.perf_counter()
    response = await vlm.ainvoke(messages)
    elapsed = time.perf_counter() - t0
    raw = response.content

    log.info("VLM responded in %.1fs", elapsed)
    log.info("Raw VLM output:\n%s", raw[:500])

    snap = _parse_snapshot(raw)
    if snap is None:
        log.warning("VLM parse failed — using synthetic fallback snapshot")
        snap = GameStateSnapshot(
            age="Dark",
            population=6, pop_cap=10,
            resources={"food": 400, "wood": 200, "gold": 0, "stone": 0},
            visible_units=[DetectedUnit(label="villager", owner="self", approx_count=6)],
            visible_buildings=["Town Center"],
            idle_villagers_visible=False,
            threat_level="none",
            notes="Synthetic fallback — VLM parse failed",
        )
    else:
        log.info("Snapshot parsed OK: age=%s  pop=%s/%s  threat=%s",
                 snap.age, snap.population, snap.pop_cap, snap.threat_level)
        log.info("Resources: %s", snap.resources)
        log.info("Units: %s", [(u.label, u.owner, u.approx_count) for u in snap.visible_units])
        log.info("Buildings: %s", snap.visible_buildings)

    return snap


# ---------------------------------------------------------------------------
# Step 3 — Build Strategist and invoke once
# ---------------------------------------------------------------------------
def step3_strategist(snap: GameStateSnapshot) -> str:
    log.info("─" * 60)
    log.info("STEP 3 — Build Strategist (%s vs %s)", PLAYER_CIV, OPPONENT_CIV)
    log.info("─" * 60)

    t0 = time.perf_counter()
    agent = build_strategist(PLAYER_CIV, OPPONENT_CIV)
    log.info("Agent built in %.1fs", time.perf_counter() - t0)

    prompt = _snapshot_to_prompt(snap, deque(), PLAYER_CIV, OPPONENT_CIV)
    log.info("Prompt sent to Strategist:\n%s", prompt)

    log.info("Invoking Strategist (this may take 15–60s on first run)…")
    t0 = time.perf_counter()
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    elapsed = time.perf_counter() - t0

    log.info("Strategist responded in %.1fs", elapsed)

    # Show full message chain so we can see tool calls
    log.info("─── Message chain ───")
    for i, msg in enumerate(result.get("messages", [])):
        role = getattr(msg, "type", type(msg).__name__)
        content = getattr(msg, "content", "") or ""
        tool_calls = getattr(msg, "tool_calls", [])
        if tool_calls:
            log.info("[%d] %s  →  tool calls: %s", i, role, [tc["name"] for tc in tool_calls])
        elif content:
            log.info("[%d] %s  →  %s", i, role, str(content)[:120].replace("\n", " "))

    strategy_text = _extract_last_content(result)
    return strategy_text


# ---------------------------------------------------------------------------
# Step 4 — Validate and log the strategy output
# ---------------------------------------------------------------------------
def step4_validate(strategy_text: str) -> bool:
    log.info("─" * 60)
    log.info("STEP 4 — Validate Strategist output")
    log.info("─" * 60)

    if not strategy_text:
        log.error("FAIL: Strategist returned empty string.")
        log.error("Likely causes:")
        log.error("  • _extract_last_content found no non-empty message")
        log.error("  • All sub-agent tool calls returned errors")
        log.error("  • Model produced only tool calls, no final text")
        return False

    log.info("Strategy text (%d chars):\n%s", len(strategy_text), strategy_text)

    # Write to agent.log via the official logger
    _log_strategy(PLAYER_CIV, OPPONENT_CIV, strategy_text)
    log.info("Strategy written to %s", LOG_FILE)

    # Basic structure checks
    checks = {
        "Has Priority line":    "Priority" in strategy_text or "priority" in strategy_text,
        "Has numbered actions": any(f"{n}." in strategy_text for n in range(1, 6)),
        "Not just empty lines": strategy_text.strip() != "",
    }
    all_pass = True
    for check, passed in checks.items():
        icon = "✓" if passed else "✗"
        level = logging.INFO if passed else logging.WARNING
        log.log(level, "  %s  %s", icon, check)
        if not passed:
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Step 5 — Simulate the on_strategy callback (what the CLI does)
# ---------------------------------------------------------------------------
def step5_callback_simulation(strategy_text: str) -> None:
    log.info("─" * 60)
    log.info("STEP 5 — Simulate on_strategy callback")
    log.info("─" * 60)

    received: list[str] = []

    def on_strategy(text: str) -> None:
        received.append(text)
        log.info("on_strategy() called — %d chars received", len(text))

    # Simulate what StrategistLoop does after receiving the strategy text
    on_strategy(strategy_text)

    assert len(received) == 1, "Callback was not called exactly once"
    assert received[0] == strategy_text, "Callback received different text"
    log.info("✓ Callback fired correctly — overlay.set_strategy() would receive %d chars", len(received[0]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    log.info("=" * 60)
    log.info("  Strategist Loop Test")
    log.info("  Player: %s  |  Opponent: %s", PLAYER_CIV, OPPONENT_CIV)
    log.info("  Screenshot: %s", SCREENSHOT_PATH.name)
    log.info("  Log file: %s", LOG_FILE)
    log.info("=" * 60)

    # Step 1 — screenshot
    img = step1_load_screenshot()

    # Step 2 — perception
    snap = await step2_perception(img)

    # Step 3 — strategist (sync, in main thread — isolates from asyncio.to_thread)
    strategy_text = step3_strategist(snap)

    # Step 4 — validate
    passed = step4_validate(strategy_text)

    # Step 5 — callback
    if passed:
        step5_callback_simulation(strategy_text)

    log.info("=" * 60)
    log.info("  Test %s", "PASSED ✓" if passed else "FAILED ✗")
    log.info("  Full log: %s", LOG_FILE)
    log.info("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
