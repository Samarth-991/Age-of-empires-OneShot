"""End-to-end CLI test — mimics a player by feeding one fixed screenshot
repeatedly through the real dual-loop orchestration used by `cli.py start`.

Run from src/aoe2_agent/:
    python tests/test_cli_e2e.py                  # 4 sends, 10s apart (~45s + model latency)
    python tests/test_cli_e2e.py --sends 3 --interval 5
    python tests/test_cli_e2e.py --fast           # 3 sends, 2s apart (quick smoke)

What it exercises (the same path as `cli.py start`):
    capture (faked) → PerceptionLoop → VLM → GameStateSnapshot
                    → StrategistLoop → Strategist agent → on_strategy callback
                    → (headless) overlay

Why a fake capture:
    The real CaptureLoop deduplicates identical frames (perceptual hash), so
    sending the SAME image would make frames 2..N skip the VLM entirely. A live
    game screen always changes, so this test forces needs_vlm=True every tick to
    faithfully mimic continuous play.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — run from anywhere
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.parent          # src/aoe2_agent/
sys.path.insert(0, str(HERE.parent))         # src/

LOGS_DIR = HERE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEST_LOG = LOGS_DIR / "cli_e2e_test.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(TEST_LOG, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("test_cli_e2e")

# ---------------------------------------------------------------------------
# Imports after path setup
# ---------------------------------------------------------------------------
from PIL import Image
import sys 
sys.path.append('/mnt/e/Personal/Samarth/repository/AOE-Agent/src/aoe2_agent')
from config import PERCEPTION_INTERVAL_SECS, STRATEGIST_INTERVAL_SECS
from schemas import GameStateSnapshot
from perception.loop import PerceptionLoop
from strategy.agent import StrategistLoop

SCREENSHOT_PATH = LOGS_DIR / "screenshots" / "screenshot_1782755179413.png"
PLAYER_CIV = "Britons"
OPPONENT_CIV = "Teutons"


# ---------------------------------------------------------------------------
# Headless overlay — same interface as CoachOverlay, records calls instead of rendering
# ---------------------------------------------------------------------------
class RecordingOverlay:
    """Drop-in stand-in for CoachOverlay that records every callback."""

    def __init__(self, player_civ: str = "") -> None:
        self.player_civ = player_civ
        self.snapshots: list[tuple[float, GameStateSnapshot]] = []
        self.strategies: list[tuple[float, str]] = []
        self.alarms: list[tuple[float, list[str]]] = []
        self.clears = 0

    def set_snapshot(self, snap: GameStateSnapshot) -> None:
        self.snapshots.append((time.time(), snap))
        log.info("overlay.set_snapshot  #%d  age=%s pop=%s/%s threat=%s",
                 len(self.snapshots), snap.age, snap.population, snap.pop_cap, snap.threat_level)

    def set_strategy(self, text: str) -> None:
        self.strategies.append((time.time(), text))
        log.info("overlay.set_strategy  #%d  (%d chars)", len(self.strategies), len(text))

    def set_alarms(self, alarms: list[str]) -> None:
        self.alarms.append((time.time(), alarms))
        log.info("overlay.set_alarms    %s", alarms)

    def clear_alarms(self) -> None:
        self.clears += 1


# ---------------------------------------------------------------------------
# Fake capture — returns the fixed image every tick with needs_vlm=True
# ---------------------------------------------------------------------------
class FixedImageCapture:
    """Replaces PerceptionLoop._capture. Mimics a continuously-changing screen
    by always returning the same image but forcing needs_vlm=True."""

    def __init__(self, img: Image.Image) -> None:
        self._img = img
        self.calls = 0

    def run_once(self) -> tuple[str, Image.Image, bool]:
        self.calls += 1
        log.info("capture.run_once  #%d  (sending fixed screenshot, needs_vlm=True)", self.calls)
        return (f"fixed_frame_{self.calls}.png", self._img, True)


# ---------------------------------------------------------------------------
# Main orchestration — mirrors cli._main_loop
# ---------------------------------------------------------------------------
async def run_e2e(sends: int, interval: float, strat_interval: float) -> bool:
    log.info("=" * 64)
    log.info("  CLI End-to-End Test")
    log.info("  %s vs %s", PLAYER_CIV, OPPONENT_CIV)
    log.info("  sends=%d  perception_interval=%.1fs  strategist_interval=%.1fs",
             sends, interval, strat_interval)
    log.info("  screenshot=%s", SCREENSHOT_PATH.name)
    log.info("=" * 64)

    if not SCREENSHOT_PATH.exists():
        log.error("Screenshot not found: %s", SCREENSHOT_PATH)
        log.error("Run `python cli.py start` once to generate a screenshot, then retry.")
        return False

    img = Image.open(SCREENSHOT_PATH).convert("RGB")
    log.info("Loaded screenshot %dx%d", img.size[0], img.size[1])

    overlay = RecordingOverlay(player_civ=f"{PLAYER_CIV} vs {OPPONENT_CIV}")
    stop_event = asyncio.Event()

    # ── Wire callbacks exactly as cli._main_loop does ──────────────────────────
    def on_snapshot(snap: GameStateSnapshot) -> None:
        overlay.set_snapshot(snap)

    def on_alarm(alarms: list[str]) -> None:
        overlay.set_alarms(alarms)

    def on_strategy(text: str) -> None:
        overlay.set_strategy(text)
        overlay.clear_alarms()

    # ── Real loops, but with a fake capture injected ──────────────────────────
    perception = PerceptionLoop(on_snapshot=on_snapshot, on_alarm=on_alarm, interval=interval)
    fake_capture = FixedImageCapture(img)
    perception._capture = fake_capture  # inject — forces needs_vlm=True each tick

    strategist = StrategistLoop(
        player_civ=PLAYER_CIV,
        opponent_civ=OPPONENT_CIV,
        on_strategy=on_strategy,
        interval=strat_interval,
        perception_loop=perception,
    )

    # ── Stop after the requested number of perception sends ───────────────────
    async def stopper() -> None:
        # allow `sends` perception ticks plus a margin for the final strategist cycle
        await asyncio.sleep(sends * interval + strat_interval + 5)
        log.info("Stop timer elapsed — signalling shutdown")
        stop_event.set()

    t0 = time.time()
    try:
        await asyncio.gather(
            perception.run(stop_event),
            strategist.run(stop_event),
            stopper(),
        )
    except asyncio.CancelledError:
        pass

    elapsed = time.time() - t0
    log.info("Loops stopped after %.1fs", elapsed)

    return _evaluate(overlay, fake_capture, sends)


# ---------------------------------------------------------------------------
# Single synchronous tick — fast CI-style check (no async loops)
# ---------------------------------------------------------------------------
async def run_once() -> bool:
    """Run exactly ONE perception + strategist cycle, synchronously orchestrated.

    Faster and more deterministic than the full loop test — ideal for CI or a
    quick "is the pipeline wired correctly?" check. Returns True on success.
    """
    log.info("=" * 64)
    log.info("  CLI Single-Tick Test (--once)")
    log.info("  %s vs %s", PLAYER_CIV, OPPONENT_CIV)
    log.info("=" * 64)

    if not SCREENSHOT_PATH.exists():
        log.error("Screenshot not found: %s", SCREENSHOT_PATH)
        return False

    img = Image.open(SCREENSHOT_PATH).convert("RGB")
    overlay = RecordingOverlay(player_civ=f"{PLAYER_CIV} vs {OPPONENT_CIV}")

    # ── One perception tick ────────────────────────────────────────────────────
    perception = PerceptionLoop(
        on_snapshot=lambda s: overlay.set_snapshot(s),
        on_alarm=lambda a: overlay.set_alarms(a),
        interval=1.0,
    )
    perception._capture = FixedImageCapture(img)

    log.info("Running one perception tick (VLM)…")
    _path, frame, needs_vlm = perception._capture.run_once()
    snap = await perception._call_vlm(frame)
    if snap is None:
        log.error("VLM returned no parseable snapshot — perception failed.")
        return False
    perception._history.append(snap)
    perception._last_snapshot = snap
    overlay.set_snapshot(snap)

    # ── One strategist invocation ──────────────────────────────────────────────
    log.info("Running one strategist invocation…")
    strategist = StrategistLoop(
        player_civ=PLAYER_CIV,
        opponent_civ=OPPONENT_CIV,
        on_strategy=lambda t: (overlay.set_strategy(t), overlay.clear_alarms()),
        interval=1.0,
        perception_loop=perception,
    )

    from strategy.agent import _snapshot_to_prompt, _extract_last_content
    from collections import deque as _deque
    prompt = _snapshot_to_prompt(snap, _deque(), PLAYER_CIV, OPPONENT_CIV)
    result = await asyncio.to_thread(
        strategist._agent.invoke,
        {"messages": [{"role": "user", "content": prompt}]},
    )
    strategy_text = _extract_last_content(result)
    if strategy_text:
        overlay.set_strategy(strategy_text)

    ok = bool(overlay.snapshots and overlay.strategies and overlay.strategies[-1][1].strip())
    log.info("─" * 64)
    log.info("  snapshots=%d  strategies=%d", len(overlay.snapshots), len(overlay.strategies))
    if overlay.strategies:
        log.info("─ strategy ─\n%s", overlay.strategies[-1][1])
    log.info("  SINGLE-TICK TEST %s", "PASSED ✓" if ok else "FAILED ✗")
    log.info("─" * 64)
    return ok


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _evaluate(overlay: RecordingOverlay, capture: FixedImageCapture, sends: int) -> bool:
    log.info("─" * 64)
    log.info("RESULTS")
    log.info("─" * 64)
    log.info("capture.run_once calls : %d", capture.calls)
    log.info("snapshots received     : %d", len(overlay.snapshots))
    log.info("strategy updates        : %d", len(overlay.strategies))
    log.info("alarm pushes           : %d", len(overlay.alarms))

    checks: dict[str, bool] = {
        f"capture ran >= {sends} times": capture.calls >= sends,
        "perception produced >=1 snapshot": len(overlay.snapshots) >= 1,
        "strategist produced >=1 update": len(overlay.strategies) >= 1,
        "latest strategy is non-empty": bool(overlay.strategies and overlay.strategies[-1][1].strip()),
    }

    all_pass = True
    for name, ok in checks.items():
        log.info("  %s  %s", "✓" if ok else "✗", name)
        all_pass = all_pass and ok

    if overlay.strategies:
        log.info("─ latest strategy update ─\n%s", overlay.strategies[-1][1])
    else:
        log.warning("NO strategy updates were produced — this is the 'messages not updating' bug.")
        log.warning("Inspect logs/agent.log for sub-agent errors or empty responses.")

    log.info("─" * 64)
    log.info("CLI E2E TEST %s", "PASSED ✓" if all_pass else "FAILED ✗")
    log.info("Logs: %s  and  %s", TEST_LOG, LOGS_DIR / "agent.log")
    log.info("─" * 64)
    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end CLI dual-loop test")
    parser.add_argument("--sends", type=int, default=4,
                        help="Number of times the screenshot is sent (default 4)")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Seconds between sends / perception ticks (default 10)")
    parser.add_argument("--strat-interval", type=float, default=None,
                        help="Strategist interval seconds (default = same as --interval)")
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke: 3 sends, 2s apart")
    parser.add_argument("--once", action="store_true",
                        help="Single synchronous tick (one VLM + one strategist call). Fastest check.")
    args = parser.parse_args()

    if args.once:
        ok = asyncio.run(run_once())
        sys.exit(0 if ok else 1)

    if args.fast:
        sends, interval = 3, 2.0
    else:
        sends, interval = args.sends, args.interval

    strat_interval = args.strat_interval if args.strat_interval is not None else interval

    ok = asyncio.run(run_e2e(sends, interval, strat_interval))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
