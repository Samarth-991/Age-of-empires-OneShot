"""Perception Loop — Article III of the Constitution.

Screenshot → VLM (gemma4:latest) → GameStateSnapshot

This loop runs on its own asyncio task, completely decoupled from the
Strategist Loop. A slow or failing VLM call never blocks the next capture.
"""
from __future__ import annotations
import os
import asyncio
import base64
import json
import logging
import time
from collections import deque
from typing import Awaitable, Callable, Deque

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    OLLAMA_BASE_URL,
    VLM_MODEL,
    PERCEPTION_INTERVAL_SECS,
    SCREENSHOT_OUT_DIR,
    HASH_THRESHOLD,
    VLM_MAX_OUTPUT_TOKENS,
    RESOURCE_BAR_CROP,
    RESOURCE_BAR_SCALE,
)
from schemas import MAX_SNAPSHOT_BUILDINGS, MAX_SNAPSHOT_UNITS, ResourceBarReading
from config import (
    OPENROUTER_API_KEY,OPENROUTER_VLM_MODEL
)
from schemas import GameStateSnapshot
from .capture import CaptureLoop

log = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """\
You are the vision component of a real-time Age of Empires II: Definitive Edition \
coaching assistant. You will be shown one screenshot of the game in progress. \
Describe what you see by filling in the GameStateSnapshot JSON schema.
Do NOT invent numbers you cannot actually read.

What to look for:

- Top-left resource bar, left to right: WOOD (logs icon), FOOD (meat icon), GOLD, STONE, \
  then the villager count, then population shown as "current/cap" (e.g. "22/25" means \
  population 22, pop_cap 25), then the Age name. If a value is obscured or you are not \
  confident, set it to null rather than guessing.
- An animated "idle villager" icon near the population counter: set idle_villagers_visible \
  accordingly. If unclear, leave it false.
- Main viewport: count visible units by type and owner. Units in the player's own color \
  = "self"; differently colored = "enemy"; unclear = "unknown". Group identical units into \
  ONE entry with a count — never list the same unit type and owner twice. \
  At most """ + str(MAX_SNAPSHOT_UNITS) + """ entries.
- Visible buildings in the viewport (Town Center, houses, production buildings, etc.), \
  each name once, at most """ + str(MAX_SNAPSHOT_BUILDINGS) + """ names.
- Minimap: enemy dots near the player's economy = threat_level "high"; elsewhere = "low"; \
  no enemy activity = "none".
- Anything notable that doesn't fit a field above goes in notes as one short sentence.

Respond with ONLY a valid JSON object matching the GameStateSnapshot schema. \
No markdown fences, no explanation — just the JSON.

Schema:
{
  "age": "Dark|Feudal|Castle|Imperial",
  "population": int or null,
  "pop_cap": int or null,
  "resources": {"food": int|null, "wood": int|null, "gold": int|null, "stone": int|null},
  "visible_units": [{"label": str, "owner": "self|enemy|unknown", "approx_count": int}],
  "visible_buildings": [str, ...],
  "idle_villagers_visible": bool,
  "threat_level": "none|low|high",
  "notes": str
}
"""


RESOURCE_BAR_PROMPT = (
    "This is the Age of Empires II resource bar. Each resource has a square icon followed by a LARGE "
    "number to its right: wood (logs), food (meat), gold, stone. Report those large numbers. "
    "The tiny digit inside the bottom-left corner of each icon is a worker count — ignore it. "
    "After stone is a villager icon with a tiny digit, then the population written as current/cap "
    "like 22/25. Use null for any value you cannot read. Respond with JSON only."
)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text


def _img_to_b64(img) -> str:
    """Convert PIL Image to base64 PNG string."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resource_bar(img):
    """Enlarged crop of the top-left resource bar, which is too small to read at full-frame scale."""
    from PIL import Image
    w, h = img.size
    bar = img.crop((0, 0, int(w * RESOURCE_BAR_CROP[0]), int(h * RESOURCE_BAR_CROP[1])))
    return bar.resize((bar.width * RESOURCE_BAR_SCALE, bar.height * RESOURCE_BAR_SCALE), Image.LANCZOS)


def _encode_frame(img) -> tuple[str, str]:
    return _img_to_b64(img), _img_to_b64(_resource_bar(img))


def _parse_snapshot(raw: str) -> GameStateSnapshot | None:
    """Parse VLM text output into GameStateSnapshot; return None on failure."""
    text = raw.strip()
    # Strip accidental markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        return GameStateSnapshot.model_validate(data)
    except Exception as exc:
        log.warning("Failed to parse VLM output: %s\nRaw: %.200s", exc, raw)
        return None


class PerceptionLoop:
    """Async perception loop.

    Args:
        on_snapshot: Callback invoked with each successfully parsed snapshot.
        on_alarm: Callback invoked with a list of alarm strings each tick.
        interval: Seconds between captures (overrides config default).
        on_stage: Called with (stage, info) as each frame moves through the
            pipeline: "capturing", "reading_screen", "snapshot", "vlm_failed",
            "frame_skipped", "waiting". ``info`` carries timings in seconds.
    """

    def __init__(
        self,
        on_snapshot: Callable[[GameStateSnapshot], None],
        on_alarm: Callable[[list[str]], None] | None = None,
        interval: float = PERCEPTION_INTERVAL_SECS,
        on_stage: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._on_snapshot = on_snapshot
        self._on_alarm = on_alarm
        self._on_stage = on_stage
        self._followups: list[Callable[[GameStateSnapshot], Awaitable[None]]] = []
        self._interval = interval
        self._continuous = True
        self._wake = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._capture = CaptureLoop(
            out_dir=SCREENSHOT_OUT_DIR,
            interval=interval,
            hash_threshold=HASH_THRESHOLD,
        )
        
        key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
        if key:
            from langchain_openrouter import ChatOpenRouter
            os.environ["OPENROUTER_API_KEY"] = key
            self._vlm = ChatOpenRouter(model= OPENROUTER_VLM_MODEL,temperature=0,max_tokens=4096,max_retries=2,)
            log.info("Perception is using OpenRouter model: %s", OPENROUTER_VLM_MODEL)
        else:
            self._vlm = ChatOllama(
                    model=VLM_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0,
                    reasoning=False,
                    format=GameStateSnapshot.model_json_schema(),
                    num_predict=VLM_MAX_OUTPUT_TOKENS,
                    repeat_penalty=1.15,
                )
        if key:
            self._bar_vlm = self._vlm
        else:
            self._bar_vlm = ChatOllama(
                model=VLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0,
                reasoning=False,
                format=ResourceBarReading.model_json_schema(),
                num_predict=120,
            )
        self._history: Deque[GameStateSnapshot] = deque(maxlen=20)
        self._last_snapshot: GameStateSnapshot | None = None

    @property
    def snapshot_history(self) -> Deque[GameStateSnapshot]:
        return self._history

    @property
    def latest_snapshot(self) -> GameStateSnapshot | None:
        return self._last_snapshot

    def set_continuous(self, enabled: bool, interval: float | None = None) -> None:
        """When disabled, the loop waits until ``request_capture()`` (manual mode)."""
        self._continuous = enabled
        if interval is not None:
            self._interval = interval

    def request_capture(self) -> None:
        """Trigger one capture cycle (safe from any thread once ``run`` has started)."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._wake.set)

    def add_followup(self, handler: Callable[[GameStateSnapshot], Awaitable[None]]) -> None:
        """Run ``handler(snapshot)`` right after each new snapshot, before the next capture.

        Followups are awaited in registration order, so the next VLM call never
        competes with them for the model server.
        """
        self._followups.append(handler)

    def _stage(self, stage: str, **info) -> None:
        if self._on_stage is not None:
            self._on_stage(stage, info)

    async def _call_vlm(self, img) -> GameStateSnapshot | None:
        """Send screenshot to VLM and parse the structured output."""
        b64, bar_b64 = await asyncio.to_thread(_encode_frame, img)
        messages = [
            SystemMessage(content=VISION_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze this Age of Empires II screenshot."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            ),
        ]
        try:
            response = await self._vlm.ainvoke(messages)
            snap = _parse_snapshot(response.content)
        except Exception as exc:
            log.error("VLM call failed: %s", exc)
            return None
        if snap is None:
            return None
        bar = await self._read_resource_bar(bar_b64)
        if bar is None:
            return snap
        return snap.model_copy(update={
            "resources": {"food": bar.food, "wood": bar.wood, "gold": bar.gold, "stone": bar.stone},
            "population": bar.population,
            "pop_cap": bar.pop_cap,
        })

    async def _read_resource_bar(self, bar_b64: str) -> ResourceBarReading | None:
        """Read the HUD numbers from the enlarged bar crop alone — far more accurate than the full frame."""
        message = HumanMessage(content=[
            {"type": "text", "text": RESOURCE_BAR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bar_b64}"}},
        ])
        try:
            response = await self._bar_vlm.ainvoke([message])
            return ResourceBarReading.model_validate_json(_strip_fences(response.content))
        except Exception as exc:
            log.warning("Resource bar read failed: %s", exc)
            return None

    async def _process_one_frame(self) -> None:
        """Capture one screenshot and run VLM + followups when the frame changed."""
        self._stage("capturing")
        t0 = time.perf_counter()
        _path, img, needs_vlm = await asyncio.to_thread(self._capture.run_once)
        capture_secs = time.perf_counter() - t0
        if needs_vlm:
            self._stage("reading_screen", capture_secs=capture_secs)
            t0 = time.perf_counter()
            snap = await self._call_vlm(img)
            vlm_secs = time.perf_counter() - t0
            if snap is not None:
                self._history.append(snap)
                self._last_snapshot = snap
                self._on_snapshot(snap)
                self._stage("snapshot", capture_secs=capture_secs, vlm_secs=vlm_secs)
                if self._on_alarm:
                    from alarms.rules import check_alarms
                    alarms = check_alarms(snap, self._history)
                    if alarms:
                        self._on_alarm(alarms)
                for handler in self._followups:
                    try:
                        await handler(snap)
                    except Exception as exc:
                        log.error("Snapshot followup failed: %s", exc)
            else:
                self._stage("vlm_failed", capture_secs=capture_secs, vlm_secs=vlm_secs)
        else:
            log.debug("Frame skipped (no significant change)")
            self._stage("frame_skipped", capture_secs=capture_secs)

    async def _wait_between_captures(self, stop_event: asyncio.Event | None) -> bool:
        """Sleep until interval elapses, manual wake, or stop. Returns True if stopped."""
        self._stage("waiting")
        self._wake.clear()
        sleep_task = asyncio.create_task(asyncio.sleep(self._interval))
        wake_task = asyncio.create_task(self._wake.wait())
        tasks: list[asyncio.Task] = [sleep_task, wake_task]
        if stop_event is not None:
            tasks.append(asyncio.create_task(stop_event.wait()))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        return stop_event is not None and stop_event.is_set()

    async def _wait_manual_or_stop(self, stop_event: asyncio.Event | None) -> bool:
        """Wait for manual capture, switch to live, or stop. Returns True if stopped."""
        self._stage("paused")
        self._wake.clear()
        wake_task = asyncio.create_task(self._wake.wait())
        tasks: list[asyncio.Task] = [wake_task]
        if stop_event is not None:
            tasks.append(asyncio.create_task(stop_event.wait()))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if stop_event is not None and stop_event.is_set():
            return True
        return False

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Run until stop_event is set (or forever if None)."""
        self._loop = asyncio.get_running_loop()
        mode = "continuous" if self._continuous else "manual"
        log.info("Perception loop starting (%s, interval=%.1fs)", mode, self._interval)
        while stop_event is None or not stop_event.is_set():
            if not self._continuous:
                if await self._wait_manual_or_stop(stop_event):
                    break
            try:
                await self._process_one_frame()
            except Exception as exc:
                log.error("Perception loop error: %s", exc)

            if stop_event is not None and stop_event.is_set():
                break
            if self._continuous:
                if await self._wait_between_captures(stop_event):
                    break
