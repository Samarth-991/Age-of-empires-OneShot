"""Session manager for the web UI.

Owns the loop orchestration (Perception + Strategist + optional Nimble fast
coach) in a background asyncio task and exposes the latest state for the web
frontend to poll/stream.

Key design difference from the CLI overlay:
  - `strategy` is STICKY: it is only replaced when a NEW strategy text arrives,
    so frequent game-state updates never wipe the advice from the screen.
  - `status` ("idle" / "analyzing …") is tracked separately from the strategy.
  - `nimble` holds the latest fast-coach advice; Nimble failures are reported
    in `nimble_error` and never stop the other loops.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from schemas import GameStateSnapshot
from config import (
    NIMBLE_ENABLED,
    NIMBLE_INTERVAL_SECS,
    PERCEPTION_INTERVAL_SECS,
    STRATEGIST_INTERVAL_SECS,
)
from time_display import now_hms
from perception.loop import PerceptionLoop
from strategy.agent import StrategistLoop
from strategy.nimble import NimbleAdvice, NimbleLoop

log = logging.getLogger("aoe2_agent.web.session")


@dataclass
class SessionState:
    """Snapshot of everything the frontend needs to render."""

    running: bool = False
    player_civ: str = ""
    opponent_civ: str = ""
    status: str = "idle"                 # "idle" | "analyzing …"
    strategy: str = "Waiting for first analysis…"
    strategy_structured: Optional[dict] = None
    strategy_updated_at: Optional[str] = None
    snapshot: Optional[dict] = None      # serialized GameStateSnapshot
    snapshot_updated_at: Optional[str] = None
    alarms: list[str] = field(default_factory=list)
    nimble_enabled: bool = False
    nimble: Optional[dict] = None        # serialized NimbleAdvice
    nimble_error: Optional[str] = None
    started_at: Optional[str] = None
    error: Optional[str] = None
    # Pipeline: capture -> read screen (VLM) -> Nimble -> wait
    stage: str = "idle"
    stage_since: Optional[float] = None  # epoch seconds, for "running for Ns" in the UI
    stats: dict = field(default_factory=lambda: {
        "frames": 0, "skipped": 0, "snapshots": 0, "vlm_failures": 0,
        "last_capture_secs": None, "last_vlm_secs": None, "last_nimble_ms": None,
    })
    activity: list[dict] = field(default_factory=list)          # newest first
    strategy_history: list[dict] = field(default_factory=list)  # newest first
    timezone_label: str = "IST"

    def to_dict(self) -> dict:
        return {
            "timezone_label": self.timezone_label,
            "stage": self.stage,
            "stage_since": self.stage_since,
            "stats": dict(self.stats),
            "activity": list(self.activity),
            "strategy_history": list(self.strategy_history),
            "running": self.running,
            "player_civ": self.player_civ,
            "opponent_civ": self.opponent_civ,
            "status": self.status,
            "strategy": self.strategy,
            "strategy_structured": self.strategy_structured,
            "strategy_updated_at": self.strategy_updated_at,
            "snapshot": self.snapshot,
            "snapshot_updated_at": self.snapshot_updated_at,
            "alarms": self.alarms,
            "nimble_enabled": self.nimble_enabled,
            "nimble": self.nimble,
            "nimble_error": self.nimble_error,
            "started_at": self.started_at,
            "error": self.error,
        }


MAX_ACTIVITY = 60
MAX_STRATEGY_HISTORY = 10


def launch_game(exe_path: str) -> tuple[bool, str]:
    """Best-effort launch of the game executable.

    On WSL2 a Windows .exe is launched via cmd.exe; on native systems it is run
    directly. Failure is non-fatal — the agent watches whatever is on screen.

    Returns (ok, message).
    """
    if not exe_path:
        return False, "No executable path provided — agent will watch the current screen."
    try:
        if exe_path.lower().endswith(".exe"):
            # WSL2 → launch via cmd.exe so Windows resolves the path
            subprocess.Popen(["cmd.exe", "/c", "start", "", exe_path])
        else:
            subprocess.Popen([exe_path])
        return True, f"Launched: {exe_path}"
    except Exception as exc:
        return False, f"Could not launch '{exe_path}': {exc}"


class SessionManager:
    """Runs the loops in a dedicated background thread + event loop."""

    def __init__(self) -> None:
        self.state = SessionState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._nimble_idle: Optional[asyncio.Event] = None
        self._perception: Optional[PerceptionLoop] = None

    # ------------------------------------------------------------------
    # State mutation helpers (thread-safe)
    # ------------------------------------------------------------------
    def _set_snapshot(self, snap: GameStateSnapshot) -> None:
        with self._lock:
            self.state.snapshot = snap.model_dump()
            self.state.snapshot_updated_at = now_hms()

    def _set_alarms(self, alarms: list[str]) -> None:
        with self._lock:
            self.state.alarms = list(alarms)

    def _set_strategy(self, text: str, structured: dict | None = None) -> None:
        with self._lock:
            self.state.strategy = text            # STICKY — compact markdown fallback
            self.state.strategy_structured = structured
            self.state.strategy_updated_at = now_hms()
            self.state.status = "idle"
            self.state.strategy_history.insert(
                0,
                {"at": self.state.strategy_updated_at, "text": text, "structured": structured},
            )
            del self.state.strategy_history[MAX_STRATEGY_HISTORY:]
        self._log_activity("strategist", "New strategist update")

    def _set_status(self, status: str) -> None:
        with self._lock:
            self.state.status = status
        if status != "idle":
            self._log_activity("strategist", status)

    def _set_nimble(self, advice: NimbleAdvice) -> None:
        with self._lock:
            self.state.nimble = advice.to_dict()
            self.state.nimble_error = None
            self.state.stats["last_nimble_ms"] = round(advice.latency_ms)
        self._log_activity("nimble", f"Advice in {advice.latency_ms / 1000:.1f}s: {advice.priority}")

    def _set_nimble_error(self, message: str) -> None:
        with self._lock:
            self.state.nimble_error = message
        self._log_activity("error", f"Nimble: {message}")

    def _log_activity(self, kind: str, text: str) -> None:
        with self._lock:
            self.state.activity.insert(0, {"at": now_hms(), "kind": kind, "text": text})
            del self.state.activity[MAX_ACTIVITY:]

    def _set_stage(self, stage: str, info: Optional[dict] = None) -> None:
        """Pipeline progress from PerceptionLoop (on_stage) and the Nimble followup."""
        info = info or {}
        with self._lock:
            stats = self.state.stats
            if "capture_secs" in info:
                stats["last_capture_secs"] = round(info["capture_secs"], 2)
            if "vlm_secs" in info:
                stats["last_vlm_secs"] = round(info["vlm_secs"], 1)
            if stage == "reading_screen" or stage == "frame_skipped":
                stats["frames"] += 1
            if stage == "frame_skipped":
                stats["skipped"] += 1
            elif stage == "snapshot":
                stats["snapshots"] += 1
            elif stage == "vlm_failed":
                stats["vlm_failures"] += 1
            # Outcome stages are logged but the pipeline indicator keeps the active step.
            if stage not in ("snapshot", "vlm_failed", "frame_skipped"):
                self.state.stage = stage
                self.state.stage_since = time.time()
        if stage == "snapshot":
            self._log_activity("snapshot", f"Screen read in {info.get('vlm_secs', 0):.1f}s")
        elif stage == "vlm_failed":
            self._log_activity("error", f"Screen reading failed after {info.get('vlm_secs', 0):.1f}s — no snapshot")
        elif stage == "frame_skipped":
            self._log_activity("skip", "Frame unchanged — skipped")

    def get_state(self) -> dict:
        with self._lock:
            return self.state.to_dict()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self, player_civ: str, opponent_civ: str, exe_path: str = "", use_nimble: bool = NIMBLE_ENABLED
    ) -> dict:
        if self.is_running():
            return {"ok": False, "message": "A session is already running. Stop it first."}

        # Reset state
        with self._lock:
            self.state = SessionState(
                running=True,
                player_civ=player_civ,
                opponent_civ=opponent_civ,
                nimble_enabled=use_nimble,
                started_at=now_hms(),
            )

        ok, msg = launch_game(exe_path)
        log.info("Game launch: %s", msg)

        self._thread = threading.Thread(
            target=self._run_thread, args=(player_civ, opponent_civ, use_nimble), daemon=True
        )
        self._thread.start()

        return {"ok": True, "message": msg, "launched": ok}

    def stop(self) -> dict:
        if not self.is_running():
            return {"ok": False, "message": "No session running."}
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
            if self._nimble_idle is not None:
                self._loop.call_soon_threadsafe(self._nimble_idle.set)
        with self._lock:
            self.state.running = False
            self.state.status = "idle"
        return {"ok": True, "message": "Session stopping."}

    # ------------------------------------------------------------------
    # Background thread — owns its own event loop
    # ------------------------------------------------------------------
    def _run_thread(self, player_civ: str, opponent_civ: str, use_nimble: bool) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._stop_event = asyncio.Event()
        try:
            loop.run_until_complete(self._main(player_civ, opponent_civ, use_nimble))
        except Exception as exc:
            log.error("Session crashed: %s", exc, exc_info=True)
            with self._lock:
                self.state.error = str(exc)
        finally:
            with self._lock:
                self.state.running = False
            self._perception = None
            loop.close()

    async def _main(self, player_civ: str, opponent_civ: str, use_nimble: bool) -> None:
        perception = PerceptionLoop(
            on_snapshot=self._set_snapshot,
            on_alarm=self._set_alarms,
            interval=PERCEPTION_INTERVAL_SECS,
            on_stage=self._set_stage,
        )
        self._perception = perception
        # Strategist only starts once Nimble has answered the latest snapshot,
        # so the player's priority always arrives first.
        nimble_idle = asyncio.Event() if use_nimble else None
        self._nimble_idle = nimble_idle
        strategist = StrategistLoop(
            player_civ=player_civ,
            opponent_civ=opponent_civ,
            on_strategy=self._set_strategy,
            on_status=self._set_status,
            interval=STRATEGIST_INTERVAL_SECS,
            perception_loop=perception,
            gate=nimble_idle,
            use_subagents=False,
        )
        assert self._stop_event is not None
        tasks = [perception.run(self._stop_event), strategist.run(self._stop_event)]
        if use_nimble:
            def nimble_started() -> None:
                nimble_idle.clear()
                self._set_stage("nimble")

            def nimble_advice(advice: NimbleAdvice) -> None:
                self._set_nimble(advice)
                nimble_idle.set()

            def nimble_error(message: str) -> None:
                self._set_nimble_error(message)
                nimble_idle.set()

            nimble = NimbleLoop(
                player_civ=player_civ,
                opponent_civ=opponent_civ,
                on_advice=nimble_advice,
                on_error=nimble_error,
                on_escalate=strategist.request_analysis,
                on_start=nimble_started,
                perception_loop=perception,
                interval=NIMBLE_INTERVAL_SECS,
            )
            tasks.append(nimble.run(self._stop_event))
        await asyncio.gather(*tasks)


# Module-level singleton shared across requests
session_manager = SessionManager()
