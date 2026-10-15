"""Runs the Nimble advisor on each new perception snapshot."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from config import NIMBLE_INTERVAL_SECS
from schemas import GameStateSnapshot

from .advisor import NimbleAdvice, NimbleAdvisor
from .client import NimbleError

# Child of the "aoe2_agent.strategy" logger, which writes to logs/agent.log.
log = logging.getLogger("strategy.nimble")


class NimbleLoop:
    """Fast coaching loop, run alongside the Perception and Strategist loops.

    When the perception loop supports ``add_followup`` (the real
    ``PerceptionLoop`` does), Nimble runs immediately after every new snapshot
    and the next capture waits for it. Otherwise it falls back to polling
    ``latest_snapshot`` every ``interval`` seconds.

    Args:
        on_advice: Called with each new NimbleAdvice.
        perception_loop: Source of ``latest_snapshot`` / ``snapshot_history``.
        on_error: Called with a message when Nimble is unreachable or fails.
        on_escalate: Called when Nimble starts recommending deep analysis
            (only on the transition, not every cycle).
        on_start: Called just before Nimble starts answering a snapshot.
    """

    def __init__(
        self,
        player_civ: str,
        on_advice: Callable[[NimbleAdvice], None],
        perception_loop,
        opponent_civ: str = "Unknown",
        interval: float = NIMBLE_INTERVAL_SECS,
        on_error: Callable[[str], None] | None = None,
        on_escalate: Callable[[], None] | None = None,
        advisor: NimbleAdvisor | None = None,
        on_start: Callable[[], None] | None = None,
    ) -> None:
        self._advisor = advisor or NimbleAdvisor(player_civ, opponent_civ)
        self._on_advice = on_advice
        self._perception = perception_loop
        self._interval = interval
        self._on_error = on_error
        self._on_escalate = on_escalate
        self._on_start = on_start
        self._last_snapshot_json: str | None = None
        self._was_escalating = False

    async def advise_now(self, snap: GameStateSnapshot) -> None:
        """Answer one snapshot; identical consecutive snapshots are skipped."""
        snapshot_json = snap.model_dump_json()
        if snapshot_json == self._last_snapshot_json:
            return
        history = list(self._perception.snapshot_history)
        if self._on_start is not None:
            self._on_start()
        try:
            advice = await asyncio.to_thread(self._advisor.advise, snap, history)
        except NimbleError as exc:
            log.warning("Nimble unavailable: %s", exc)
            if self._on_error is not None:
                self._on_error(str(exc))
            return
        except Exception as exc:
            log.error("Nimble loop error: %s", exc, exc_info=True)
            if self._on_error is not None:
                self._on_error(f"Nimble loop error: {exc}")
            return

        self._last_snapshot_json = snapshot_json
        log.info(
            "Nimble advice in %.0f ms (%d questions): %s",
            advice.latency_ms, advice.question_count, advice.priority,
        )
        self._on_advice(advice)
        if advice.escalate and not self._was_escalating and self._on_escalate is not None:
            self._on_escalate()
        self._was_escalating = advice.escalate

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        if hasattr(self._perception, "add_followup"):
            log.info("Nimble loop starting (runs right after each snapshot)")
            self._perception.add_followup(self.advise_now)
            if stop_event is not None:
                await stop_event.wait()
            return

        log.info("Nimble loop starting (polling every %.1fs)", self._interval)
        while stop_event is None or not stop_event.is_set():
            snap = self._perception.latest_snapshot
            if snap is not None:
                await self.advise_now(snap)
            await _sleep_unless_stopped(stop_event, self._interval)


async def _sleep_unless_stopped(stop_event: asyncio.Event | None, seconds: float) -> None:
    if stop_event is None:
        await asyncio.sleep(seconds)
        return
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
