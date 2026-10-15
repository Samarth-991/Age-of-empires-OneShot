"""Console overlay — Rich Live layout for the coaching display.

Three panels (Article VI emergency injection channel):
  - ALARMS    — top, highlighted red, pre-empts everything when active
  - STRATEGY  — Strategist Loop output
  - STATE     — latest GameStateSnapshot summary

The display is purely read-only output; it never captures input focus.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from schemas import GameStateSnapshot


@dataclass
class DisplayState:
    """Shared mutable state read by the renderer, written by loop callbacks."""

    strategy: str = "Waiting for first analysis…"
    snapshot: Optional[GameStateSnapshot] = None
    alarms: list[str] = field(default_factory=list)
    player_civ: str = ""
    last_strategy_ts: Optional[datetime] = None
    last_perception_ts: Optional[datetime] = None


def _render_alarms(state: DisplayState) -> Panel:
    if state.alarms:
        content = "\n".join(f"[bold red]⚠  {a}[/bold red]" for a in state.alarms)
        return Panel(content, title="[bold red]🚨 ALARMS[/bold red]", border_style="red", padding=(0, 1))
    return Panel(
        "[dim]No active alarms[/dim]",
        title="[green]ALARMS[/green]",
        border_style="green",
        padding=(0, 1),
    )


def _render_strategy(state: DisplayState) -> Panel:
    ts = state.last_strategy_ts.strftime("%H:%M:%S") if state.last_strategy_ts else "—"
    return Panel(
        state.strategy,
        title=f"[bold cyan]STRATEGIST[/bold cyan]  [dim]last update {ts}[/dim]",
        border_style="cyan",
        padding=(0, 1),
    )


def _render_state(state: DisplayState) -> Panel:
    snap = state.snapshot
    ts = state.last_perception_ts.strftime("%H:%M:%S") if state.last_perception_ts else "—"

    if snap is None:
        return Panel(
            "[dim]Waiting for first screenshot…[/dim]",
            title=f"[bold yellow]GAME STATE[/bold yellow]  [dim]last seen {ts}[/dim]",
            border_style="yellow",
            padding=(0, 1),
        )

    res = snap.resources
    food = res.get("food", "?")
    wood = res.get("wood", "?")
    gold = res.get("gold", "?")
    stone = res.get("stone", "?")

    res_line = f"[yellow]F[/yellow] {food}  [green]W[/green] {wood}  [gold1]G[/gold1] {gold}  [grey50]S[/grey50] {stone}"
    pop_line = f"Pop: {snap.population}/{snap.pop_cap}"
    age_line = f"Age: [bold]{snap.age}[/bold]"
    threat_color = {"none": "green", "low": "yellow", "high": "red"}.get(snap.threat_level, "white")
    threat_line = f"Threat: [{threat_color}]{snap.threat_level.upper()}[/{threat_color}]"
    idle_line = "[bold red]IDLE VILLAGERS[/bold red]" if snap.idle_villagers_visible else "[dim]No idle villagers[/dim]"

    units_text = (
        "\n".join(
            f"  {'[red]⚔[/red]' if u.owner == 'enemy' else '[blue]🛡[/blue]'} "
            f"{u.approx_count}x {u.label} ({u.owner})"
            for u in snap.visible_units[:8]
        )
        or "  [dim]none detected[/dim]"
    )

    buildings_text = ", ".join(snap.visible_buildings[:6]) or "[dim]none[/dim]"
    notes_text = snap.notes or "[dim]—[/dim]"

    body = (
        f"{age_line}  {pop_line}  {threat_line}\n"
        f"{res_line}\n"
        f"{idle_line}\n\n"
        f"[bold]Units:[/bold]\n{units_text}\n\n"
        f"[bold]Buildings:[/bold] {buildings_text}\n"
        f"[bold]Notes:[/bold] {notes_text}"
    )

    return Panel(
        body,
        title=f"[bold yellow]GAME STATE[/bold yellow]  [dim]last seen {ts}[/dim]",
        border_style="yellow",
        padding=(0, 1),
    )


def _render_header(state: DisplayState) -> Panel:
    civ_txt = f"[bold]{state.player_civ}[/bold]" if state.player_civ else "[dim]Unknown[/dim]"
    return Panel(
        f"[bold white]AoE2 Coaching Agent[/bold white]  —  Playing as {civ_txt}  "
        f"[dim](read-only observer · Ctrl+C to quit)[/dim]",
        border_style="white",
        padding=(0, 1),
    )


class CoachOverlay:
    """Thread-safe Rich Live console display.

    Usage:
        overlay = CoachOverlay(player_civ="Britons")
        overlay.start()
        overlay.set_strategy("## Update…")
        overlay.set_snapshot(snap)
        overlay.set_alarms(["LOW FOOD: 150"])
        overlay.stop()
    """

    def __init__(self, player_civ: str = "", refresh_per_second: int = 4) -> None:
        self._state = DisplayState(player_civ=player_civ)
        self._lock = threading.Lock()
        self._console = Console()
        self._refresh = refresh_per_second
        self._live: Optional[Live] = None

    # ------------------------------------------------------------------
    # Thread-safe state setters (called from async callbacks)
    # ------------------------------------------------------------------

    def set_strategy(self, text: str) -> None:
        with self._lock:
            self._state.strategy = text
            self._state.last_strategy_ts = datetime.now()

    def set_snapshot(self, snap: GameStateSnapshot) -> None:
        with self._lock:
            self._state.snapshot = snap
            self._state.last_perception_ts = datetime.now()

    def set_alarms(self, alarms: list[str]) -> None:
        with self._lock:
            self._state.alarms = alarms

    def clear_alarms(self) -> None:
        with self._lock:
            self._state.alarms = []

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _build_renderable(self):
        with self._lock:
            state = self._state

        layout = Layout()
        layout.split_column(
            Layout(_render_header(state), name="header", size=3),
            Layout(_render_alarms(state), name="alarms", size=5),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(_render_strategy(state), name="strategy", ratio=3),
            Layout(_render_state(state), name="state", ratio=2),
        )
        return layout

    def start(self) -> None:
        """Start the Live display (blocks until stop() is called from another thread)."""
        with Live(
            self._build_renderable(),
            console=self._console,
            refresh_per_second=self._refresh,
            screen=True,
        ) as live:
            self._live = live
            import time
            while self._live is not None:
                live.update(self._build_renderable())
                time.sleep(1 / self._refresh)

    def stop(self) -> None:
        """Signal the display loop to stop."""
        self._live = None
