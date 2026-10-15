"""FastAPI server for the AoE2 Coaching Agent web UI.

Endpoints:
    GET  /                 — the single-page dashboard
    GET  /api/civs         — list of all civilizations (for the dropdowns)
    POST /api/start        — start a coaching session {player_civ, opponent_civ, exe_path}
    POST /api/stop         — stop the running session
    GET  /api/state        — one-shot current state (JSON)
    GET  /api/stream       — Server-Sent Events stream of state updates
    POST /api/nimble/advise — fast-coach advice for a posted snapshot (no session needed)

Run with:
    python -m aoe2_agent web
or
    uvicorn aoe2_agent.web.server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import KB_CIVS, NIMBLE_ENABLED
from schemas import GameStateSnapshot
from strategy.nimble import NimbleAdvisor, NimbleError
from web.session import session_manager

log = logging.getLogger("aoe2_agent.web.server")

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AoE2 Coaching Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    player_civ: str
    opponent_civ: str = "Unknown"
    exe_path: str = ""
    use_nimble: bool = NIMBLE_ENABLED


class NimbleAdviseRequest(BaseModel):
    player_civ: str
    opponent_civ: str = "Unknown"
    snapshot: GameStateSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_civ_names() -> list[str]:
    with open(KB_CIVS, encoding="utf-8") as f:
        civs = json.load(f)
    return sorted(c["name"] for c in civs)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/civs")
def get_civs() -> JSONResponse:
    return JSONResponse({"civs": _load_civ_names()})


@app.post("/api/start")
def start_session(req: StartRequest) -> JSONResponse:
    result = session_manager.start(req.player_civ, req.opponent_civ, req.exe_path, req.use_nimble)
    return JSONResponse(result)


@app.post("/api/nimble/advise")
def nimble_advise(req: NimbleAdviseRequest) -> JSONResponse:
    advisor = NimbleAdvisor(req.player_civ, req.opponent_civ)
    try:
        advice = advisor.advise(req.snapshot)
    except NimbleError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=503)
    return JSONResponse({"ok": True, "advice": advice.to_dict()})


@app.post("/api/stop")
def stop_session() -> JSONResponse:
    return JSONResponse(session_manager.stop())


@app.get("/api/state")
def get_state() -> JSONResponse:
    return JSONResponse(session_manager.get_state())


@app.get("/api/stream")
async def stream():
    """Server-Sent Events — pushes the full state ~1×/second."""
    async def event_generator():
        last_payload = None
        while True:
            state = session_manager.get_state()
            payload = json.dumps(state)
            # Only send when something changed, plus a periodic heartbeat
            if payload != last_payload:
                last_payload = payload
                yield f"data: {payload}\n\n"
            else:
                yield ": heartbeat\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
