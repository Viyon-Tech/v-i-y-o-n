"""FastAPI + WebSocket server for the VIYON HUD; streams live agent data to the browser.

Serves the static HUD and pushes a JSON telemetry payload over ``/ws`` about
twice a second: system vitals (psutil), time/date/weather, the 12 agents and
their live status (from the CORE event bus), the active agent, last command, and
listening/alert flags. ``POST /command`` runs a text command through the
orchestrator so the HUD is usable without voice.

Run:  python -m hud.server      (or uvicorn "hud.server:app")
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import events
from core.router import AGENT_REGISTRY

logger = logging.getLogger("viyon.hud")

STATIC_DIR = Path(__file__).parent / "static"
_PUSH_INTERVAL = 0.5          # seconds between telemetry pushes (~2 Hz)
_WEATHER_TTL = 600            # seconds (10 min)

# Cached weather: (text, fetched_at).
_weather_cache: tuple[str | None, float] = (None, 0.0)


def _agents_payload() -> list[dict]:
    """Build the 12-agent list with current status from the event-bus snapshot."""
    statuses = events.bus.snapshot.get("agents", {})
    return [
        {"name": name, "emoji": meta["emoji"], "status": statuses.get(name, "idle")}
        for name, meta in AGENT_REGISTRY.items()
    ]


def _vitals() -> dict:
    """One-shot system vitals via psutil (non-blocking cpu sample)."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        battery = psutil.sensors_battery()
        return {
            "cpu": round(psutil.cpu_percent(interval=None), 1),
            "mem": round(mem.percent, 1),
            "disk": round(disk.percent, 1),
            "battery": round(battery.percent) if battery else None,
        }
    except Exception:
        return {"cpu": 0.0, "mem": 0.0, "disk": 0.0, "battery": None}


async def _net_rates(prev: dict) -> tuple[float, float]:
    """Return (up_kbps, down_kbps) since the previous counters in ``prev``."""
    try:
        import psutil

        now = psutil.net_io_counters()
        t = time.monotonic()
        dt = max(t - prev.get("t", t), 1e-3)
        up = (now.bytes_sent - prev.get("sent", now.bytes_sent)) / dt / 1024
        down = (now.bytes_recv - prev.get("recv", now.bytes_recv)) / dt / 1024
        prev.update(sent=now.bytes_sent, recv=now.bytes_recv, t=t)
        return (max(0.0, round(up, 1)), max(0.0, round(down, 1)))
    except Exception:
        return (0.0, 0.0)


async def _weather() -> str | None:
    """Best-effort weather (wttr.in), cached 10 minutes; never blocks fatally."""
    global _weather_cache
    text, fetched = _weather_cache
    if text is not None and (time.monotonic() - fetched) < _WEATHER_TTL:
        return text
    try:
        from tools import web

        raw = await web.fetch("https://wttr.in/?format=%C+%t")
        text = raw.strip()[:40] or None
    except Exception:
        text = None
    _weather_cache = (text, time.monotonic())
    return text


def create_app(core=None) -> FastAPI:
    """Build the HUD FastAPI app, optionally wired to a VIYONCore for /command."""
    app = FastAPI(title="VIYON HUD")
    app.state.core = core

    index_file = STATIC_DIR / "index.html"

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(index_file)

    @app.get("/boot")
    async def boot() -> FileResponse:
        # Same HUD; the boot sequence runs on load. Open this full-screen.
        return FileResponse(index_file)

    @app.post("/command")
    async def command(payload: dict) -> JSONResponse:
        text = (payload or {}).get("text", "").strip()
        if not text:
            return JSONResponse({"ok": False, "error": "empty command"}, status_code=400)
        core_ = app.state.core
        if core_ is None:
            events.emit_command(text)
            return JSONResponse(
                {"ok": True, "reply": "(no orchestrator wired — HUD preview mode)"}
            )
        reply = await core_.handle(text)
        return JSONResponse({"ok": True, "reply": reply})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = events.bus.subscribe()
        net_prev: dict = {}
        tick = 0
        weather = await _weather()
        try:
            while True:
                # Drain any pending CORE events into the snapshot (already applied by the bus).
                while not queue.empty():
                    queue.get_nowait()

                if tick % int(_WEATHER_TTL / _PUSH_INTERVAL) == 0:
                    weather = await _weather()
                up, down = await _net_rates(net_prev)
                now = datetime.now()
                snap = events.bus.snapshot
                payload = {
                    **_vitals(),
                    "net_up": up,
                    "net_down": down,
                    "time": now.strftime("%H:%M:%S"),
                    "date": now.strftime("%a %d %b %Y").upper(),
                    "weather": weather,
                    "agents": _agents_payload(),
                    "active_agent": snap.get("active_agent"),
                    "last_command": snap.get("last_command", ""),
                    "listening": snap.get("listening", False),
                    "alert": snap.get("alert", False),
                }
                await websocket.send_json(payload)
                tick += 1
                await asyncio.sleep(_PUSH_INTERVAL)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("HUD websocket closed: %s", exc)
        finally:
            events.bus.unsubscribe(queue)

    # Mount static assets (after routes so "/" stays the HUD).
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


# Default app instance for `uvicorn hud.server:app`.
app = create_app()


def run(host: str | None = None, port: int | None = None, core=None) -> None:
    """Launch the HUD with uvicorn (reads hud.host/hud.port from config by default)."""
    import uvicorn

    from core import config

    host = host or config.get("hud", "host", "127.0.0.1")
    port = port or int(config.get("hud", "port", 8765))
    global app
    if core is not None:
        app = create_app(core)
    logger.info("VIYON HUD on http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
