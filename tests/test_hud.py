"""Backend smoke tests for the HUD server and the CORE event bus.

The FastAPI app is exercised with TestClient; weather is mocked so no network is
touched. The websocket push is checked for the documented payload shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import hud.server as server
from core import events
from hud.server import _agents_payload, create_app


def test_agents_payload_has_twelve():
    agents = _agents_payload()
    assert len(agents) == 12
    assert all("name" in a and "emoji" in a and "status" in a for a in agents)


def test_root_and_boot_serve_html():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200 and "VIYON" in r.text
    assert client.get("/boot").status_code == 200


def test_static_assets_served():
    client = TestClient(create_app())
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/hud.js").status_code == 200


def test_command_preview_without_core():
    client = TestClient(create_app())  # no orchestrator wired
    ok = client.post("/command", json={"text": "hello viyon"})
    assert ok.status_code == 200 and ok.json()["ok"] is True
    bad = client.post("/command", json={"text": "   "})
    assert bad.status_code == 400


def test_command_routes_to_core():
    class FakeCore:
        def __init__(self):
            self.seen = None

        async def handle(self, text):
            self.seen = text
            return f"handled: {text}"

    core = FakeCore()
    client = TestClient(create_app(core))
    r = client.post("/command", json={"text": "open safari"})
    assert r.json()["reply"] == "handled: open safari"
    assert core.seen == "open safari"


def test_ws_pushes_payload(monkeypatch):
    monkeypatch.setattr(server, "_weather", AsyncMock(return_value="CLEAR +18C"))
    client = TestClient(create_app())
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
    for key in ("cpu", "mem", "disk", "net_up", "net_down", "time", "date",
                "agents", "active_agent", "last_command", "listening", "alert"):
        assert key in data, f"missing {key}"
    assert len(data["agents"]) == 12


def test_event_bus_snapshot_reflects_agent_status():
    events.emit_reset()
    events.emit_agent("NOVA", "working", active=True)
    assert events.bus.snapshot["agents"]["NOVA"] == "working"
    assert events.bus.snapshot["active_agent"] == "NOVA"
    events.emit_alert(True)
    assert events.bus.snapshot["alert"] is True
    events.emit_reset()
    assert events.bus.snapshot["agents"] == {}
