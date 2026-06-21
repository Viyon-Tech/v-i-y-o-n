"""Lightweight async pub/sub event bus for HUD ↔ CORE state updates.

VIYON CORE publishes agent-state changes (listening, working, done, alerts,
last command) via the module-level :data:`bus`. The HUD's WebSocket handler
subscribes to receive them. Publishing is fire-and-forget and never blocks or
raises — if there are no subscribers (no HUD open), it's a no-op.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("viyon.events")


class EventBus:
    """A tiny fan-out bus: publishers push dict events to all subscriber queues."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        # Last-known snapshot of mutable state, so a freshly-opened HUD gets current values.
        self.snapshot: dict = {
            "agents": {},          # name -> status
            "active_agent": None,
            "last_command": "",
            "listening": False,
            "alert": False,
        }

    def subscribe(self) -> asyncio.Queue:
        """Register and return a new subscriber queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        self._subscribers.discard(q)

    def publish(self, event: dict) -> None:
        """Push ``event`` to every subscriber (dropping on full queues)."""
        self._apply_to_snapshot(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("Subscriber queue full; dropping event.")

    def _apply_to_snapshot(self, event: dict) -> None:
        """Fold an event into the persistent snapshot."""
        etype = event.get("type")
        if etype == "agent":
            name = event.get("agent")
            if name:
                self.snapshot["agents"][name] = event.get("status", "idle")
            if "active_agent" in event:
                self.snapshot["active_agent"] = event["active_agent"]
        elif etype == "command":
            self.snapshot["last_command"] = event.get("last_command", "")
            if "listening" in event:
                self.snapshot["listening"] = event["listening"]
        elif etype == "listening":
            self.snapshot["listening"] = bool(event.get("listening"))
        elif etype == "alert":
            self.snapshot["alert"] = bool(event.get("alert"))
        elif etype == "reset":
            self.snapshot["agents"] = {}
            self.snapshot["active_agent"] = None
            self.snapshot["alert"] = False


# Module-level singleton used across VIYON.
bus = EventBus()


# -- convenience publishers (used by the orchestrator) -----------------------

def emit_listening(on: bool) -> None:
    bus.publish({"type": "listening", "listening": on})


def emit_command(text: str) -> None:
    bus.publish({"type": "command", "last_command": text, "listening": False})


def emit_agent(name: str, status: str, active: bool = False) -> None:
    event = {"type": "agent", "agent": name, "status": status}
    if active:
        event["active_agent"] = name
    bus.publish(event)


def emit_alert(on: bool) -> None:
    bus.publish({"type": "alert", "alert": on})


def emit_reset() -> None:
    bus.publish({"type": "reset"})
