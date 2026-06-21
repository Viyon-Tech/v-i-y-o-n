"""Tests for ECHO, TEMPO, and NEXUS.

ECHO/TEMPO AppleScript is mocked; NEXUS runs pandas for real in a subprocess on a
tiny temp CSV. Contracts: ECHO won't send without approval; TEMPO.plan_day merges
tasks + events; NEXUS.profile returns column stats.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.echo import EchoAgent
from agents.nexus import NexusAgent
from agents.tempo import TempoAgent
from tools import mac_control
from tools.approval import ApprovalGate


def fake_llm(text="ok"):
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))


def gate(answer: str) -> ApprovalGate:
    return ApprovalGate(callback=lambda prompt: answer, config={})


# -- ECHO --------------------------------------------------------------------

async def test_echo_refuses_to_send_without_approval(monkeypatch):
    """Denied approval → no AppleScript send is ever issued."""
    applescript = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(mac_control, "run_applescript", applescript)

    echo = EchoAgent(llm=fake_llm(), tools=None, log=None, approval=gate("no"), config={})
    result = await echo.send_email("bob@example.com", "Lunch?", "Free at noon?")

    assert result.ok is False
    assert "approved" in result.summary.lower() or "aborted" in result.summary.lower()
    applescript.assert_not_called()  # nothing was sent


async def test_echo_sends_when_approved(monkeypatch):
    applescript = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(mac_control, "run_applescript", applescript)

    echo = EchoAgent(llm=fake_llm(), tools=None, log=None, approval=gate("yes"), config={})
    result = await echo.send_email("bob@example.com", "Lunch?", "Free at noon?")

    assert result.ok is True
    assert "sent" in result.summary.lower()
    applescript.assert_called_once()


async def test_echo_imessage_gated(monkeypatch):
    applescript = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(mac_control, "run_applescript", applescript)
    echo = EchoAgent(llm=fake_llm(), tools=None, log=None, approval=gate("no"), config={})
    result = await echo.send_imessage("+15550100", "hi")
    assert result.ok is False
    applescript.assert_not_called()


# -- TEMPO -------------------------------------------------------------------

async def test_tempo_plan_day_merges_tasks_and_events():
    tempo = TempoAgent(llm=fake_llm(), tools=None, log=None, approval=gate("yes"), config={})
    tempo.calendar_events = AsyncMock(
        return_value=[{"title": "Standup", "start": "09:00"},
                      {"title": "1:1 with Sam", "start": "14:00"}]
    )
    tempo.list_tasks = AsyncMock(
        return_value=[{"title": "Email Bob", "due": None}, {"title": "Write spec", "due": None}]
    )

    result = await tempo.plan_day(ctx={})

    assert result.ok is True
    assert "Standup" in result.detail and "1:1 with Sam" in result.detail
    assert "Email Bob" in result.detail and "Write spec" in result.detail
    assert "2 events" in result.summary and "2 tasks" in result.summary


async def test_tempo_plan_day_clear():
    tempo = TempoAgent(llm=fake_llm(), tools=None, log=None, approval=gate("yes"), config={})
    tempo.calendar_events = AsyncMock(return_value=[])
    tempo.list_tasks = AsyncMock(return_value=[])
    result = await tempo.plan_day(ctx={})
    assert result.ok is True
    assert "clear" in result.summary.lower()


# -- NEXUS -------------------------------------------------------------------

async def test_nexus_profile_returns_column_stats(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b,label\n1,2.5,x\n3,4.5,y\n5,6.5,x\n")

    nexus = NexusAgent(llm=fake_llm(), tools=None, log=None, approval=gate("yes"), config={})
    result = await nexus.profile(str(csv))

    assert result.ok is True, result.summary
    assert "3 rows" in result.detail
    # numeric columns report mean; label column reports unique
    assert "a (" in result.detail and "b (" in result.detail and "label" in result.detail
    assert "mean=" in result.detail
    assert "unique=" in result.detail


async def test_nexus_profile_refused_without_approval(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a\n1\n2\n")
    nexus = NexusAgent(llm=fake_llm(), tools=None, log=None, approval=gate("no"), config={})
    result = await nexus.profile(str(csv))
    assert result.ok is False
    assert "approved" in result.summary.lower()


async def test_nexus_dashboard_hands_off_to_nova():
    nexus = NexusAgent(llm=fake_llm(), tools=None, log=None, approval=gate("yes"), config={})
    result = await nexus.run("build a dashboard for sales.csv", ctx={})
    assert result.handoff == "NOVA"
