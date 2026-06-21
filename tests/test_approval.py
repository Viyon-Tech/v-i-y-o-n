"""Tests for the tools.approval Approval Gate."""

from __future__ import annotations

import pytest

from logs.logger import VLogger
from tools.approval import ApprovalGate


@pytest.fixture()
def logger(tmp_path):
    """A throwaway VLogger writing to a temp dir, closed after the test."""
    vlog = VLogger(base_dir=tmp_path)
    yield vlog
    vlog.close()


def make_gate(logger, answer="no", config=None):
    """Build a gate whose callback always returns ``answer`` and counts calls.

    Returns the gate and a list that records every prompt the callback received,
    so tests can assert whether the user was actually asked.
    """
    asked: list[str] = []

    def callback(prompt: str) -> str:
        asked.append(prompt)
        return answer

    cfg = config if config is not None else {"auto_approve_low": False, "require_confirm_for": []}
    gate = ApprovalGate(callback=callback, config=cfg, logger=logger)
    return gate, asked


async def test_high_risk_always_asks(logger):
    """High risk must prompt the user even when auto-approve-low is on."""
    config = {"auto_approve_low": True, "require_confirm_for": []}
    gate, asked = make_gate(logger, answer="yes", config=config)

    approved = await gate.request("scan", "Run a full vuln scan", risk="high")

    assert approved is True
    assert len(asked) == 1  # the user was asked


async def test_explicit_no_aborts(logger):
    """An explicit 'no' returns False (abort)."""
    gate, asked = make_gate(logger, answer="no")

    approved = await gate.request("delete", "Delete ~/project", risk="high")

    assert approved is False
    assert len(asked) == 1


@pytest.mark.parametrize("answer", ["yes", "y", "go ahead", "do it", "confirm", "  YES  "])
async def test_affirmatives_approve(logger, answer):
    """All recognized affirmatives (case/space-insensitive) approve."""
    gate, _ = make_gate(logger, answer=answer)
    assert await gate.request("run_shell", "ls -la", risk="medium") is True


@pytest.mark.parametrize("answer", ["", "no", "nope", "maybe", "yeah", "cancel"])
async def test_non_affirmatives_abort(logger, answer):
    """Anything that is not an explicit affirmative aborts."""
    gate, _ = make_gate(logger, answer=answer)
    assert await gate.request("run_shell", "rm -rf /tmp/x", risk="medium") is False


async def test_require_confirm_for_overrides_auto_approve(logger):
    """A force-listed action must ask even at low risk with auto-approve on."""
    config = {"auto_approve_low": True, "require_confirm_for": ["delete"]}
    gate, asked = make_gate(logger, answer="no", config=config)

    approved = await gate.request("delete", "Delete a file", risk="low")

    assert approved is False
    assert len(asked) == 1  # was asked despite auto_approve_low + low risk


async def test_auto_approve_low_skips_prompt(logger):
    """Low risk auto-approves silently when enabled and not force-listed."""
    config = {"auto_approve_low": True, "require_confirm_for": ["delete"]}
    gate, asked = make_gate(logger, answer="no", config=config)

    approved = await gate.request("read_file", "Read a config file", risk="low")

    assert approved is True
    assert asked == []  # never prompted


async def test_low_risk_asks_when_auto_approve_disabled(logger):
    """Low risk still asks when auto_approve_low is false."""
    config = {"auto_approve_low": False, "require_confirm_for": []}
    gate, asked = make_gate(logger, answer="yes", config=config)

    approved = await gate.request("read_file", "Read a file", risk="low")

    assert approved is True
    assert len(asked) == 1


async def test_invalid_risk_raises(logger):
    """An unrecognized risk level fails loudly."""
    gate, _ = make_gate(logger)
    with pytest.raises(ValueError):
        await gate.request("delete", "x", risk="critical")


async def test_decision_is_logged(logger):
    """Every decision is recorded via the logger."""
    gate, _ = make_gate(logger, answer="no")
    await gate.request("delete", "Delete ~/project", risk="high")

    recent = logger.get_recent(5)
    assert len(recent) == 1
    assert recent[0]["status"] == "aborted"
    assert recent[0]["confirmed"] is False
    assert recent[0]["result"] == "denied"
