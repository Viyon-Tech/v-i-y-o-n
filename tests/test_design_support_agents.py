"""Tests for VISTA and LUNA.

VISTA returns a usable Canva link even when Canva is absent; LUNA validates
before reflecting, surfaces resources on crisis cues, and honors the privacy flag.
The LLM is mocked; Canva/Notion are unconnected stubs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.luna import LunaAgent, _VALIDATION
from agents.vista import VistaAgent
from logs.logger import VLogger


def fake_llm(text: str):
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))


# -- VISTA -------------------------------------------------------------------

async def test_vista_returns_stub_link_when_canva_absent():
    vista = VistaAgent(llm=fake_llm("ok"), tools=None, log=None, approval=None, config={})
    result = await vista.create_design("poster", "a poster for the hackathon")

    assert result.ok is True
    assert result.artifacts and result.artifacts[0].startswith("https://www.canva.com")
    assert "connect" in result.detail.lower()  # explains how to connect


async def test_vista_run_detects_type():
    vista = VistaAgent(llm=fake_llm("ok"), tools=None, log=None, approval=None, config={})
    result = await vista.run("make a presentation about climate", ctx={})
    assert result.ok is True
    assert "type=presentation" in result.artifacts[0]


# -- LUNA --------------------------------------------------------------------

def make_luna(llm_text="I hear how heavy that feels. What's been weighing on you most?", log=None):
    return LunaAgent(llm=fake_llm(llm_text), tools=None, log=log, approval=None, config={})


async def test_luna_validates_before_advising():
    """The response must lead with validation, before any reflection/advice."""
    advice = "Have you tried making a to-do list to feel more in control?"
    luna = make_luna(advice)

    result = await luna.talk("I'm so overwhelmed with everything right now", ctx={})

    assert result.ok is True
    # Validation is present and comes first.
    assert result.summary.startswith(_VALIDATION)
    assert _VALIDATION in result.summary
    # The reflective/advice text comes only after validation.
    assert result.summary.index(_VALIDATION) < result.summary.index(advice)


async def test_luna_surfaces_resources_on_crisis():
    luna = make_luna()
    result = await luna.talk("honestly I don't want to live anymore", ctx={})
    assert result.summary.startswith(_VALIDATION)  # still validates first
    assert "988" in result.summary
    assert "alone" in result.summary.lower()
    # never tried to "fix" via the model
    assert luna.llm.messages.create.call_count == 0


async def test_luna_respects_privacy_flag(tmp_path):
    vlog = VLogger(base_dir=tmp_path)
    luna = make_luna(log=vlog)

    secret = "I feel awful about a very personal situation I won't repeat"
    await luna.talk(secret, ctx={"private": True})

    rec = vlog.get_recent(1)[0]
    assert rec["result"] == "private entry"
    assert rec["raw_input"] == "private entry"
    assert secret not in (rec["result"] + str(rec["raw_input"]))
    vlog.close()


async def test_luna_logs_content_when_not_private(tmp_path):
    vlog = VLogger(base_dir=tmp_path)
    luna = make_luna(log=vlog)
    await luna.talk("just a normal day, feeling okay", ctx={})
    rec = vlog.get_recent(1)[0]
    assert "normal day" in rec["result"]
    vlog.close()


async def test_luna_journal_stub(tmp_path):
    vlog = VLogger(base_dir=tmp_path)
    luna = make_luna(log=vlog)
    result = await luna.journal("Today I felt proud of finishing the project.", ctx={"private": True})
    assert result.ok is True
    # private flag still redacts the log
    assert vlog.get_recent(1)[0]["result"] == "private entry"
    vlog.close()
