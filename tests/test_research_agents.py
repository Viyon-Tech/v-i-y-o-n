"""Tests for PULSE and SAGE. The web and LLM layers are mocked — no network.

Contracts: PULSE returns cited sources from search results; SAGE answers a plain
definition question with no external calls, and hands off to PULSE when the
question needs fresh facts.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.pulse import PulseAgent
from agents.sage import SageAgent
from tools import web


def fake_llm(text: str):
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))


def make_pulse(llm_text="Vector databases index embeddings for similarity search [1]."):
    return PulseAgent(llm=fake_llm(llm_text), tools=None, log=None, approval=None, config={})


def make_sage(llm_text="Recursion is when a function calls itself."):
    return SageAgent(llm=fake_llm(llm_text), tools=None, log=None, approval=None, config={})


# -- PULSE -------------------------------------------------------------------

async def test_pulse_returns_sources(monkeypatch):
    fake_results = [
        {"title": "Vector DBs 101", "url": "https://example.com/vdb", "description": "intro"},
        {"title": "Embeddings guide", "url": "https://example.org/emb", "description": "guide"},
    ]
    monkeypatch.setattr(web, "search", AsyncMock(return_value=fake_results))
    monkeypatch.setattr(web, "fetch", AsyncMock(return_value=""))  # keep it offline

    pulse = make_pulse()
    result = await pulse.run("research vector databases", ctx={})

    assert result.ok is True
    assert "Sources:" in result.detail
    assert "https://example.com/vdb" in result.detail
    assert "https://example.org/emb" in result.detail
    assert "2 sources" in result.summary


async def test_pulse_fails_gracefully_with_no_results(monkeypatch):
    monkeypatch.setattr(web, "search", AsyncMock(return_value=[]))
    pulse = make_pulse()
    result = await pulse.run("research something obscure", ctx={})
    assert result.ok is False
    assert "couldn't find" in result.summary.lower()


async def test_pulse_dedupes_sources(monkeypatch):
    dup = {"title": "T", "url": "https://dup.example/x", "description": "d"}
    monkeypatch.setattr(web, "search", AsyncMock(return_value=[dup, dict(dup)]))
    monkeypatch.setattr(web, "fetch", AsyncMock(return_value=""))
    pulse = make_pulse()
    result = await pulse.run("look up dup", ctx={})
    assert result.detail.count("https://dup.example/x") == 1


# -- SAGE --------------------------------------------------------------------

async def test_sage_definition_no_external_calls(monkeypatch):
    """A definition question is answered directly — no web, no handoff."""
    # If SAGE ever touched the web, this would explode.
    monkeypatch.setattr(web, "search", AsyncMock(side_effect=AssertionError("no web!")))
    sage = make_sage()

    result = await sage.run("what is recursion in programming", ctx={})

    assert result.ok is True
    assert result.handoff is None
    assert "function calls itself" in result.detail
    assert sage.llm.messages.create.call_count == 1


async def test_sage_hands_off_to_pulse_for_fresh_facts():
    sage = make_sage()
    result = await sage.run("what's the latest news on AI chips today", ctx={})
    assert result.handoff == "PULSE"
    assert sage.llm.messages.create.call_count == 0  # didn't even call the model


async def test_sage_offers_to_save_long_answer():
    long_answer = "Recursion " * 80  # > 400 chars
    sage = make_sage(long_answer)
    result = await sage.run("explain recursion in depth", ctx={})
    assert result.ok is True
    assert "notes" in result.summary.lower() or "notion" in result.summary.lower()


# -- web helper --------------------------------------------------------------

def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><style>.x{}</style></head><body><h1>Hi</h1>" \
           "<script>alert(1)</script><p>World &amp; more</p></body></html>"
    text = web._html_to_text(html)
    assert "Hi" in text and "World" in text
    assert "alert" not in text and "<" not in text
