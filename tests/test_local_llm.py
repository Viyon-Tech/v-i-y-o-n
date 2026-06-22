"""Tests for the hybrid Claude/Ollama brain-routing layer.

The Ollama and Anthropic layers are mocked — no server, no network. We verify:
health-ping failure, the Ollama→Claude fallback, MCP forcing the Claude path, and
the parallel RAM guard serializing heavy local models.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.base_agent import AgentResult, BaseAgent
from core.parallel import run_agents
from core.router import AgentTask, RoutePlan
from integrations import local_llm


def fake_llm(text: str):
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))


class _Dummy(BaseAgent):
    name = "X"

    async def run(self, task, ctx):  # pragma: no cover - not exercised here
        return self.succeed("noop")


def make_agent(name, provider="ollama", model="qwen2.5:14b", fallback=True, llm=None):
    config = {
        "llm": {
            "agent_models": {name: {"provider": provider, "model": model}},
            "fallback_to_claude": fallback,
            "model": "claude-sonnet-4-6",
            "max_tokens": 256,
        }
    }
    agent = _Dummy(llm=llm, tools=None, log=None, approval=None, config=config)
    agent.name = name
    return agent


# -- local_llm health --------------------------------------------------------

async def test_is_ollama_up_false_when_unreachable(monkeypatch):
    class Dead:
        async def list(self):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(local_llm, "_client", lambda host=None: Dead())
    assert await local_llm.is_ollama_up() is False


async def test_think_local_raises_when_down(monkeypatch):
    monkeypatch.setattr(local_llm, "is_ollama_up", AsyncMock(return_value=False))
    with pytest.raises(local_llm.LocalLLMUnavailable):
        await local_llm.think_local("qwen2.5:14b", "sys", [{"role": "user", "content": "hi"}])


# -- fallback to Claude ------------------------------------------------------

async def test_ollama_agent_falls_back_to_claude(monkeypatch):
    async def boom(*a, **k):
        raise local_llm.LocalLLMUnavailable("ollama down")

    monkeypatch.setattr(local_llm, "think_local", boom)
    agent = make_agent("SAGE", provider="ollama", fallback=True, llm=fake_llm("Claude answered"))

    out = await agent.think("explain recursion", ctx={})

    assert out == "Claude answered"
    agent.llm.messages.create.assert_called_once()


async def test_no_fallback_reraises(monkeypatch):
    async def boom(*a, **k):
        raise local_llm.LocalLLMUnavailable("ollama down")

    monkeypatch.setattr(local_llm, "think_local", boom)
    agent = make_agent("SAGE", provider="ollama", fallback=False, llm=fake_llm("unused"))

    with pytest.raises(local_llm.LocalLLMUnavailable):
        await agent.think("explain recursion", ctx={})
    agent.llm.messages.create.assert_not_called()


async def test_ollama_path_used_when_available(monkeypatch):
    """When Ollama works, the local path is used and Claude is NOT called."""
    monkeypatch.setattr(local_llm, "think_local", AsyncMock(return_value="local reply"))
    agent = make_agent("SAGE", provider="ollama", llm=fake_llm("claude reply"))

    out = await agent.think("hello", ctx={})

    assert out == "local reply"
    agent.llm.messages.create.assert_not_called()


# -- MCP forces Claude -------------------------------------------------------

async def test_mcp_task_forces_claude_even_if_ollama(monkeypatch):
    """A task with MCP servers must use Claude even if the agent is set to ollama."""
    local_called = {"v": False}

    async def local_spy(*a, **k):
        local_called["v"] = True
        return "should not be used"

    monkeypatch.setattr(local_llm, "think_local", local_spy)
    agent = make_agent("NOVA", provider="ollama", model="qwen2.5:7b", llm=fake_llm("claude+tools"))

    out = await agent.think(
        "read app.py and propose a fix",
        ctx={},
        mcp_servers=[{"type": "url", "url": "http://localhost:3000/mcp", "name": "claude-code"}],
    )

    assert out == "claude+tools"
    assert local_called["v"] is False
    _, kwargs = agent.llm.messages.create.call_args
    assert "mcp_servers" in kwargs


# -- parallel RAM guard ------------------------------------------------------

class _Tracking(BaseAgent):
    """Records peak concurrency so tests can tell parallel from serial."""

    def __init__(self, name, config, shared):
        super().__init__(llm=None, tools=None, log=None, approval=None, config=config)
        self.name = name
        self.shared = shared

    async def run(self, task, ctx):
        self.shared["cur"] += 1
        self.shared["max"] = max(self.shared["max"], self.shared["cur"])
        await asyncio.sleep(0.05)
        self.shared["cur"] -= 1
        return AgentResult(agent=self.name, ok=True, summary="done")


async def test_parallel_serializes_two_heavy_local():
    shared = {"cur": 0, "max": 0}
    config = {"llm": {"agent_models": {
        "SAGE": {"provider": "ollama", "model": "qwen2.5:14b"},
        "PULSE": {"provider": "ollama", "model": "qwen2.5:14b"},
    }}}
    agents = {
        "SAGE": _Tracking("SAGE", config, shared),
        "PULSE": _Tracking("PULSE", config, shared),
    }
    plan = RoutePlan(
        agents=[AgentTask(name="SAGE", task="a"), AgentTask(name="PULSE", task="b")],
        parallel=True,
    )
    results = await run_agents(plan, agents, {})

    assert len(results) == 2 and all(r.ok for r in results)
    assert shared["max"] == 1  # heavy local agents ran one at a time


async def test_parallel_keeps_claude_agents_parallel():
    shared = {"cur": 0, "max": 0}
    config = {"llm": {"agent_models": {
        "VISTA": {"provider": "claude", "model": "claude-sonnet-4-6"},
        "SHIELD": {"provider": "claude", "model": "claude-sonnet-4-6"},
    }}}
    agents = {
        "VISTA": _Tracking("VISTA", config, shared),
        "SHIELD": _Tracking("SHIELD", config, shared),
    }
    plan = RoutePlan(
        agents=[AgentTask(name="VISTA", task="a"), AgentTask(name="SHIELD", task="b")],
        parallel=True,
    )
    results = await run_agents(plan, agents, {})

    assert len(results) == 2
    assert shared["max"] == 2  # claude agents ran concurrently


async def test_small_local_agents_stay_parallel():
    """Two 7B (<=8B) local agents are not 'heavy' → still parallel."""
    shared = {"cur": 0, "max": 0}
    config = {"llm": {"agent_models": {
        "ATLAS": {"provider": "ollama", "model": "qwen2.5:7b"},
        "GHOST": {"provider": "ollama", "model": "qwen2.5:7b"},
    }}}
    agents = {
        "ATLAS": _Tracking("ATLAS", config, shared),
        "GHOST": _Tracking("GHOST", config, shared),
    }
    plan = RoutePlan(
        agents=[AgentTask(name="ATLAS", task="a"), AgentTask(name="GHOST", task="b")],
        parallel=True,
    )
    await run_agents(plan, agents, {})
    assert shared["max"] == 2
