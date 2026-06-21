"""Tests for core.router intent parsing/routing and core.parallel execution.

The Anthropic client is mocked — no network. A fake response mimics the SDK's
``response.content`` list of typed blocks (``block.type`` / ``block.text``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.base_agent import AgentResult, BaseAgent
from core.parallel import run_agents
from core.router import AgentTask, RoutePlan, Router


def fake_client(text: str):
    """A stand-in AsyncAnthropic whose messages.create returns ``text`` as one block."""
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))
    return client


# -- Router: LLM path --------------------------------------------------------

async def test_coding_request_routes_to_nova():
    """A coding request the model maps to NOVA is returned as a NOVA plan."""
    body = (
        '{"agents": [{"name": "NOVA", "task": "write a fib function", "risk": "low"}], '
        '"parallel": false, "reply_style": "short", "needs_confirm": false}'
    )
    router = Router(client=fake_client(body))
    plan = await router.route("write me a fibonacci function", ctx={})

    assert [a.name for a in plan.agents] == ["NOVA"]
    assert plan.agents[0].risk == "low"
    assert plan.parallel is False


async def test_research_and_build_returns_two_parallel_agents():
    """A 'research X and build Y' command yields two agents with parallel=true."""
    body = (
        '{"agents": ['
        '{"name": "PULSE", "task": "research vector databases", "risk": "low"},'
        '{"name": "NOVA", "task": "build a demo using one", "risk": "medium"}'
        '], "parallel": true, "reply_style": "detailed", "needs_confirm": false}'
    )
    router = Router(client=fake_client(body))
    plan = await router.route("research vector databases and build a demo", ctx={})

    assert {a.name for a in plan.agents} == {"PULSE", "NOVA"}
    assert len(plan.agents) == 2
    assert plan.parallel is True


async def test_fenced_json_is_parsed():
    """Code-fenced JSON from the model is still parsed."""
    body = (
        "```json\n"
        '{"agents": [{"name": "GHOST", "task": "check cpu", "risk": "low"}], '
        '"parallel": false, "reply_style": "short", "needs_confirm": false}\n'
        "```"
    )
    router = Router(client=fake_client(body))
    plan = await router.route("how's my cpu doing", ctx={})
    assert [a.name for a in plan.agents] == ["GHOST"]


# -- Router: keyword fallback ------------------------------------------------

async def test_malformed_json_falls_back_to_keyword_routing():
    """When the model returns non-JSON, routing falls back to keyword matching."""
    router = Router(client=fake_client("sorry, I am just chatting, not JSON"))
    plan = await router.route("scan my network for vulnerabilities", ctx={})

    assert any(a.name == "SHIELD" for a in plan.agents)


async def test_malformed_json_with_no_keywords_defaults_to_sage():
    """Malformed output with no keyword hits defaults to SAGE (general chat)."""
    router = Router(client=fake_client("not json"))
    plan = await router.route("hmm, interesting", ctx={})
    assert [a.name for a in plan.agents] == ["SAGE"]


async def test_llm_exception_falls_back_to_keywords():
    """If the client raises, the router still returns a keyword plan."""
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("network down")))
    )
    router = Router(client=client)
    plan = await router.route("explain how transformers work", ctx={})
    assert any(a.name == "SAGE" for a in plan.agents)


async def test_invalid_risk_in_json_falls_back():
    """A schema-violating plan (bad risk enum) falls back rather than crashing."""
    body = '{"agents": [{"name": "NOVA", "task": "x", "risk": "catastrophic"}], "parallel": false}'
    router = Router(client=fake_client(body))
    plan = await router.route("fix this bug in my code", ctx={})
    # Fell back to keywords → NOVA via "bug"/"code".
    assert any(a.name == "NOVA" for a in plan.agents)


# -- Parallel execution ------------------------------------------------------

class _EchoAgent(BaseAgent):
    def __init__(self, name):
        self.name = name

    async def run(self, task, ctx):
        return AgentResult(agent=self.name, ok=True, summary=f"{self.name} did {task}")


class _BoomAgent(BaseAgent):
    name = "SHIELD"

    def __init__(self):
        pass

    async def run(self, task, ctx):
        raise RuntimeError("kaboom")


async def test_parallel_runs_all_agents():
    plan = RoutePlan(
        agents=[AgentTask(name="NOVA", task="a"), AgentTask(name="PULSE", task="b")],
        parallel=True,
    )
    agents = {"NOVA": _EchoAgent("NOVA"), "PULSE": _EchoAgent("PULSE")}
    results = await run_agents(plan, agents, ctx={})

    assert len(results) == 2
    assert all(r.ok for r in results)
    assert {r.agent for r in results} == {"NOVA", "PULSE"}


async def test_sequential_threads_prior_results_into_ctx():
    """Sequential runs pass earlier results forward (SHIELD → NOVA handoff)."""
    seen = {}

    class _ContextAware(BaseAgent):
        name = "NOVA"

        def __init__(self):
            pass

        async def run(self, task, ctx):
            seen["prior"] = ctx.get("prior_results", [])
            return AgentResult(agent=self.name, ok=True, summary="done")

    plan = RoutePlan(
        agents=[AgentTask(name="PULSE", task="research"), AgentTask(name="NOVA", task="build")],
        parallel=False,
    )
    agents = {"PULSE": _EchoAgent("PULSE"), "NOVA": _ContextAware()}
    await run_agents(plan, agents, ctx={})

    assert len(seen["prior"]) == 1
    assert seen["prior"][0].agent == "PULSE"


async def test_agent_exception_is_isolated():
    """One agent crashing becomes a failed result; others still run."""
    plan = RoutePlan(
        agents=[AgentTask(name="SHIELD", task="scan"), AgentTask(name="NOVA", task="code")],
        parallel=True,
    )
    agents = {"SHIELD": _BoomAgent(), "NOVA": _EchoAgent("NOVA")}
    results = await run_agents(plan, agents, ctx={})

    by_name = {r.agent: r for r in results}
    assert by_name["SHIELD"].ok is False
    assert "kaboom" in by_name["SHIELD"].detail
    assert by_name["NOVA"].ok is True


async def test_unknown_agent_becomes_failed_result():
    plan = RoutePlan(agents=[AgentTask(name="MYSTERY", task="x")], parallel=False)
    results = await run_agents(plan, agents={}, ctx={})
    assert results[0].ok is False
    assert "MYSTERY" in results[0].summary
