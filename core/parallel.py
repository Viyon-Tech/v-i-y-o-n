"""Run multiple agents concurrently or in sequence, isolating per-agent failure.

CORE hands a :class:`RoutePlan` and the agent registry here. Independent agents
run together with ``asyncio.gather``; otherwise they run in listed order with
each result threaded into ``ctx['prior_results']`` so a later agent can build on
an earlier one (e.g. SHIELD → NOVA). A crash in one agent is wrapped as a failed
:class:`AgentResult` — it never takes down the whole run.
"""

from __future__ import annotations

import asyncio
import logging

from agents.base_agent import AgentResult
from core.router import AgentTask, RoutePlan

logger = logging.getLogger("viyon.parallel")


async def run_agents(plan: RoutePlan, agents: dict, ctx: dict) -> list[AgentResult]:
    """Execute the agents named in ``plan`` and return their results in order.

    Args:
        plan: The routing decision (agents, parallel flag).
        agents: Mapping of agent name -> agent instance (the registry).
        ctx: Shared context passed to every agent.
    """
    runnable = [task for task in plan.agents if task.name in agents]

    if plan.parallel and len(runnable) > 1:
        results = await asyncio.gather(
            *(_run_one(task, agents.get(task.name), ctx) for task in plan.agents)
        )
        return list(results)

    # Sequential: thread each result forward so later agents can use earlier output.
    results: list[AgentResult] = []
    working_ctx = dict(ctx)
    for task in plan.agents:
        result = await _run_one(task, agents.get(task.name), working_ctx)
        results.append(result)
        working_ctx["prior_results"] = list(results)
    return results


async def _run_one(task: AgentTask, agent, ctx: dict) -> AgentResult:
    """Run a single agent, wrapping a missing agent or exception as a failure."""
    if agent is None:
        logger.warning("No agent registered named %r; skipping.", task.name)
        return AgentResult(
            agent=task.name,
            ok=False,
            summary=f"No agent registered named {task.name!r}",
        )
    try:
        return await agent.run(task.task, ctx)
    except Exception as exc:
        logger.warning("Agent %s failed: %s", task.name, exc)
        return AgentResult(
            agent=task.name,
            ok=False,
            summary=f"{task.name} hit an error.",
            detail=f"{type(exc).__name__}: {exc}",
        )
