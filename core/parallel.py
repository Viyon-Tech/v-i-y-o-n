"""Run multiple agents concurrently or in sequence, isolating per-agent failure.

CORE hands a :class:`RoutePlan` and the agent registry here. Independent agents
run together with ``asyncio.gather``; otherwise they run in listed order with
each result threaded into ``ctx['prior_results']`` so a later agent can build on
an earlier one (e.g. SHIELD → NOVA). A crash in one agent is wrapped as a failed
:class:`AgentResult` — it never takes down the whole run.

RAM guard: if a parallel plan would run more than one "heavy" local (Ollama)
model at once, those heavy-local agents are serialized (run one at a time) to
protect memory. Claude agents and small local agents still run in parallel.
"""

from __future__ import annotations

import asyncio
import logging

from agents.base_agent import AgentResult, ClaudeUnavailable
from core.router import AgentTask, RoutePlan

logger = logging.getLogger("viyon.parallel")


def _is_heavy_local(agent) -> bool:
    """True if the agent is configured to run a heavy local model."""
    checker = getattr(agent, "is_heavy_local", None)
    try:
        return bool(checker()) if callable(checker) else False
    except Exception:
        return False


async def run_agents(plan: RoutePlan, agents: dict, ctx: dict) -> list[AgentResult]:
    """Execute the agents named in ``plan`` and return their results in order.

    Args:
        plan: The routing decision (agents, parallel flag).
        agents: Mapping of agent name -> agent instance (the registry).
        ctx: Shared context passed to every agent.
    """
    runnable = [task for task in plan.agents if task.name in agents]

    if plan.parallel and len(runnable) > 1:
        heavy = {task.name for task in runnable if _is_heavy_local(agents.get(task.name))}
        if len(heavy) > 1:
            logger.warning(
                "serialized local agents to protect RAM (%d heavy local models: %s)",
                len(heavy), ", ".join(sorted(heavy)),
            )
            return await _run_mixed(plan.agents, agents, ctx, serialize=heavy)
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


async def _run_mixed(tasks: list[AgentTask], agents: dict, ctx: dict, serialize: set[str]) -> list[AgentResult]:
    """Run light agents in parallel while running ``serialize`` agents one at a time."""
    by_index: dict[int, AgentResult] = {}
    light = [(i, t) for i, t in enumerate(tasks) if t.name in agents and t.name not in serialize]
    heavy = [(i, t) for i, t in enumerate(tasks) if t.name in agents and t.name in serialize]
    missing = [(i, t) for i, t in enumerate(tasks) if t.name not in agents]

    async def run_light() -> None:
        results = await asyncio.gather(
            *(_run_one(t, agents.get(t.name), ctx) for _, t in light)
        )
        for (i, _), r in zip(light, results):
            by_index[i] = r

    async def run_heavy_serial() -> None:
        working_ctx = dict(ctx)
        for i, t in heavy:
            r = await _run_one(t, agents.get(t.name), working_ctx)
            by_index[i] = r
            working_ctx["prior_results"] = [by_index[k] for k in sorted(by_index)]

    async def run_missing() -> None:
        for i, t in missing:
            by_index[i] = await _run_one(t, agents.get(t.name), ctx)

    await asyncio.gather(run_light(), run_heavy_serial(), run_missing())
    return [by_index[i] for i in sorted(by_index)]


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
    except ClaudeUnavailable as exc:
        # Claude is unavailable to this account and there's no local fallback.
        logger.warning("Agent %s: %s", task.name, exc)
        return AgentResult(
            agent=task.name, ok=False, summary=str(exc), detail="claude_unavailable"
        )
    except Exception as exc:
        logger.warning("Agent %s failed: %s", task.name, exc)
        return AgentResult(
            agent=task.name,
            ok=False,
            summary=f"{task.name} hit an error.",
            detail=f"{type(exc).__name__}: {exc}",
        )
