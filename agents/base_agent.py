"""BaseAgent abstract class and the AgentResult model shared by every VIYON agent.

Every agent subclasses :class:`BaseAgent` and implements ``async run(task, ctx)``.
No agent calls another agent directly — only CORE coordinates. An agent may
*request* a handoff by returning an :class:`AgentResult` with ``handoff`` set;
CORE decides whether to honor it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """The structured result every agent returns.

    Attributes:
        agent: Name of the agent that produced this result (e.g. ``"NOVA"``).
        task: The task the agent was asked to perform.
        ok: True on success, False on failure.
        output: Human-readable result text (used when composing the spoken reply).
        error: Error message when ``ok`` is False.
        handoff: Optional agent name this agent suggests CORE route to next.
        data: Optional structured payload for downstream agents/the HUD.
        duration_ms: How long the agent ran, in milliseconds.
    """

    agent: str
    task: str = ""
    ok: bool = True
    output: str = ""
    error: str | None = None
    handoff: str | None = None
    data: dict = Field(default_factory=dict)
    duration_ms: int | None = None


class BaseAgent(ABC):
    """Abstract base for all VIYON agents.

    Subclasses set the ``name``/``emoji``/``description`` class attributes and
    implement :meth:`run`. The :meth:`ok` and :meth:`fail` helpers build
    correctly-tagged :class:`AgentResult` objects.
    """

    name: str = "BASE"
    emoji: str = ""
    description: str = "Base agent"

    @abstractmethod
    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Perform ``task`` with the given context and return an AgentResult."""
        raise NotImplementedError

    def ok(self, output: str, **kwargs) -> AgentResult:
        """Build a successful AgentResult for this agent."""
        return AgentResult(agent=self.name, ok=True, output=output, **kwargs)

    def fail(self, error: str, **kwargs) -> AgentResult:
        """Build a failed AgentResult for this agent."""
        return AgentResult(agent=self.name, ok=False, error=error, **kwargs)
