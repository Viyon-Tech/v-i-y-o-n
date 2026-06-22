"""BaseAgent abstract class and the AgentResult model shared by every VIYON agent.

Every agent subclasses :class:`BaseAgent`, defines its persona in
:meth:`system_prompt`, and implements ``async run(task, ctx)``. No agent calls
another agent directly — only CORE coordinates; an agent may *request* a handoff
by returning an :class:`AgentResult` with ``handoff`` set.

All side effects MUST go through ``self.tools`` and be wrapped in
:meth:`guarded`, which routes destructive/state-changing actions through the
Approval Gate before they run.

The Anthropic SDK is only touched inside :meth:`think` via the injected client,
so this module imports cleanly without the SDK installed.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """The structured result every agent returns.

    Attributes:
        agent: Name of the agent that produced this result (e.g. ``"NOVA"``).
        ok: True on success, False on failure or abort.
        summary: One-line, speakable result (used to compose the voice reply).
        detail: Optional longer detail (full output, error, transcript).
        artifacts: Paths to files the agent produced.
        handoff: Optional agent name this agent suggests CORE route to next.
        needs_confirm: True if the agent wants CORE to confirm a follow-up.
    """

    agent: str
    ok: bool = True
    summary: str = ""
    detail: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    handoff: str | None = None
    needs_confirm: bool = False


class BaseAgent(ABC):
    """Abstract base for all VIYON agents.

    Args:
        llm: An ``anthropic.AsyncAnthropic``-like client used by :meth:`think`.
        tools: The tools registry — the only sanctioned path for side effects.
        log: Structured logger (``logs.logger`` singleton or a VLogger).
        approval: The :class:`~tools.approval.ApprovalGate`.
        config: Settings source — the ``core.config`` module or a nested dict.

    Subclasses override the ``name``/``emoji``/``scope`` class attributes,
    :meth:`system_prompt`, and :meth:`run`.
    """

    name: str = "BASE"
    emoji: str = ""
    scope: str = "Base agent — override in subclasses."

    def __init__(self, llm, tools, log, approval, config) -> None:
        self.llm = llm
        self.tools = tools
        self.log = log
        self.approval = approval
        self.config = config
        self.model = self._conf("llm", "model", "claude-sonnet-4-6")
        self.max_tokens = int(self._conf("llm", "max_tokens", 1024))

    # -- persona -----------------------------------------------------------

    def system_prompt(self) -> str:
        """The agent's persona and rules. Override per agent."""
        return (
            f"You are {self.name} {self.emoji}, a VIYON sub-agent. {self.scope}\n"
            "Be precise and concise. You never act autonomously — every "
            "state-changing action is gated by the user's approval."
        )

    # -- abstract ----------------------------------------------------------

    @abstractmethod
    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Perform ``task`` with the given context and return an AgentResult."""
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------

    async def think(
        self,
        task: str,
        ctx: dict | None = None,
        tools: list | None = None,
        mcp_servers: list | None = None,
    ) -> str:
        """Call Claude with this agent's persona and return the response text.

        Recent session turns from ``ctx['history']`` are folded into the prompt.
        Pass ``tools`` (Anthropic tool definitions) and/or ``mcp_servers`` (remote
        MCP connector configs) to let the model use them. Returns the
        concatenated text of the response's content blocks.

        Raises:
            RuntimeError: if no LLM client is configured (no ANTHROPIC_API_KEY /
                SDK). Callers that can degrade should catch this; the orchestrator
                wraps it into a clean failed AgentResult.
        """
        if self.llm is None:
            raise RuntimeError(
                "LLM unavailable — set ANTHROPIC_API_KEY and install the anthropic SDK."
            )
        ctx = ctx or {}
        history = ctx.get("history") or []
        convo = "\n".join(f"{role}: {content}" for role, content in history[-6:])
        user = task if not convo else f"Context:\n{convo}\n\nTask: {task}"

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt(),
            "messages": [{"role": "user", "content": user}],
        }
        if tools:
            kwargs["tools"] = tools
        if mcp_servers:
            kwargs["mcp_servers"] = mcp_servers
            kwargs["extra_headers"] = {"anthropic-beta": "mcp-client-2025-04-04"}

        response = await self.llm.messages.create(**kwargs)
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

    async def guarded(
        self,
        action: str,
        detail: str,
        risk: str,
        fn: Callable[[], Any] | Awaitable[Any],
    ) -> AgentResult:
        """Run ``fn`` only if the Approval Gate approves; else return an abort.

        Args:
            action: Short action id (e.g. ``"git_push"``).
            detail: Human-readable description of exactly what will happen.
            risk: ``"low" | "medium" | "high"``.
            fn: A zero-arg callable (sync or async) — or an awaitable — that
                performs the action and returns an :class:`AgentResult`.
        """
        approved = await self.approval.request(action, detail, risk)
        if not approved:
            return AgentResult(
                agent=self.name,
                ok=False,
                summary=f"Aborted: {action} was not approved.",
                detail=detail,
            )
        result = fn() if callable(fn) else fn
        if inspect.isawaitable(result):
            result = await result
        return result

    # -- result builders ---------------------------------------------------

    def succeed(self, summary: str, **kwargs) -> AgentResult:
        """Build a successful AgentResult tagged with this agent's name."""
        return AgentResult(agent=self.name, ok=True, summary=summary, **kwargs)

    def fail(self, summary: str, **kwargs) -> AgentResult:
        """Build a failed AgentResult tagged with this agent's name."""
        return AgentResult(agent=self.name, ok=False, summary=summary, **kwargs)

    # -- internal ----------------------------------------------------------

    def _conf(self, section: str, key: str, default: Any) -> Any:
        """Read ``config[section][key]`` whether config is the module or a dict."""
        cfg = self.config
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return (cfg.get(section) or {}).get(key, default)
        getter = getattr(cfg, "get", None)
        if callable(getter):
            try:
                return getter(section, key, default)
            except TypeError:
                return default
        return default
