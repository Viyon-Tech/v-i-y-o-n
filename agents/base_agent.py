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
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger("viyon.agent")

# Approximate parameter counts (in billions) for local models whose tag has no
# explicit size suffix. Used by the parallel RAM guard.
_KNOWN_LOCAL_SIZES_B = {"phi4": 14.0, "phi3": 4.0, "mistral": 7.0, "gemma2": 9.0}


def _parse_param_b(model: str) -> float | None:
    """Best-effort parameter size in billions from a model tag (e.g. 'qwen2.5:14b' -> 14)."""
    if not model:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model.lower())
    if m:
        return float(m.group(1))
    return _KNOWN_LOCAL_SIZES_B.get(model.lower().split(":")[0])


class ClaudeUnavailable(RuntimeError):
    """Claude can't serve this account (billing/auth/quota) and no local fallback exists.

    Carries a human-readable, speakable message for the orchestrator.
    """


def _classify_claude_failure(exc: Exception) -> str | None:
    """Classify an Anthropic error as an *account availability* problem, or None.

    Returns ``"billing"`` / ``"auth"`` / ``"quota"`` for conditions that mean
    "Claude is unavailable to this account" — the only cases we fall back on.
    Ordinary errors (500s, malformed requests, other 400s) return None and must
    propagate unchanged.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    message = (getattr(exc, "message", None) or str(exc) or "").lower()
    if status == 401 or "authentication" in message or "invalid x-api-key" in message:
        return "auth"
    if status == 429 or "rate limit" in message or "quota" in message:
        return "quota"
    if status == 400 and "credit balance" in message:
        return "billing"
    return None


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
        """Run this agent's persona prompt on its configured brain and return text.

        The provider (claude | ollama) comes from config ``llm.agent_models`` for
        this agent, falling back to ``llm.default_provider``. This only changes
        *where* the model runs — the prompt, context, and returned text are the
        same, so the agent's behavior contract is unchanged.

        - ``tools``/``mcp_servers`` force the Claude path (tool-use is Claude-only).
        - If the local (Ollama) model is unavailable and ``llm.fallback_to_claude``
          is set, transparently fall back to Claude.

        Raises:
            RuntimeError: if the chosen path has no usable model (e.g. Claude
                requested but no client, and no local fallback). The orchestrator
                wraps this into a clean failed AgentResult.
        """
        # Keep the caller's dict (even if empty) so ctx["degraded"] propagates back.
        ctx = {} if ctx is None else ctx
        history = ctx.get("history") or []
        convo = "\n".join(f"{role}: {content}" for role, content in history[-6:])
        user = task if not convo else f"Context:\n{convo}\n\nTask: {task}"

        provider, agent_model = self._provider_for()

        # Tool-use / MCP only works on Claude — force it regardless of config.
        if (tools or mcp_servers) and provider != "claude":
            logger.info(
                "%s: task needs MCP/tools — using Claude (local models can't do tool-use).",
                self.name,
            )
            provider, agent_model = "claude", None

        # -- local (Ollama) path, with fallback to Claude --------------------
        if provider == "ollama":
            from integrations.local_llm import LocalLLMUnavailable, think_local

            model = agent_model or self.model
            try:
                return await think_local(
                    model,
                    self.system_prompt(),
                    [{"role": "user", "content": user}],
                    host=self._conf("llm", "ollama_host", "http://localhost:11434"),
                )
            except LocalLLMUnavailable as exc:
                if not bool(self._conf("llm", "fallback_to_claude", True)):
                    raise
                logger.warning(
                    "%s: local model %s unavailable (%s) — falling back to Claude.",
                    self.name, model, exc,
                )
                agent_model = None  # use the Claude default below

        # -- Claude path -----------------------------------------------------
        if self.llm is None:
            raise RuntimeError(
                "LLM unavailable — set ANTHROPIC_API_KEY and install the anthropic SDK."
            )
        kwargs: dict[str, Any] = {
            "model": agent_model or self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt(),
            "messages": [{"role": "user", "content": user}],
        }
        if tools:
            kwargs["tools"] = tools
        if mcp_servers:
            kwargs["mcp_servers"] = mcp_servers
            kwargs["extra_headers"] = {"anthropic-beta": "mcp-client-2025-04-04"}

        try:
            response = await self.llm.messages.create(**kwargs)
        except Exception as exc:
            reason = _classify_claude_failure(exc)
            if reason is None:
                raise  # ordinary error (500, malformed request) — not an account problem
            return await self._on_claude_unavailable(reason, user, ctx)
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

    async def _on_claude_unavailable(self, reason: str, user: str, ctx: dict) -> str:
        """Reverse fallback: retry on a local model, or raise a clean ClaudeUnavailable.

        Sets ``ctx['degraded'] = True`` when it falls back, so MCP-tool agents
        (NOVA/FORGE) know not to claim they used file tools.
        """
        fb_model = self._local_fallback_model()
        if bool(self._conf("llm", "claude_failure_fallback", True)) and fb_model:
            logger.warning(
                "Claude unavailable (%s) — %s falling back to local %s.",
                reason, self.name, fb_model,
            )
            ctx["degraded"] = True
            from integrations.local_llm import LocalLLMUnavailable, think_local

            try:
                return await think_local(
                    fb_model,
                    self.system_prompt(),
                    [{"role": "user", "content": user}],
                    host=self._conf("llm", "ollama_host", "http://localhost:11434"),
                )
            except LocalLLMUnavailable as lexc:
                raise ClaudeUnavailable(
                    f"{self.name} is unavailable: Claude failed ({reason}) and the local "
                    f"fallback {fb_model} is also unavailable. {self._fix_hint(reason)}"
                ) from lexc
        raise ClaudeUnavailable(self._claude_unavailable_message(reason))

    def _local_fallback_model(self) -> str | None:
        """The local model to use if Claude is unavailable for this agent (or None)."""
        fallbacks = self._conf("llm", "agent_local_fallback", {}) or {}
        return fallbacks.get(self.name)

    def _claude_unavailable_message(self, reason: str) -> str:
        cause = {
            "billing": "Anthropic credits are exhausted",
            "auth": "the Anthropic API key is invalid or unauthorized",
            "quota": "the Anthropic quota or rate limit is exhausted",
        }.get(reason, "Claude is unavailable")
        return f"{self.name} is unavailable: {cause}. {self._fix_hint(reason)}"

    @staticmethod
    def _fix_hint(reason: str) -> str:
        if reason == "billing":
            return "Add credits at console.anthropic.com, or enable a local fallback for this agent."
        if reason == "auth":
            return "Fix ANTHROPIC_API_KEY, or enable a local fallback for this agent."
        return "Try again later, raise your quota, or enable a local fallback for this agent."

    def degraded(self, ctx: dict | None) -> bool:
        """True if ``think`` fell back to a local model on this turn (read from ctx)."""
        return bool((ctx or {}).get("degraded"))

    # -- provider routing --------------------------------------------------

    def _provider_for(self) -> tuple[str, str | None]:
        """Resolve (provider, model) for this agent from config; default to claude."""
        agent_models = self._conf("llm", "agent_models", {}) or {}
        entry = agent_models.get(self.name) or {}
        provider = (entry.get("provider") or self._conf("llm", "default_provider", "claude")).lower()
        return provider, entry.get("model")

    def is_heavy_local(self) -> bool:
        """True if this agent runs a local model larger than ``llm.parallel_local_max_b``."""
        provider, model = self._provider_for()
        if provider != "ollama":
            return False
        size = _parse_param_b(model)
        threshold = float(self._conf("llm", "parallel_local_max_b", 8))
        return size is not None and size > threshold

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
