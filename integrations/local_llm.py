"""Local LLM provider via Ollama — lets agents run their "brain" on a local model.

This is a brain-routing layer only: it returns the same plain-text completion that
the Claude path returns, so an agent's behavior contract is unchanged. Tool-use /
MCP are NOT supported here (those stay on the Claude path).

Host comes from config ``llm.ollama_host`` (default ``http://localhost:11434``).
If Ollama is down or the model isn't pulled, raise :class:`LocalLLMUnavailable`
so the caller can fall back to Claude.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("viyon.local_llm")

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class LocalLLMUnavailable(RuntimeError):
    """Raised when Ollama is unreachable or the requested model isn't available."""


def _host(host: str | None = None) -> str:
    """Resolve the Ollama host (explicit arg → config → default)."""
    if host:
        return host
    try:
        from core import config

        return config.get("llm", "ollama_host", DEFAULT_OLLAMA_HOST)
    except Exception:
        return DEFAULT_OLLAMA_HOST


def _client(host: str | None = None):
    """Build an Ollama AsyncClient, raising LocalLLMUnavailable if the SDK is absent."""
    try:
        import ollama
    except ImportError as exc:
        raise LocalLLMUnavailable("the 'ollama' package is not installed") from exc
    return ollama.AsyncClient(host=_host(host))


def _model_names(list_response: Any) -> list[str]:
    """Extract model name tags from an Ollama ``list()`` response (dict or object)."""
    models = getattr(list_response, "models", None)
    if models is None and isinstance(list_response, dict):
        models = list_response.get("models")
    names: list[str] = []
    for m in models or []:
        name = (
            getattr(m, "model", None)
            or getattr(m, "name", None)
            or (m.get("model") if isinstance(m, dict) else None)
            or (m.get("name") if isinstance(m, dict) else None)
        )
        if name:
            names.append(name)
    return names


async def is_ollama_up(host: str | None = None) -> bool:
    """Quick health ping — True if the Ollama server answers."""
    try:
        await _client(host).list()
        return True
    except Exception as exc:
        logger.debug("Ollama health check failed: %s", exc)
        return False


async def model_available(model: str, host: str | None = None) -> bool:
    """True if ``model`` appears in ``ollama list`` (exact tag or base-name match)."""
    try:
        names = _model_names(await _client(host).list())
    except Exception as exc:
        logger.debug("Ollama list failed: %s", exc)
        return False
    base = (model or "").split(":")[0]
    return any(name == model or name.split(":")[0] == base for name in names)


async def think_local(
    model: str,
    system: str,
    messages: list[dict],
    tools: list | None = None,
    host: str | None = None,
) -> str:
    """Run a chat completion on the local Ollama server and return the text.

    Args:
        model: Ollama model tag, e.g. ``"qwen2.5:14b"``.
        system: System prompt (the agent's persona).
        messages: Chat messages (``[{"role": "user", "content": ...}]``).
        tools: Ignored — tool-use is Claude-only; present for signature parity.
        host: Override the configured Ollama host.

    Raises:
        LocalLLMUnavailable: if Ollama is down or the model isn't pulled.
    """
    if tools:
        logger.debug("think_local ignoring tools — tool-use is Claude-only.")
    if not await is_ollama_up(host):
        raise LocalLLMUnavailable(f"Ollama is not reachable at {_host(host)}")
    if not await model_available(model, host):
        raise LocalLLMUnavailable(
            f"Ollama model {model!r} is not pulled — run: ollama pull {model}"
        )

    chat_messages = ([{"role": "system", "content": system}] if system else []) + list(messages)
    try:
        response = await _client(host).chat(model=model, messages=chat_messages)
    except Exception as exc:
        raise LocalLLMUnavailable(f"Ollama chat failed: {exc}") from exc

    # Response is dict-like or an object: {"message": {"role": ..., "content": ...}}.
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return (content or "").strip()
