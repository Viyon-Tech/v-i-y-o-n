"""Client for the Notion MCP server backing TEMPO, SAGE, and LUNA.

Stub for now: :func:`is_connected` reports whether a Notion credential is
configured, and :func:`save_note` no-ops with a clear message when it isn't, so
SAGE/LUNA can offer to save without failing. Wire up the real Notion MCP later.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("viyon.notion_mcp")

SERVER_NAME = "notion"
DEFAULT_MCP_URL = "https://mcp.notion.com/mcp"


def _read(config: Any, section: str, key: str, default: Any) -> Any:
    """Read ``config[section][key]`` from the core.config module or a dict."""
    if config is None:
        try:
            from core import config as _module

            return _module.get(section, key, default)
        except Exception:
            return default
    if isinstance(config, dict):
        return (config.get(section) or {}).get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(section, key, default)
        except TypeError:
            return default
    return default


def server_config(config: Any = None) -> dict:
    """Return the MCP server block for the Anthropic ``mcp_servers`` parameter."""
    return {
        "type": "url",
        "url": _read(config, "notion", "mcp_url", DEFAULT_MCP_URL),
        "name": SERVER_NAME,
    }


async def is_connected(config: Any = None) -> bool:
    """True if a Notion credential is configured (env)."""
    try:
        from core import config as _module

        _module.load_env()
    except Exception:
        pass
    return bool(os.getenv("NOTION_API_KEY") or os.getenv("NOTION_TOKEN"))


async def save_note(title: str, content: str, config: Any = None) -> tuple[bool, str]:
    """Save a note to Notion. Stubbed: no-op with a message when not connected."""
    if not await is_connected(config):
        logger.info("Notion not connected; skipping save of note %r.", title)
        return (False, "Notion isn't connected yet, so I didn't save that note.")
    # TODO: real Notion MCP call (create a page with title/content).
    logger.info("Saved note %r to Notion (stub).", title)
    return (True, f"Saved a note titled '{title}' to Notion.")
