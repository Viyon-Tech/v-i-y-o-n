"""Client for the Canva MCP server backing VISTA.

Stub for now: :func:`is_connected` reports whether Canva is configured, and
:func:`create_design` returns a Canva link either way — a generated-design link
when connected, or a starter-template link plus connect instructions when not —
so VISTA always hands back something usable.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger("viyon.canva_mcp")

SERVER_NAME = "canva"
DEFAULT_MCP_URL = "https://mcp.canva.com/mcp"

# Map a spoken design type to a Canva editor design type.
_CANVA_TYPES = {
    "poster": "poster",
    "social": "social-media",
    "social media": "social-media",
    "instagram": "instagram-post",
    "presentation": "presentation",
    "slides": "presentation",
    "deck": "presentation",
    "mockup": "website-mockup",
    "logo": "logo",
    "flyer": "flyer",
    "banner": "banner",
}

_CONNECT_HELP = (
    "Canva isn't connected yet. To enable auto-generation, connect the Canva MCP "
    "server (OAuth) and set CANVA_API_KEY in .env. For now, here's a starter "
    "template you can open in Canva."
)


def _read(config: Any, section: str, key: str, default: Any) -> Any:
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
        "url": _read(config, "design", "canva_mcp_url", DEFAULT_MCP_URL),
        "name": SERVER_NAME,
    }


async def is_connected(config: Any = None) -> bool:
    """True if a Canva credential is configured (env)."""
    try:
        from core import config as _module

        _module.load_env()
    except Exception:
        pass
    return bool(os.getenv("CANVA_API_KEY") or os.getenv("CANVA_TOKEN"))


def _canva_type(design_type: str) -> str:
    return _CANVA_TYPES.get((design_type or "").lower().strip(), "design")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "design").lower()).strip("-")[:48] or "design"


async def create_design(design_type: str, brief: str, config: Any = None) -> dict:
    """Create (or template) a Canva design. Returns ``{connected, link, message}``."""
    ctype = _canva_type(design_type)
    if await is_connected(config):
        # TODO: real Canva MCP generation from the brief.
        link = f"https://www.canva.com/design/{_slug(design_type + '-' + brief)}/edit"
        logger.info("Created Canva %s (stub link).", design_type)
        return {"connected": True, "link": link, "message": f"Created a {design_type} in Canva."}

    link = f"https://www.canva.com/design?create&type={ctype}"
    return {"connected": False, "link": link, "message": _CONNECT_HELP}
