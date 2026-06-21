"""Client for the claude-code MCP server backing NOVA and FORGE.

The server (codeaashu/claude-code, ``mcp-server/``) exposes code tools —
read_file, str_replace, bash, search_code, git — over HTTP. We attach it to
Anthropic API calls via the ``mcp_servers`` connector. When it isn't running,
NOVA/FORGE degrade to plain Claude code generation (no file tools).

Setup (per docs/CLAUDE.md):
    1. Clone codeaashu/claude-code into ~/viyon/claude-code
    2. cd ~/viyon/claude-code/mcp-server && npm install && npm run build
    3. node ~/viyon/claude-code/mcp-server/dist/index.js   (or call start_server)
The server then listens at config ``coding.mcp_url`` (default localhost:3000/mcp).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("viyon.claude_code_mcp")

DEFAULT_MCP_URL = "http://localhost:3000/mcp"
DEFAULT_SERVER_SCRIPT = "~/viyon/claude-code/mcp-server/dist/index.js"
SERVER_NAME = "claude-code-explorer"


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
        "url": _read(config, "coding", "mcp_url", DEFAULT_MCP_URL),
        "name": SERVER_NAME,
    }


async def health_check(config: Any = None, timeout: float = 2.0) -> bool:
    """Return True if the MCP server's host:port accepts a TCP connection."""
    url = _read(config, "coding", "mcp_url", DEFAULT_MCP_URL)
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


async def start_server(config: Any = None) -> tuple[bool, str]:
    """Launch the local claude-code MCP server, or explain why it can't start.

    Returns ``(running, message)``. Does not block on the server process; it
    spawns it detached and then health-checks.
    """
    script = Path(
        _read(config, "coding", "mcp_server_script", DEFAULT_SERVER_SCRIPT)
    ).expanduser()

    if await health_check(config):
        return (True, "claude-code MCP server is already running.")
    if shutil.which("node") is None:
        return (False, "Node.js not found — install Node to run the claude-code MCP server.")
    if not script.exists():
        return (
            False,
            f"MCP server not built at {script}. Clone codeaashu/claude-code into "
            "~/viyon/claude-code and run `npm install && npm run build` in mcp-server/.",
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            "node",
            str(script),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except Exception as exc:
        return (False, f"Failed to launch MCP server: {exc}")

    await asyncio.sleep(0.6)
    if await health_check(config):
        return (True, f"Launched claude-code MCP server (pid {proc.pid}).")
    return (False, "Started node but the MCP server isn't responding yet — check its logs.")


async def is_available(config: Any = None) -> bool:
    """Alias for :func:`health_check` — is the coding engine reachable?"""
    return await health_check(config)
