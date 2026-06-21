"""Read-only access to config.yaml and .env for VIYON modules.

A tiny shared helper so every module reads settings the same way. Config is
cached after first load; ``.env`` is loaded once (best-effort — missing
python-dotenv is not fatal so development isn't blocked).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config.yaml"
_env_loaded = False


def load_env() -> None:
    """Load ``.env`` into the process environment once (best-effort)."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Return the parsed config.yaml as a dict (empty if absent)."""
    if not _CONFIG_PATH.exists():
        return {}
    import yaml

    return yaml.safe_load(_CONFIG_PATH.read_text()) or {}


def get(section: str, key: str | None = None, default: Any = None) -> Any:
    """Look up ``config[section][key]``, returning ``default`` if missing.

    With ``key=None`` returns the whole section mapping (or ``default``).
    """
    sec = load_config().get(section) or {}
    if key is None:
        return sec or default
    return sec.get(key, default)
