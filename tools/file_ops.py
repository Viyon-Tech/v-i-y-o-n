"""File operations, path-confined to the user's home (plus configured roots).

Read helpers (`list_dir`, `read_file`, `find`) run un-gated; mutating helpers
(`write_file`, `move`, `copy`, `delete`) go through the Approval Gate — ``delete``
at high risk. Every path is validated against the allowed roots: the user's home,
anything in config ``filesystem.allowed_paths``, plus per-call ``allowed_roots``
(used by tests). Paths outside raise :class:`PermissionError`.

Gated helpers return ``(ok: bool, message: str)``; reads return their data.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("viyon.file_ops")


def _config_allowed_roots() -> list[Path]:
    """Allowed roots declared in config ``filesystem.allowed_paths``."""
    try:
        from core import config

        paths = config.get("filesystem", "allowed_paths", []) or []
    except Exception:
        paths = []
    return [Path(p).expanduser().resolve() for p in paths]


def _validate(path, allowed_roots: list | None = None) -> Path:
    """Resolve ``path`` and ensure it sits within an allowed root."""
    resolved = Path(path).expanduser().resolve()
    roots = [Path.home().resolve()]
    roots += [Path(r).expanduser().resolve() for r in (allowed_roots or [])]
    roots += _config_allowed_roots()
    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved
    raise PermissionError(f"Path {resolved} is outside the allowed roots")


async def _gate(approval, action: str, detail: str, risk: str) -> bool:
    """Return True only if the Approval Gate approves (False when no gate)."""
    if approval is None:
        logger.warning("%s requires approval but no gate was provided.", action)
        return False
    return await approval.request(action, detail, risk)


# -- reads -------------------------------------------------------------------

async def list_dir(path, allowed_roots=None) -> list[str]:
    """List entry names in a directory (sorted)."""
    p = _validate(path, allowed_roots)
    return sorted(entry.name for entry in p.iterdir())


async def read_file(path, allowed_roots=None) -> str:
    """Read and return a text file's contents."""
    p = _validate(path, allowed_roots)
    return p.read_text()


async def find(pattern: str, root=None, allowed_roots=None) -> list[str]:
    """Recursively glob ``pattern`` under ``root`` (defaults to home)."""
    base = _validate(root or Path.home(), allowed_roots)
    return [str(match) for match in base.rglob(pattern)]


async def make_dir(path, allowed_roots=None) -> tuple[bool, str]:
    """Create a directory (and parents); idempotent."""
    p = _validate(path, allowed_roots)
    p.mkdir(parents=True, exist_ok=True)
    return (True, f"created {p}")


# -- mutations (gated) -------------------------------------------------------

async def write_file(path, content: str, approval=None, allowed_roots=None,
                     risk: str = "medium", require_approval: bool = True) -> tuple[bool, str]:
    """Write text to a file (creating parents). Gated unless ``require_approval`` is False.

    ``require_approval=False`` lets a caller that already obtained one batch
    approval (e.g. FORGE scaffolding many files) write without re-prompting.
    """
    p = _validate(path, allowed_roots)
    if require_approval and not await _gate(
        approval, "write_file", f"write {len(content)} chars to {p}", risk
    ):
        return (False, "aborted: write not approved")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return (True, f"wrote {p}")


async def move(src, dst, approval=None, allowed_roots=None) -> tuple[bool, str]:
    """Move/rename a file or directory. Gated."""
    s = _validate(src, allowed_roots)
    d = _validate(dst, allowed_roots)
    if not await _gate(approval, "move", f"move {s} -> {d}", "medium"):
        return (False, "aborted: move not approved")
    shutil.move(str(s), str(d))
    return (True, f"moved {s} to {d}")


async def copy(src, dst, approval=None, allowed_roots=None) -> tuple[bool, str]:
    """Copy a file or directory tree. Gated."""
    s = _validate(src, allowed_roots)
    d = _validate(dst, allowed_roots)
    if not await _gate(approval, "copy", f"copy {s} -> {d}", "medium"):
        return (False, "aborted: copy not approved")
    if s.is_dir():
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)
    return (True, f"copied {s} to {d}")


async def delete(path, approval=None, allowed_roots=None,
                 require_approval: bool = True) -> tuple[bool, str]:
    """Delete a file or directory tree. Gated at HIGH risk."""
    p = _validate(path, allowed_roots)
    if require_approval and not await _gate(approval, "delete", f"delete {p}", "high"):
        return (False, "aborted: delete not approved")
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)
    return (True, f"deleted {p}")
