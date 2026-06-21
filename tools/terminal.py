"""Shell/terminal command execution — the single chokepoint for running processes.

Every shell invocation in VIYON goes through :func:`run`. When
``approval_required`` is set, the command must pass the Approval Gate first; with
no gate available it is refused rather than run. Commands are executed via
``create_subprocess_exec`` (no shell), so arguments are never word-split or
glob-expanded by a shell.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("viyon.terminal")

# Conventional return codes for gate outcomes (mirroring shell conventions).
_NOT_APPROVED = 125
_NO_GATE = 126
_NOT_FOUND = 127
_TIMEOUT = 124


async def run(
    cmd: list[str],
    approval_required: bool = True,
    approval=None,
    risk: str = "medium",
    timeout: float = 30.0,
    input_text: str | None = None,
) -> tuple[int, str, str]:
    """Run ``cmd`` and return ``(returncode, stdout, stderr)``.

    Args:
        cmd: Command and arguments (list form — no shell).
        approval_required: If True, the Approval Gate must approve first.
        approval: An :class:`~tools.approval.ApprovalGate` (required when gated).
        risk: Risk label passed to the gate.
        timeout: Seconds before the process is killed.
        input_text: Optional text piped to the process's stdin.
    """
    if not isinstance(cmd, (list, tuple)) or not cmd:
        raise ValueError("cmd must be a non-empty list of arguments")
    cmd = [str(c) for c in cmd]

    if approval_required:
        if approval is None:
            logger.warning("Command requires approval but no gate was provided: %s", cmd)
            return (_NO_GATE, "", "approval required but no approval gate provided")
        if not await approval.request("run_shell", " ".join(cmd), risk):
            return (_NOT_APPROVED, "", "aborted: command not approved")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if input_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        return (_NOT_FOUND, "", str(exc))

    stdin_bytes = input_text.encode() if input_text is not None else None
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin_bytes), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return (_TIMEOUT, "", f"timed out after {timeout}s")

    return (proc.returncode, out.decode(errors="replace"), err.decode(errors="replace"))
