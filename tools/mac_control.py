"""Mac control primitives via AppleScript (osascript) and macOS CLIs.

Thin async wrappers; every one shells out through :func:`tools.terminal.run`.
Read-only / reversible actions run un-gated; ``quit_app`` and arbitrary
``run_applescript`` default to gated since they can change state. Each helper
returns terminal's ``(code, stdout, stderr)`` unless noted.
"""

from __future__ import annotations

from tools import terminal


def _osa_escape(text: str) -> str:
    """Escape a string for safe embedding inside an AppleScript double-quoted literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


async def open_app(name: str, approval=None) -> tuple[int, str, str]:
    """Open (launch or bring up) an application by name. Reversible — un-gated."""
    return await terminal.run(["open", "-a", name], approval_required=False)


async def quit_app(name: str, approval=None) -> tuple[int, str, str]:
    """Quit an application (may discard unsaved work) — gated."""
    return await run_applescript(
        f'tell application "{_osa_escape(name)}" to quit',
        approval=approval,
        approval_required=True,
        risk="medium",
    )


async def focus_app(name: str, approval=None) -> tuple[int, str, str]:
    """Bring an application to the foreground. Reversible — un-gated."""
    return await terminal.run(
        ["osascript", "-e", f'tell application "{_osa_escape(name)}" to activate'],
        approval_required=False,
    )


async def list_running_apps() -> list[str]:
    """Return the names of foreground (non-background) applications."""
    script = (
        'tell application "System Events" to get name of '
        "(every process whose background only is false)"
    )
    _, out, _ = await terminal.run(["osascript", "-e", script], approval_required=False)
    return [name.strip() for name in out.split(",") if name.strip()]


async def run_applescript(
    script: str, approval=None, approval_required: bool = True, risk: str = "medium"
) -> tuple[int, str, str]:
    """Run an arbitrary AppleScript snippet. Gated by default (powerful)."""
    return await terminal.run(
        ["osascript", "-e", script],
        approval_required=approval_required,
        approval=approval,
        risk=risk,
    )


async def set_dark_mode(on: bool, approval=None) -> tuple[int, str, str]:
    """Toggle system Dark Mode. Reversible — un-gated."""
    value = "true" if on else "false"
    script = (
        "tell application \"System Events\" to tell appearance preferences "
        f"to set dark mode to {value}"
    )
    return await terminal.run(["osascript", "-e", script], approval_required=False)


async def screenshot(save_path: str, approval=None) -> str:
    """Capture the screen to ``save_path`` (no shutter sound) and return the path."""
    await terminal.run(["screencapture", "-x", str(save_path)], approval_required=False)
    return str(save_path)


async def clipboard_get() -> str:
    """Return the current clipboard text."""
    _, out, _ = await terminal.run(["pbpaste"], approval_required=False)
    return out


async def clipboard_set(text: str, approval=None) -> bool:
    """Set the clipboard text. Returns True on success."""
    code, _, _ = await terminal.run(["pbcopy"], approval_required=False, input_text=text)
    return code == 0


async def notify(title: str, body: str, approval=None) -> tuple[int, str, str]:
    """Post a macOS notification."""
    script = (
        f'display notification "{_osa_escape(body)}" '
        f'with title "{_osa_escape(title)}"'
    )
    return await terminal.run(["osascript", "-e", script], approval_required=False)
