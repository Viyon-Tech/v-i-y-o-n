"""ATLAS agent (Mac control): apps, files, windows, clipboard, and system settings.

ATLAS interprets a spoken command and dispatches to the ``mac_control`` and
``file_ops`` tools, passing its Approval Gate so destructive actions (quit,
delete) are confirmed first. It explains what it did in a spoken-friendly
summary. Anything it can't map to a concrete action is answered conversationally
via the LLM.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agents.base_agent import AgentResult, BaseAgent
from tools import file_ops, mac_control

logger = logging.getLogger("viyon.atlas")


class AtlasAgent(BaseAgent):
    """VIYON's Mac control agent."""

    name = "ATLAS"
    emoji = "🖥️"
    scope = "Mac control — apps, files, windows, clipboard, system settings."

    def system_prompt(self) -> str:
        return (
            "You are ATLAS 🖥️, VIYON's Mac control agent. You operate the user's Mac: "
            "opening and quitting apps, managing files, the clipboard, screenshots, and "
            "system settings.\n"
            "Rules:\n"
            "- Explain what you're about to do before acting.\n"
            "- Never delete or overwrite without explicit confirmation.\n"
            "- Prefer reversible actions; when unsure, ask rather than guess.\n"
            "- Route every side effect through your tools — never touch the shell directly."
        )

    def _allowed_roots(self) -> list:
        """Extra filesystem roots ATLAS may touch, from config."""
        return self._conf("filesystem", "allowed_paths", []) or []

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Interpret ``task`` and perform the matching Mac action."""
        text = (task or "").strip()
        low = text.lower()

        try:
            # Open an application.
            m = re.search(r"\bopen (?:the )?(?:app(?:lication)? )?(.+)", low)
            if m and "file" not in low:
                app = self._clean_app(m.group(1))
                code, _, err = await mac_control.open_app(app)
                return self._from_code(code, f"Opened {app}.", f"Couldn't open {app}: {err}")

            # Quit an application (gated inside mac_control).
            m = re.search(r"\b(?:quit|close) (?:the )?(?:app(?:lication)? )?(.+)", low)
            if m:
                app = self._clean_app(m.group(1))
                code, _, err = await mac_control.quit_app(app, approval=self.approval)
                if code in (125, 126):
                    return self.fail(f"Quitting {app} was not approved.")
                return self._from_code(code, f"Quit {app}.", f"Couldn't quit {app}: {err}")

            # Focus / switch to an application.
            m = re.search(r"\b(?:focus|switch to|bring up) (?:the )?(.+)", low)
            if m:
                app = self._clean_app(m.group(1))
                code, _, err = await mac_control.focus_app(app)
                return self._from_code(code, f"Focused {app}.", f"Couldn't focus {app}: {err}")

            # Dark / light mode.
            if "dark mode" in low or "light mode" in low:
                on = "dark" in low and "off" not in low and "light mode" not in low
                await mac_control.set_dark_mode(on)
                return self.succeed(f"Turned {'on' if on else 'off'} Dark Mode.")

            # What's running.
            if re.search(r"(what.*running|list.*apps|running apps)", low):
                apps = await mac_control.list_running_apps()
                return self.succeed(
                    f"{len(apps)} apps running: {', '.join(apps[:6])}"
                    + ("…" if len(apps) > 6 else "."),
                    detail="\n".join(apps),
                )

            # Clipboard.
            if "clipboard" in low and ("read" in low or "what" in low or "get" in low):
                content = await mac_control.clipboard_get()
                return self.succeed("Read the clipboard.", detail=content)
            m = re.search(r"(?:set clipboard to|copy to clipboard)[: ]+(.+)", text, re.IGNORECASE)
            if m:
                await mac_control.clipboard_set(m.group(1).strip())
                return self.succeed("Copied that to the clipboard.")

            # Screenshot.
            if "screenshot" in low or "screen shot" in low:
                target = str(Path.home() / "Desktop" / "viyon-screenshot.png")
                path = await mac_control.screenshot(target)
                return self.succeed("Took a screenshot.", artifacts=[path])

            # File operations — match against the original text so paths keep their case.
            m = re.search(r"\b(?:read|open) (?:the )?file (.+)", text, re.IGNORECASE)
            if m:
                path = self._clean_path(m.group(1))
                content = await file_ops.read_file(path, allowed_roots=self._allowed_roots())
                return self.succeed(f"Read {Path(path).name}.", detail=content)

            m = re.search(
                r"\b(?:list|show)(?: the)? (?:files in|contents of|directory|folder) (.+)",
                text,
                re.IGNORECASE,
            )
            if m:
                path = self._clean_path(m.group(1))
                entries = await file_ops.list_dir(path, allowed_roots=self._allowed_roots())
                return self.succeed(
                    f"{len(entries)} items in {Path(path).name}.", detail="\n".join(entries)
                )

            m = re.search(r"\bdelete (?:the )?(?:file )?(.+)", text, re.IGNORECASE)
            if m:
                path = self._clean_path(m.group(1))
                ok, msg = await file_ops.delete(
                    path, approval=self.approval, allowed_roots=self._allowed_roots()
                )
                return self.succeed(f"Deleted {Path(path).name}.") if ok else self.fail(msg)

        except PermissionError as exc:
            return self.fail(f"That path isn't allowed: {exc}")
        except FileNotFoundError as exc:
            return self.fail(f"Not found: {exc}")
        except Exception as exc:
            logger.warning("ATLAS action failed: %s", exc)
            return self.fail(f"That Mac action failed: {exc}")

        # No concrete action matched — answer conversationally.
        try:
            reply = await self.think(task, ctx)
            return self.succeed(reply or "I'm not sure how to do that on the Mac yet.")
        except Exception:
            return self.fail("I couldn't map that to a Mac action.")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _clean_app(raw: str) -> str:
        """Tidy an extracted application name."""
        name = raw.strip().strip(".!?\"'")
        return name.title() if name.islower() else name

    @staticmethod
    def _clean_path(raw: str) -> str:
        """Tidy an extracted filesystem path."""
        return raw.strip().strip(".!?\"'")

    def _from_code(self, code: int, ok_msg: str, fail_msg: str) -> AgentResult:
        """Build an AgentResult from a terminal return code."""
        return self.succeed(ok_msg) if code == 0 else self.fail(fail_msg)
