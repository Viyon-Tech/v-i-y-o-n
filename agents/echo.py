"""ECHO agent (communications): Mail, Calendar, Messages via AppleScript.

ECHO reads mail/calendar (read-only, un-gated) and ALWAYS drafts and reads back
before sending. Sending email, creating events, and sending iMessages are gated
through the Approval Gate — ECHO never auto-sends.
"""

from __future__ import annotations

import logging

from agents.base_agent import AgentResult, BaseAgent
from tools import mac_control

logger = logging.getLogger("viyon.echo")


def _esc(text: str) -> str:
    """Escape a string for an AppleScript double-quoted literal."""
    return (text or "").replace("\\", "\\\\").replace('"', '\\"')


class EchoAgent(BaseAgent):
    """VIYON's communications agent (Mail / Calendar / Messages)."""

    name = "ECHO"
    emoji = "📡"
    scope = "Communications — Mail, Calendar, Messages; always drafts before sending."

    def system_prompt(self) -> str:
        return (
            "You are ECHO 📡, VIYON's communications agent. You handle Mail, Calendar, "
            "and Messages.\n"
            "Rules:\n"
            "- ALWAYS draft a message and read it back to the user before sending.\n"
            "- NEVER send, reply, or create an event without explicit approval.\n"
            "- Keep drafts concise and match the user's tone."
        )

    # -- read-only ---------------------------------------------------------

    async def read_unread(self, n: int = 5) -> list[str]:
        """Return up to ``n`` unread inbox messages as 'sender | subject' lines."""
        script = (
            'tell application "Mail"\n'
            "set output to {}\n"
            "set msgs to (messages of inbox whose read status is false)\n"
            "set k to count of msgs\n"
            f"if k > {int(n)} then set k to {int(n)}\n"
            "repeat with i from 1 to k\n"
            "set m to item i of msgs\n"
            'set end of output to (sender of m) & " | " & (subject of m)\n'
            "end repeat\n"
            "return output\n"
            "end tell"
        )
        _, out, _ = await mac_control.run_applescript(script, approval_required=False)
        return [line.strip() for line in out.split(", ") if line.strip()]

    async def calendar_events(self, range_days: int = 1) -> list[str]:
        """Return event titles/times in the next ``range_days`` days."""
        script = (
            'tell application "Calendar"\n'
            "set output to {}\n"
            "set today to current date\n"
            f"set laterDate to today + ({int(range_days)} * days)\n"
            "repeat with c in calendars\n"
            "set evs to (every event of c whose start date ≥ today and start date ≤ laterDate)\n"
            "repeat with e in evs\n"
            'set end of output to (summary of e) & " @ " & (start date of e as string)\n'
            "end repeat\n"
            "end repeat\n"
            "return output\n"
            "end tell"
        )
        _, out, _ = await mac_control.run_applescript(script, approval_required=False)
        return [line.strip() for line in out.split(", ") if line.strip()]

    def draft_email(self, to: str, subject: str, body: str) -> str:
        """Compose a human-readable draft for read-back (no side effect)."""
        return f"To: {to}\nSubject: {subject}\n\n{body}"

    # -- gated sends -------------------------------------------------------

    async def send_email(self, to: str, subject: str, body: str) -> AgentResult:
        """Send an email — gated at HIGH risk. Drafts first, sends only on approval."""
        draft = self.draft_email(to, subject, body)
        script = (
            'tell application "Mail"\n'
            "set newMessage to make new outgoing message with properties "
            f'{{subject:"{_esc(subject)}", content:"{_esc(body)}", visible:false}}\n'
            "tell newMessage\n"
            "make new to recipient at end of to recipients with properties "
            f'{{address:"{_esc(to)}"}}\n'
            "end tell\n"
            "send newMessage\n"
            "end tell"
        )

        async def _send() -> AgentResult:
            code, _, err = await mac_control.run_applescript(script, approval_required=False)
            if code == 0:
                return self.succeed(f"Sent the email to {to}.", detail=draft)
            return self.fail(f"Mail couldn't send: {err}", detail=draft)

        return await self.guarded(
            "send_email", f"send email to {to} — subject '{subject}'", "high", _send
        )

    async def create_event(self, title: str, start: str, end: str | None = None) -> AgentResult:
        """Create a calendar event — gated."""
        end_clause = f'set end date of newEvent to date "{_esc(end)}"\n' if end else ""
        script = (
            'tell application "Calendar"\n'
            "tell calendar 1\n"
            "set newEvent to make new event with properties "
            f'{{summary:"{_esc(title)}", start date:date "{_esc(start)}"}}\n'
            f"{end_clause}"
            "end tell\n"
            "end tell"
        )

        async def _create() -> AgentResult:
            code, _, err = await mac_control.run_applescript(script, approval_required=False)
            if code == 0:
                return self.succeed(f"Added '{title}' to your calendar at {start}.")
            return self.fail(f"Calendar error: {err}")

        return await self.guarded(
            "create_event", f"create event '{title}' at {start}", "medium", _create
        )

    async def send_imessage(self, to: str, text: str) -> AgentResult:
        """Send an iMessage — gated at HIGH risk."""
        script = (
            'tell application "Messages"\n'
            'set targetService to 1st account whose service type = iMessage\n'
            f'set targetBuddy to participant "{_esc(to)}" of targetService\n'
            f'send "{_esc(text)}" to targetBuddy\n'
            "end tell"
        )

        async def _send() -> AgentResult:
            code, _, err = await mac_control.run_applescript(script, approval_required=False)
            if code == 0:
                return self.succeed(f"Sent your message to {to}.", detail=text)
            return self.fail(f"Messages error: {err}", detail=text)

        return await self.guarded(
            "send_imessage", f"send iMessage to {to}: {text[:60]}", "high", _send
        )

    # -- run ---------------------------------------------------------------

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Interpret a communications request (read-only here; sends need explicit args)."""
        low = (task or "").lower()
        try:
            if "unread" in low or "new email" in low or "check mail" in low or "inbox" in low:
                msgs = await self.read_unread()
                if not msgs:
                    return self.succeed("No unread mail.")
                return self.succeed(
                    f"You have {len(msgs)} unread: " + "; ".join(m.split(' | ')[-1] for m in msgs[:3]),
                    detail="\n".join(msgs),
                )
            if "calendar" in low or "events" in low or "schedule" in low or "what's on" in low:
                events = await self.calendar_events()
                if not events:
                    return self.succeed("Nothing on your calendar in that range.")
                return self.succeed(f"You have {len(events)} events.", detail="\n".join(events))
        except Exception as exc:
            logger.warning("ECHO read failed: %s", exc)
            return self.fail(f"Couldn't reach Mail/Calendar: {exc}")

        # Drafting a send: compose, read back, and require an explicit confirmed send.
        draft = await self.think(
            f"Draft a short message for this request, then I will read it back before "
            f"sending. Request: {task}",
            ctx,
        )
        return self.succeed(
            "Here's a draft — say 'send it' to confirm. I won't send anything on my own.",
            detail=draft,
            needs_confirm=True,
        )
