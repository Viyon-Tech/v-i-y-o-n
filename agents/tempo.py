"""TEMPO agent (PA & scheduler): tasks, reminders, and day planning.

TEMPO keeps tasks in Notion (via the Notion MCP — stubbed until connected) and
uses macOS Reminders/Calendar via AppleScript. plan_day merges calendar events
with open tasks into a spoken-friendly agenda.
"""

from __future__ import annotations

import logging

from agents.base_agent import AgentResult, BaseAgent
from integrations import notion_mcp
from tools import mac_control

logger = logging.getLogger("viyon.tempo")


def _esc(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace('"', '\\"')


class TempoAgent(BaseAgent):
    """VIYON's personal assistant and scheduler."""

    name = "TEMPO"
    emoji = "📋"
    scope = "PA & scheduler — tasks, reminders, day planning."

    def system_prompt(self) -> str:
        return (
            "You are TEMPO 📋, VIYON's personal assistant and scheduler. You manage tasks, "
            "reminders, and the user's day. Be organized and proactive; propose realistic "
            "schedules and surface conflicts."
        )

    # -- tasks -------------------------------------------------------------

    async def add_task(self, title: str, due: str | None = None) -> AgentResult:
        """Add a task — to Notion if connected, else macOS Reminders."""
        connected = await notion_mcp.is_connected(self.config)
        if connected:
            ok, msg = await notion_mcp.save_note(f"TASK: {title}", due or "", self.config)
            if ok:
                return self.succeed(f"Added task '{title}' to Notion.")
        # Fallback: macOS Reminders.
        script = (
            'tell application "Reminders"\n'
            f'make new reminder with properties {{name:"{_esc(title)}"}}\n'
            "end tell"
        )
        code, _, err = await mac_control.run_applescript(script, approval_required=False)
        if code == 0:
            return self.succeed(f"Added a reminder: {title}.")
        return self.fail(f"Couldn't add the task: {err}")

    async def list_tasks(self) -> list[dict]:
        """Return open tasks (Notion stub + macOS Reminders)."""
        tasks: list[dict] = []
        script = (
            'tell application "Reminders"\n'
            "set output to {}\n"
            "repeat with r in (reminders whose completed is false)\n"
            "set end of output to name of r\n"
            "end repeat\n"
            "return output\n"
            "end tell"
        )
        _, out, _ = await mac_control.run_applescript(script, approval_required=False)
        tasks += [{"title": t.strip(), "due": None} for t in out.split(", ") if t.strip()]
        return tasks

    async def calendar_events(self, range_days: int = 1) -> list[dict]:
        """Return today's calendar events as ``{title, start}``."""
        script = (
            'tell application "Calendar"\n'
            "set output to {}\n"
            "set today to current date\n"
            f"set laterDate to today + ({int(range_days)} * days)\n"
            "repeat with c in calendars\n"
            "set evs to (every event of c whose start date ≥ today and start date ≤ laterDate)\n"
            "repeat with e in evs\n"
            'set end of output to (summary of e) & "||" & (start date of e as string)\n'
            "end repeat\n"
            "end repeat\n"
            "return output\n"
            "end tell"
        )
        _, out, _ = await mac_control.run_applescript(script, approval_required=False)
        events = []
        for chunk in out.split(", "):
            if "||" in chunk:
                title, start = chunk.split("||", 1)
                events.append({"title": title.strip(), "start": start.strip()})
        return events

    async def remind(self, at: str, text: str) -> AgentResult:
        """Create a reminder for a given time."""
        script = (
            'tell application "Reminders"\n'
            f'make new reminder with properties {{name:"{_esc(text)}", remind me date:date "{_esc(at)}"}}\n'
            "end tell"
        )
        code, _, err = await mac_control.run_applescript(script, approval_required=False)
        if code == 0:
            return self.succeed(f"I'll remind you to {text} at {at}.")
        return self.fail(f"Couldn't set the reminder: {err}")

    # -- planning ----------------------------------------------------------

    async def plan_day(self, ctx: dict | None = None) -> AgentResult:
        """Merge calendar events and open tasks into a spoken-friendly agenda."""
        events = await self.calendar_events()
        tasks = await self.list_tasks()

        lines = ["Today's agenda:"]
        if events:
            lines.append("\nEvents:")
            lines += [f"  • {e['start']} — {e['title']}" for e in events]
        if tasks:
            lines.append("\nTasks:")
            lines += [f"  ☐ {t['title']}" for t in tasks]
        if not events and not tasks:
            return self.succeed("Your day is clear — no events or open tasks.")

        agenda = "\n".join(lines)
        spoken = (
            f"You have {len(events)} event{'s' if len(events) != 1 else ''} and "
            f"{len(tasks)} task{'s' if len(tasks) != 1 else ''} today."
        )
        if events:
            spoken += f" First up: {events[0]['title']} at {events[0]['start']}."
        return self.succeed(spoken, detail=agenda)

    # -- run ---------------------------------------------------------------

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Dispatch a PA request to the matching capability."""
        low = (task or "").lower()

        if "plan" in low and "day" in low or "agenda" in low:
            return await self.plan_day(ctx)
        if low.startswith("remind me") or "remind me" in low:
            return await self.add_task(task)
        if "add task" in low or "add a task" in low or low.startswith("todo"):
            title = task.split("task", 1)[-1].strip(" :")
            return await self.add_task(title or task)
        if "tasks" in low or "to-do" in low or "todo" in low:
            tasks = await self.list_tasks()
            if not tasks:
                return self.succeed("You have no open tasks.")
            return self.succeed(
                f"You have {len(tasks)} open tasks.",
                detail="\n".join(f"☐ {t['title']}" for t in tasks),
            )

        # Default: a planning narrative.
        reply = await self.think(task, ctx)
        return self.succeed(reply or "I can plan your day, add tasks, or set reminders.")
