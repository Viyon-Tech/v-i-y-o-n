"""LUNA agent (emotional support): a warm, non-clinical companion.

LUNA always validates the feeling before offering anything, never diagnoses, and
never minimizes. If the user expresses serious distress or risk of self-harm,
LUNA gently surfaces real support resources and encourages reaching out to a
trusted person or professional — it does not try to "fix" it.

Privacy: when ``ctx['private']`` is set, LUNA logs only "private entry" — never
the content — to the command log. Journals save to a private Notion journal
(stubbed until connected).
"""

from __future__ import annotations

import logging

from agents.base_agent import AgentResult, BaseAgent
from integrations import notion_mcp

logger = logging.getLogger("viyon.luna")

# Cues that warrant gently surfacing crisis resources.
_CRISIS_CUES = (
    "suicide", "suicidal", "kill myself", "end my life", "end it all",
    "don't want to live", "do not want to live", "hurt myself", "harm myself",
    "self-harm", "self harm", "better off dead", "no reason to live", "want to die",
)

# A calm, non-clinical validation opener — deterministic so it ALWAYS comes first.
_VALIDATION = (
    "Thank you for telling me — that sounds genuinely hard, and it makes complete "
    "sense that you'd feel this way."
)

_CRISIS_MESSAGE = (
    "I'm really glad you told me, and I want you to be safe. I'm not able to be the "
    "support you deserve for this on my own. Please reach out to someone who can be "
    "with you right now — a trusted person, or a trained counselor. In the US you can "
    "call or text 988 (Suicide & Crisis Lifeline), any time. If you're outside the US, "
    "your local emergency number or findahelpline.com can connect you. You don't have "
    "to carry this alone."
)


class LunaAgent(BaseAgent):
    """VIYON's emotional-support companion."""

    name = "LUNA"
    emoji = "🌙"
    scope = "Emotional support — supportive conversation, private journaling, check-ins."

    def system_prompt(self) -> str:
        return (
            "You are LUNA 🌙, a warm, calm, non-clinical companion. Your job is to make "
            "the person feel heard.\n"
            "Rules:\n"
            "- ALWAYS validate the feeling first, before offering any thought or suggestion.\n"
            "- Never diagnose, never label, never minimize ('at least…', 'it could be worse').\n"
            "- Be gentle and brief; reflect back what you hear; ask open, caring questions.\n"
            "- You are a companion, not a therapist — for serious distress, encourage "
            "reaching out to a trusted person or professional rather than trying to fix it."
        )

    # -- capabilities ------------------------------------------------------

    async def talk(self, message: str, ctx: dict | None = None) -> AgentResult:
        """Supportive conversation. Validates first; surfaces resources on crisis cues."""
        ctx = ctx or {}

        if self._is_crisis(message):
            response = f"{_VALIDATION}\n\n{_CRISIS_MESSAGE}"
            self._record(message, ctx)
            return AgentResult(
                agent=self.name,
                ok=True,
                summary=response,
                detail="crisis_support_surfaced",
                needs_confirm=False,
            )

        # Validation is deterministic and always precedes the reflective part.
        # If the LLM is unavailable, LUNA still validates — it never hard-fails.
        try:
            reflection = await self.think(
                f"The person said: {message}\nReflect back warmly and ask one gentle, open "
                f"question. Do not give advice or solutions.",
                ctx,
            )
        except Exception as exc:
            logger.warning("LUNA reflection unavailable (%s); validating only.", exc)
            reflection = "I'm here with you. Do you want to tell me more about it?"
        response = _VALIDATION + ("\n\n" + reflection if reflection else "")
        self._record(message, ctx)
        return self.succeed(response, detail=reflection or "")

    async def journal(self, entry: str, ctx: dict | None = None) -> AgentResult:
        """Save a journal entry to the private Notion journal (stubbed)."""
        ctx = ctx or {}
        saved, msg = await notion_mcp.save_note("LUNA Journal", entry, self.config)
        self._record(entry, ctx)
        if saved:
            return self.succeed("Saved that to your private journal. Thank you for trusting me with it.")
        return self.succeed(
            "I've noted that. I can save it to your private Notion journal once it's connected."
        )

    async def weekly_checkin(self, ctx: dict | None = None) -> AgentResult:
        """A gentle weekly check-in prompt."""
        return self.succeed(
            "Hey — just checking in. How has this week felt for you, honestly? "
            "No right answer; I'm here to listen."
        )

    # -- run ---------------------------------------------------------------

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Route to journal / check-in / supportive talk."""
        low = (task or "").lower()
        if "journal" in low or "dear diary" in low or "write down" in low:
            entry = task
            return await self.journal(entry, ctx)
        if "check in" in low or "check-in" in low or "checkin" in low:
            return await self.weekly_checkin(ctx)
        return await self.talk(task, ctx)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _is_crisis(message: str) -> bool:
        low = (message or "").lower()
        return any(cue in low for cue in _CRISIS_CUES)

    def _record(self, content: str, ctx: dict) -> None:
        """Log the session, redacting content when ``ctx['private']`` is set."""
        if self.log is None:
            return
        private = bool((ctx or {}).get("private"))
        payload = "private entry" if private else content
        try:
            self.log.log_command(
                {
                    "raw_input": payload,
                    "agents": ["LUNA"],
                    "result": payload,
                    "status": "ok",
                    "confirmed": True,
                }
            )
        except Exception as exc:  # logging must never break a support conversation
            logger.warning("LUNA could not log session: %s", exc)
