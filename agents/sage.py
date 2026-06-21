"""SAGE agent (knowledge & chat): explanations, tutoring, and the knowledge base.

SAGE answers and explains as a patient tutor. For questions that need fresh
external facts it requests a handoff to PULSE. For long explanations it offers to
save a note to Notion (via the Notion MCP — stubbed until connected).
"""

from __future__ import annotations

import logging

from agents.base_agent import AgentResult, BaseAgent
from integrations import notion_mcp

logger = logging.getLogger("viyon.sage")

# Cues that the answer depends on current/external information PULSE should fetch.
_FRESH_CUES = (
    "latest", "current", "today", "tonight", "right now", "recent", "recently",
    "news", "price", "stock", "weather", "this week", "this month", "this year",
    "as of", "2024", "2025", "2026", "release date", "who won", "score",
)

# Longer answers get an offer to save to the knowledge base.
_SAVE_THRESHOLD = 400


class SageAgent(BaseAgent):
    """VIYON's knowledge & tutoring agent."""

    name = "SAGE"
    emoji = "💬"
    scope = "Knowledge & chat — explanations, tutoring, the knowledge base."

    def system_prompt(self) -> str:
        return (
            "You are SAGE 💬, VIYON's patient tutor. Explain clearly and adapt the depth "
            "to the user: start simple, then add detail. Use concrete analogies, give "
            "examples, and check understanding. Be encouraging and never condescending."
        )

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Answer/explain; hand off to PULSE for fresh facts; offer to save long notes."""
        if self._needs_fresh_facts(task):
            return AgentResult(
                agent=self.name,
                ok=True,
                summary="That needs current information — I'll hand this to PULSE for fresh sources.",
                detail=task,
                handoff="PULSE",
            )

        answer = (await self.think(task, ctx)) or "I'm not sure how to explain that yet."
        result = self.succeed(self._spoken(answer), detail=answer)

        if len(answer) >= _SAVE_THRESHOLD:
            saved, msg = await notion_mcp.save_note(self._note_title(task), answer, self.config)
            result.summary += f" {msg}" if saved else " I can save this to your notes once Notion is connected."

        return result

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _needs_fresh_facts(task: str) -> bool:
        low = (task or "").lower()
        return any(cue in low for cue in _FRESH_CUES)

    @staticmethod
    def _spoken(answer: str, limit: int = 300) -> str:
        answer = answer.strip()
        if len(answer) <= limit:
            return answer
        return answer[:limit].rsplit(" ", 1)[0] + "… (full explanation in detail)"

    @staticmethod
    def _note_title(task: str) -> str:
        words = task.strip().split()
        return " ".join(words[:8]) + ("…" if len(words) > 8 else "")
