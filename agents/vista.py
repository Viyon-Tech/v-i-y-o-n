"""VISTA agent (design): mockups and graphics via the Canva (and Figma) MCP.

VISTA turns a brief into a Canva design and returns the link as an artifact. When
Canva isn't connected it degrades to a starter-template link and explains how to
connect, so the user always gets something actionable.
"""

from __future__ import annotations

import logging
import re

from agents.base_agent import AgentResult, BaseAgent
from integrations import canva_mcp

logger = logging.getLogger("viyon.vista")

_TYPE_CUES = {
    "poster": "poster",
    "social": "social",
    "instagram": "social",
    "tweet": "social",
    "presentation": "presentation",
    "slides": "presentation",
    "deck": "presentation",
    "mockup": "mockup",
    "wireframe": "mockup",
    "logo": "logo",
    "flyer": "flyer",
    "banner": "banner",
}


class VistaAgent(BaseAgent):
    """VIYON's design agent."""

    name = "VISTA"
    emoji = "🎨"
    scope = "Design — posters, social, presentations, and mockups via Canva/Figma."

    def system_prompt(self) -> str:
        return (
            "You are VISTA 🎨, VIYON's design agent. You turn a brief into polished visual "
            "designs via Canva and Figma. Ask for the essentials (audience, vibe, key text) "
            "only if missing, then produce a clear, on-brief design."
        )

    async def create_design(self, design_type: str, brief: str) -> AgentResult:
        """Create a design of ``design_type`` from ``brief``; return the link as an artifact."""
        result = await canva_mcp.create_design(design_type, brief, self.config)
        link = result["link"]
        if result["connected"]:
            summary = f"Your {design_type} is ready in Canva: {link}"
            detail = f"{result['message']}\nBrief: {brief}"
        else:
            summary = f"Here's a {design_type} starter in Canva: {link}"
            detail = f"{result['message']}\nBrief: {brief}"
        return self.succeed(summary, detail=detail, artifacts=[link])

    # Convenience generators.
    async def generate_poster(self, brief: str) -> AgentResult:
        return await self.create_design("poster", brief)

    async def generate_social(self, brief: str) -> AgentResult:
        return await self.create_design("social", brief)

    async def generate_presentation(self, brief: str) -> AgentResult:
        return await self.create_design("presentation", brief)

    async def generate_mockup(self, brief: str) -> AgentResult:
        return await self.create_design("mockup", brief)

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Detect the design type from the task and create it."""
        design_type = self._detect_type(task)
        brief = self._brief(task)
        return await self.create_design(design_type, brief)

    @staticmethod
    def _detect_type(task: str) -> str:
        low = (task or "").lower()
        for cue, dtype in _TYPE_CUES.items():
            if cue in low:
                return dtype
        return "design"

    @staticmethod
    def _brief(task: str) -> str:
        # Strip a leading "design/make/create a <type>" so the brief is the substance.
        brief = re.sub(
            r"^\s*(?:design|create|make|build|generate)\s+(?:a|an|some)?\s*",
            "",
            task or "",
            flags=re.IGNORECASE,
        )
        return brief.strip() or task or ""
