"""Intent parsing and routing of commands to one or more named VIYON agents.

The router asks Claude to turn a transcript into a strict-JSON :class:`RoutePlan`
naming which agents to run, with what task, at what risk. If the model is
unavailable or returns malformed JSON, it falls back to keyword matching against
:data:`AGENT_REGISTRY`, defaulting to SAGE for general chat.

The Anthropic import is deferred so this module loads without the SDK installed;
tests inject a mock client.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from core import config

logger = logging.getLogger("viyon.router")

# name -> (one-line capability, trigger keywords). Matches the 12 agents in CLAUDE.md.
AGENT_REGISTRY: dict[str, dict] = {
    "NOVA": {
        "emoji": "💻",
        "description": "Coding — writes, edits, reviews, and debugs code; runs git.",
        "keywords": ["code", "coding", "function", "debug", "refactor", "bug",
                     "git", "program", "script", "implement", "compile"],
    },
    "FORGE": {
        "emoji": "🏗️",
        "description": "Project scaffolding — generates full project structures and opens them.",
        "keywords": ["scaffold", "new project", "boilerplate", "set up a project",
                     "create a project", "starter", "skeleton"],
    },
    "SHIELD": {
        "emoji": "🛡️",
        "description": "Cybersecurity — vulnerability scans, dependency audits, network analysis.",
        "keywords": ["security", "vulnerability", "vuln", "scan", "audit",
                     "exploit", "penetration", "pentest", "network analysis"],
    },
    "PULSE": {
        "emoji": "🔬",
        "description": "Research — web search and paper/URL summarization.",
        "keywords": ["research", "search", "look up", "find information",
                     "summarize", "paper", "article", "web search"],
    },
    "ATLAS": {
        "emoji": "🖥️",
        "description": "Mac control — apps, files, windows, clipboard, system settings.",
        "keywords": ["open app", "open ", "file", "window", "clipboard",
                     "system settings", "finder", "screenshot", "launch"],
    },
    "ECHO": {
        "emoji": "📡",
        "description": "Communications — Mail, Calendar, Messages; drafts before sending.",
        "keywords": ["email", "mail", "calendar", "message", "imessage",
                     "send", "schedule a meeting", "draft", "reply to"],
    },
    "NEXUS": {
        "emoji": "📊",
        "description": "Data & analytics — CSV/Excel/JSON, charts, reports.",
        "keywords": ["csv", "excel", "spreadsheet", "chart", "graph",
                     "report", "analyze data", "dataset", "pivot"],
    },
    "VISTA": {
        "emoji": "🎨",
        "description": "Design — mockups via Canva and Figma.",
        "keywords": ["design", "mockup", "canva", "figma", "logo",
                     "poster", "ui ", "wireframe", "banner"],
    },
    "TEMPO": {
        "emoji": "📋",
        "description": "PA & scheduler — tasks, reminders, planning (Notion).",
        "keywords": ["remind", "reminder", "task", "todo", "to-do",
                     "plan my", "agenda", "schedule my", "notion"],
    },
    "SAGE": {
        "emoji": "💬",
        "description": "Knowledge & chat — explanations, tutoring, the knowledge base.",
        "keywords": ["explain", "what is", "how does", "teach", "tutor",
                     "define", "tell me about", "why is"],
    },
    "LUNA": {
        "emoji": "🌙",
        "description": "Emotional support — wellness and journaling; warm, validates first.",
        "keywords": ["feeling", "stressed", "anxious", "journal", "wellness",
                     "vent", "sad", "overwhelmed", "lonely"],
    },
    "GHOST": {
        "emoji": "👁️",
        "description": "System monitor — CPU, memory, disk, processes.",
        "keywords": ["cpu", "memory", "ram", "disk", "process",
                     "monitor", "performance", "system usage", "temperature"],
    },
}


class AgentTask(BaseModel):
    """A single agent assignment within a RoutePlan."""

    name: str
    task: str
    risk: Literal["low", "medium", "high"] = "medium"


class RoutePlan(BaseModel):
    """The structured routing decision for one command."""

    agents: list[AgentTask] = Field(default_factory=list)
    parallel: bool = False
    reply_style: Literal["short", "detailed"] = "short"
    needs_confirm: bool = False


def _extract_json(text: str) -> str | None:
    """Strip ``` fences and isolate the outermost JSON object, or None."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return t[start : end + 1]


class Router:
    """Routes transcripts to agents via Claude, with a keyword fallback.

    Args:
        client: An ``anthropic.AsyncAnthropic``-like client (injected in tests).
            Built lazily from the environment when omitted.
        model: Anthropic model id. Defaults to config ``llm.model``.
        max_tokens: Max tokens for the routing call. Defaults to config
            ``llm.max_tokens``.
    """

    def __init__(self, client=None, model: str | None = None, max_tokens: int | None = None) -> None:
        config.load_env()
        self._client = client
        self.model = model or config.get("llm", "model", "claude-sonnet-4-6")
        self.max_tokens = max_tokens or int(config.get("llm", "max_tokens", 1024))

    @property
    def client(self):
        """Lazily build an AsyncAnthropic client if one wasn't injected."""
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def route(self, transcript: str, ctx: dict) -> RoutePlan:
        """Return a RoutePlan for ``transcript`` (LLM first, keyword fallback)."""
        try:
            raw = await self._call_llm(transcript, ctx)
            plan = self._parse(raw)
            if plan is not None and plan.agents:
                return plan
            logger.warning("Router LLM returned an unusable plan; using keyword fallback.")
        except Exception as exc:
            logger.warning("Router LLM call failed (%s); using keyword fallback.", exc)
        return self._keyword_fallback(transcript)

    async def _call_llm(self, transcript: str, ctx: dict) -> str:
        """Call Claude and return the concatenated text of its response."""
        history = ctx.get("history") or []
        convo = "\n".join(f"{role}: {content}" for role, content in history[-6:])
        user = transcript if not convo else (
            f"Recent conversation:\n{convo}\n\nNew command: {transcript}"
        )
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    @staticmethod
    def _system_prompt() -> str:
        """Build the routing system prompt enumerating every agent."""
        roster = "\n".join(
            f"- {name} {meta['emoji']}: {meta['description']}"
            for name, meta in AGENT_REGISTRY.items()
        )
        return (
            "You are VIYON CORE's intent router. Given a user command, decide which "
            "agent(s) should handle it and return STRICT JSON only — no prose, no code "
            "fences.\n\n"
            f"Agents:\n{roster}\n\n"
            "Return exactly this shape:\n"
            '{"agents": [{"name": "NOVA", "task": "<what this agent should do>", '
            '"risk": "low|medium|high"}], "parallel": true|false, '
            '"reply_style": "short|detailed", "needs_confirm": true|false}\n\n'
            "Rules:\n"
            "- Use only agent names from the list above.\n"
            "- Set parallel=true only when agents are independent and can run at once.\n"
            "- risk=high for destructive/state-changing work (delete, push, send, run shell).\n"
            "- needs_confirm=true whenever any agent will change state or send something.\n"
            "- For general chat, questions, or explanations, route to SAGE.\n"
            "- Keep each task short and imperative."
        )

    def _parse(self, raw: str) -> RoutePlan | None:
        """Parse model output into a RoutePlan, or None if malformed."""
        snippet = _extract_json(raw or "")
        if not snippet:
            return None
        try:
            data = json.loads(snippet)
            return RoutePlan(**data)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return None

    def _keyword_fallback(self, transcript: str) -> RoutePlan:
        """Route by keyword matching; default to SAGE for general chat."""
        lowered = transcript.lower()
        matched = [
            name
            for name, meta in AGENT_REGISTRY.items()
            if any(keyword in lowered for keyword in meta["keywords"])
        ]
        if not matched:
            matched = ["SAGE"]
        agents = [AgentTask(name=name, task=transcript, risk="medium") for name in matched]
        return RoutePlan(
            agents=agents,
            parallel=len(agents) > 1,
            reply_style="short",
            needs_confirm=False,
        )
