"""PULSE agent (research): web search and paper/URL summarization.

PULSE plans queries, searches the web, fetches the top sources for context, and
synthesizes a structured summary that cites its sources. It returns a short
spoken summary plus full detail (findings + a source list).
"""

from __future__ import annotations

import logging

from agents.base_agent import AgentResult, BaseAgent
from tools import web

logger = logging.getLogger("viyon.pulse")

_MAX_SOURCES = 5
_FETCH_TOP = 2  # how many top sources to fetch full text for


class PulseAgent(BaseAgent):
    """VIYON's research agent."""

    name = "PULSE"
    emoji = "🔬"
    scope = "Research — web search and paper/URL summarization."

    def system_prompt(self) -> str:
        return (
            "You are PULSE 🔬, VIYON's rigorous researcher. Synthesize the provided "
            "search results and excerpts into a clear, structured summary.\n"
            "Rules:\n"
            "- Cite sources inline by their number, e.g. [1], and never invent a source.\n"
            "- Clearly distinguish established fact from your own inference.\n"
            "- Lead with the key findings; be concise and neutral.\n"
            "- If the sources are thin or conflicting, say so."
        )

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Search, fetch top sources, and synthesize a cited summary."""
        queries = self._plan_queries(task)

        results: list[dict] = []
        seen: set[str] = set()
        for query in queries:
            for r in await web.search(query, count=_MAX_SOURCES):
                url = r.get("url")
                if url and url not in seen:
                    seen.add(url)
                    results.append(r)

        if not results:
            return self.fail(
                "I couldn't find sources for that — web search may be unavailable "
                "(no API key) or returned nothing."
            )

        top = results[:_MAX_SOURCES]
        excerpts = await self._gather_excerpts(top)
        synthesis = await self.think(self._frame(task, top, excerpts), ctx)

        source_lines = [f"[{i}] {r['title']} — {r['url']}" for i, r in enumerate(top, 1)]
        detail = (synthesis or "").strip() + "\n\nSources:\n" + "\n".join(source_lines)
        spoken = f"Found {len(top)} sources. " + self._first_sentences(synthesis)
        return self.succeed(spoken, detail=detail)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _plan_queries(task: str) -> list[str]:
        """Plan up to two search queries from the task."""
        base = (task or "").strip()
        cleaned = base
        for prefix in ("research ", "look up ", "find out ", "search for ", "tell me about "):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        queries = [base]
        if cleaned and cleaned != base:
            queries.append(cleaned)
        return queries

    async def _gather_excerpts(self, sources: list[dict]) -> list[str]:
        """Best-effort fetch of the top sources' text (failures are ignored)."""
        excerpts = []
        for i, r in enumerate(sources[:_FETCH_TOP], 1):
            try:
                text = await web.fetch(r["url"])
            except Exception:
                text = ""
            if text:
                excerpts.append(f"[{i}] {r['url']}\n{text[:1000]}")
        return excerpts

    def _frame(self, task: str, sources: list[dict], excerpts: list[str]) -> str:
        listed = "\n".join(
            f"[{i}] {r['title']} — {r['url']}\n    {r.get('description', '')}"
            for i, r in enumerate(sources, 1)
        )
        body = f"Research question: {task}\n\nSearch results:\n{listed}"
        if excerpts:
            body += "\n\nFetched excerpts:\n" + "\n\n".join(excerpts)
        body += "\n\nWrite a structured summary with key findings and inline [n] citations."
        return body

    @staticmethod
    def _first_sentences(text: str, limit: int = 220) -> str:
        text = (text or "").strip().replace("\n", " ")
        return text[:limit] + ("…" if len(text) > limit else "")
