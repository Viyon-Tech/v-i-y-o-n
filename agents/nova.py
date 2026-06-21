"""NOVA agent (coding): writes, edits, reviews, and debugs code via the claude-code MCP.

NOVA attaches the claude-code MCP server so Claude can read_file / str_replace /
bash / search_code / git on the active project. It reads and plans first, then
gates any mutation (edit, shell, git push) through the Approval Gate before
applying. If the MCP server is offline it degrades to plain code generation
(no file access) and says so.
"""

from __future__ import annotations

import logging
import os

from agents.base_agent import AgentResult, BaseAgent
from integrations import claude_code_mcp

logger = logging.getLogger("viyon.nova")

# Verbs that imply changing the project (vs. reviewing/explaining it).
_MUTATION_WORDS = (
    "edit", "change", "fix", "refactor", "implement", "add", "create", "write",
    "delete", "remove", "rename", "update", "modify", "commit", "push",
    "install", "run", "build", "format", "migrate",
)


class NovaAgent(BaseAgent):
    """VIYON's coding agent, backed by the claude-code MCP server."""

    name = "NOVA"
    emoji = "💻"
    scope = "Coding — read, write, review, debug; runs git via the claude-code MCP."

    def system_prompt(self) -> str:
        return (
            "You are NOVA 💻, VIYON's senior software engineer. You work on the user's "
            "active project through the claude-code tools.\n"
            "Rules:\n"
            "- ALWAYS read the relevant files before proposing or making changes.\n"
            "- Explain the planned changes clearly before applying them.\n"
            "- NEVER apply edits, run shell commands, or push to git without explicit "
            "approval — VIYON gates those for you.\n"
            "- After applying changes, run the project's tests and report the result.\n"
            "- Keep diffs minimal and match the surrounding code's style."
        )

    # -- run ---------------------------------------------------------------

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Plan with the coding tools, then gate any mutation before applying."""
        project = self._project_path(ctx)
        mcp_up = await claude_code_mcp.health_check(self.config)

        if not mcp_up:
            code = await self.think(self._frame_plan(task, project), ctx)
            return self.succeed(
                "The claude-code server is offline, so I generated the code without "
                "touching your files. Review and apply it yourself.",
                detail=code,
            )

        mcp_servers = [claude_code_mcp.server_config(self.config)]
        plan = await self.think(self._frame_plan(task, project), ctx, mcp_servers=mcp_servers)

        if not self._is_mutation(task):
            return self.succeed(plan or "No findings.", detail=plan)

        async def _apply() -> AgentResult:
            outcome = await self.think(
                self._frame_apply(task, project, plan), ctx, mcp_servers=mcp_servers
            )
            return self.succeed("Applied the changes and ran the tests.", detail=outcome)

        return await self.guarded(
            "apply_changes",
            f"NOVA will edit {project} and run tests for: {task}",
            "high",
            _apply,
        )

    # -- helpers -----------------------------------------------------------

    def _project_path(self, ctx: dict) -> str:
        """Active project: from ctx, then config, then the current directory."""
        return (
            (ctx or {}).get("project")
            or (ctx or {}).get("project_path")
            or self._conf("coding", "project_path", "")
            or os.getcwd()
        )

    @staticmethod
    def _is_mutation(task: str) -> bool:
        low = (task or "").lower()
        return any(word in low for word in _MUTATION_WORDS)

    @staticmethod
    def _frame_plan(task: str, project: str) -> str:
        return (
            f"Active project: {project}\n\n"
            f"Task: {task}\n\n"
            "Read the relevant files first, then explain your plan and the exact changes "
            "you would make. Do not apply anything yet."
        )

    @staticmethod
    def _frame_apply(task: str, project: str, plan: str) -> str:
        return (
            f"Active project: {project}\n\n"
            f"Apply the following approved plan using your tools, then run the project's "
            f"tests and report results:\n\n{plan}"
        )
