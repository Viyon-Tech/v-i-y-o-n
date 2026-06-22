"""NOVA agent (coding): writes, edits, reviews, and debugs code via the claude-code MCP.

NOVA attaches the claude-code MCP server so Claude can read_file / str_replace /
bash / search_code / git on the active project. It reads and plans first, then
gates any mutation (edit, shell, git push) through the Approval Gate before
applying.

Degraded mode: if the claude-code server is offline, OR Claude is unavailable
(billing/auth/quota) and NOVA falls back to a local model, the MCP file tools are
NOT available. In that mode NOVA must NOT claim it edited files — it generates the
code and offers to save it via tools/file_ops (gated), being honest that it's
running locally.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from agents.base_agent import AgentResult, BaseAgent, ClaudeUnavailable
from integrations import claude_code_mcp
from tools import file_ops

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

        # No MCP server → generate-only (degraded) mode.
        if not mcp_up:
            try:
                code = await self.think(self._frame_plan(task, project), ctx)
            except ClaudeUnavailable as exc:
                return self.fail(str(exc), detail="claude_unavailable")
            return await self._local_result(
                task, project, ctx, code, note="the claude-code server is offline"
            )

        mcp_servers = [claude_code_mcp.server_config(self.config)]
        try:
            plan = await self.think(self._frame_plan(task, project), ctx, mcp_servers=mcp_servers)
        except ClaudeUnavailable as exc:
            return self.fail(str(exc), detail="claude_unavailable")

        # Claude billing/etc. fell back to a local model → MCP tools were NOT used.
        if self.degraded(ctx):
            return await self._local_result(
                task, project, ctx, plan,
                note="Claude is unavailable, so I'm running on a local model",
            )

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

    # -- degraded (local) mode --------------------------------------------

    async def _local_result(self, task: str, project: str, ctx: dict, code: str, note: str) -> AgentResult:
        """Honest local mode: no MCP edits. Generate code; offer to save via file_ops."""
        code = code or "(no code generated)"
        if not self._is_mutation(task):
            return self.succeed(
                f"{note} — here's my analysis (local mode, no file changes).", detail=code
            )

        # Route the write through file_ops (gated) — NOT the MCP editor.
        target = self._derive_target(task, project)
        try:
            ok, _ = await file_ops.write_file(
                target, code, approval=self.approval, allowed_roots=self._roots(ctx)
            )
        except Exception as exc:  # e.g. path outside allowed roots
            logger.warning("NOVA local save failed: %s", exc)
            ok = False
        if ok:
            return self.succeed(
                f"{note}, so I couldn't use the code editor. I generated the code and saved it "
                f"to {target} via file tools.",
                detail=code,
                artifacts=[target],
            )
        return self.succeed(
            f"{note}, so I can't edit files directly. Here's the generated code — saving to "
            f"{target} wasn't approved, so nothing was written.",
            detail=code,
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

    def _roots(self, ctx: dict) -> list:
        extra = self._conf("filesystem", "allowed_paths", []) or []
        base = (ctx or {}).get("project_dir") or (ctx or {}).get("project")
        return [base, *extra] if base else extra

    @staticmethod
    def _derive_target(task: str, project: str) -> str:
        """Pick an output filename from the task, else a default, under the project."""
        m = re.search(r"[\w./-]+\.[A-Za-z0-9]+", task or "")
        name = m.group(0) if m else "viyon_generated.txt"
        path = Path(name)
        return str(path if path.is_absolute() else Path(project) / path)

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
