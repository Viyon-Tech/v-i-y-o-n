"""SHIELD agent (cybersecurity): secret scans, dependency audits, network checks.

SHIELD reports risks — it does not exploit. Read-only scans (secrets) run
un-gated; anything that executes a tool (dependency audit, listening-port check)
goes through the Approval Gate. Operate only on systems the user owns/authorizes.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from agents.base_agent import AgentResult, BaseAgent
from tools import file_ops, terminal

logger = logging.getLogger("viyon.shield")

_SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "Anthropic API key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)\b(password|secret|token|api[_-]?key)\b\s*[=:]\s*['\"][^'\"]{6,}"),
     "hard-coded credential"),
]
_SCAN_EXTS = {".py", ".js", ".ts", ".env", ".yaml", ".yml", ".json", ".sh", ".txt", ".cfg", ".ini"}


class ShieldAgent(BaseAgent):
    """VIYON's cybersecurity agent."""

    name = "SHIELD"
    emoji = "🛡️"
    scope = "Cybersecurity — secret scans, dependency audits, network checks."

    def system_prompt(self) -> str:
        return (
            "You are SHIELD 🛡️, VIYON's security analyst. Find and clearly explain risks "
            "in code and configuration. Only ever act on systems the user owns or has "
            "authorized. You report and recommend — you never exploit or attack."
        )

    def _roots(self) -> list:
        return self._conf("filesystem", "allowed_paths", []) or []

    # -- capabilities ------------------------------------------------------

    async def scan_secrets(self, path: str = ".") -> AgentResult:
        """Read-only scan of a directory for committed secrets/credentials."""
        try:
            files = await file_ops.find("*", root=path, allowed_roots=self._roots())
        except PermissionError as exc:
            return self.fail(f"That path isn't allowed: {exc}")

        hits: list[str] = []
        for f in files:
            if Path(f).suffix.lower() not in _SCAN_EXTS:
                continue
            try:
                content = await file_ops.read_file(f, allowed_roots=self._roots())
            except Exception:
                continue
            for rx, label in _SECRET_PATTERNS:
                if rx.search(content):
                    hits.append(f"{label} → {f}")
            if len(hits) >= 100:
                break

        if not hits:
            return self.succeed(f"No obvious secrets found in {Path(path).name}.")
        return self.succeed(
            f"⚠ Found {len(hits)} potential secret(s) in {Path(path).name}.",
            detail="\n".join(hits[:100]),
        )

    async def dependency_audit(self, path: str = ".") -> AgentResult:
        """Run pip-audit against a requirements file — gated."""
        req = Path(path).expanduser() / "requirements.txt"

        async def _audit() -> AgentResult:
            code, out, err = await terminal.run(
                [sys.executable, "-m", "pip_audit", "-r", str(req)],
                approval_required=False,
            )
            if code == 0:
                return self.succeed("Dependency audit clean — no known vulnerabilities.", detail=out)
            if "No module named" in err or code == 127:
                return self.fail("pip-audit isn't installed (pip install pip-audit).")
            return self.succeed("Dependency audit found issues.", detail=(out or err)[:1500])

        return await self.guarded(
            "dependency_audit", f"run pip-audit on {req}", "low", _audit
        )

    async def network_connections(self) -> AgentResult:
        """List listening TCP ports (macOS lsof) — gated."""
        async def _net() -> AgentResult:
            code, out, err = await terminal.run(
                ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], approval_required=False
            )
            if code != 0:
                return self.fail(f"Couldn't list connections: {err[:200]}")
            lines = [l for l in out.splitlines() if l.strip()][1:]
            return self.succeed(
                f"{len(lines)} listening port(s).", detail="\n".join(lines[:50])
            )

        return await self.guarded("network_scan", "list listening TCP ports", "low", _net)

    # -- run ---------------------------------------------------------------

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Dispatch a security request."""
        low = (task or "").lower()
        path = self._extract_path(task) or self._conf("coding", "project_path", "") or "."

        if "network" in low or "port" in low or "connection" in low:
            return await self.network_connections()
        if "depend" in low or "audit" in low or "package" in low or ("vuln" in low and "code" not in low):
            return await self.dependency_audit(path)
        # Default: scan for secrets/credentials.
        return await self.scan_secrets(path)

    @staticmethod
    def _extract_path(task: str) -> str | None:
        m = re.search(r"(/\S+|~\S+|\S+/\S+)", task or "")
        return m.group(1) if m else None
