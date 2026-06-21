"""FORGE agent (project scaffolding): turns a voice description into a real project.

FORGE picks a stack (FastAPI/Python, Next.js/React, Tailwind), plans a full
directory + file tree, and — after one approval — creates it via the file_ops
tools with sensible starter content. It returns the tree in ``detail`` and offers
to open the project in the editor (via ATLAS/mac_control).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agents.base_agent import AgentResult, BaseAgent
from tools import file_ops, mac_control

logger = logging.getLogger("viyon.forge")


class ForgeAgent(BaseAgent):
    """VIYON's project scaffolding specialist."""

    name = "FORGE"
    emoji = "🏗️"
    scope = "Project scaffolding — React/Next.js, FastAPI, Python, Tailwind."

    def system_prompt(self) -> str:
        return (
            "You are FORGE 🏗️, VIYON's project scaffolding specialist. From a short "
            "description you design a complete, idiomatic project structure. You know "
            "React/Next.js, FastAPI, Python, and Tailwind well. Plan the full tree, "
            "explain it, and only create files after the user approves."
        )

    # -- planning ----------------------------------------------------------

    def detect_stack(self, description: str) -> str:
        """Pick a stack from the description (defaults to plain Python)."""
        d = description.lower()
        if "next" in d:
            return "nextjs"
        if "react" in d:
            return "react"
        if "fastapi" in d:
            return "fastapi"
        if "flask" in d:
            return "flask"
        return "python"

    def plan_tree(self, description: str) -> list[str]:
        """Return the planned relative paths for the detected stack + features."""
        stack = self.detect_stack(description)
        d = description.lower()
        wants_auth = "jwt" in d or "auth" in d
        wants_tailwind = "tailwind" in d

        if stack == "fastapi":
            tree = [
                "app/__init__.py",
                "app/main.py",
                "app/core/__init__.py",
                "app/core/config.py",
                "app/api/__init__.py",
                "app/api/routes.py",
                "app/models/__init__.py",
                "requirements.txt",
                ".env.example",
                ".gitignore",
                "README.md",
                "tests/__init__.py",
                "tests/test_main.py",
            ]
            if wants_auth:
                tree += [
                    "app/core/security.py",
                    "app/api/auth.py",
                    "app/models/user.py",
                    "tests/test_auth.py",
                ]
            return sorted(tree)

        if stack in ("nextjs", "react"):
            tree = [
                "package.json",
                "tsconfig.json",
                "next.config.js",
                "app/layout.tsx",
                "app/page.tsx",
                "components/.gitkeep",
                "public/.gitkeep",
                "styles/globals.css",
                ".gitignore",
                "README.md",
            ]
            if wants_tailwind or True:  # Next.js scaffolds ship Tailwind by default here
                tree += ["tailwind.config.js", "postcss.config.js"]
            return sorted(tree)

        if stack == "flask":
            return sorted([
                "app/__init__.py",
                "app/routes.py",
                "requirements.txt",
                ".gitignore",
                "README.md",
                "tests/test_app.py",
            ])

        # Plain Python package.
        return sorted([
            "src/__init__.py",
            "src/main.py",
            "pyproject.toml",
            ".gitignore",
            "README.md",
            "tests/__init__.py",
            "tests/test_main.py",
        ])

    # -- run ---------------------------------------------------------------

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Plan a project tree, gate creation once, then scaffold it."""
        description = task or ""
        stack = self.detect_stack(description)
        tree = self.plan_tree(description)
        name = self._project_name(description)
        target = self._target_dir(ctx, name)
        rendered = self._render_tree(name, tree)

        async def _build() -> AgentResult:
            for rel in tree:
                path = target / rel
                await file_ops.make_dir(path.parent, allowed_roots=self._roots(ctx))
                await file_ops.write_file(
                    path,
                    self._starter(rel, stack, name),
                    allowed_roots=self._roots(ctx),
                    require_approval=False,  # one batch approval already granted
                )
            editor = self._conf("coding", "editor", "your editor")
            return self.succeed(
                f"Scaffolded a {stack} project '{name}' with {len(tree)} files at {target}. "
                f"Say 'open it' to launch it in {editor}.",
                detail=rendered,
                artifacts=[str(target)],
            )

        result = await self.guarded(
            "scaffold_project",
            f"create {len(tree)} files for a {stack} project at {target}",
            "medium",
            _build,
        )
        if not result.ok:
            # Not approved — still hand back the plan so the user can review it.
            return self.fail(
                f"Scaffold not approved. Here's the planned {stack} project '{name}'.",
                detail=rendered,
            )
        return result

    async def open_in_editor(self, target: str) -> tuple:
        """Open a scaffolded project in the configured editor (reversible)."""
        editor = self._conf("coding", "editor", "Finder")
        return await mac_control.open_app(editor)

    # -- helpers -----------------------------------------------------------

    def _roots(self, ctx: dict) -> list:
        extra = self._conf("filesystem", "allowed_paths", []) or []
        base = (ctx or {}).get("project_dir")
        return [base, *extra] if base else extra

    def _target_dir(self, ctx: dict, name: str) -> Path:
        base = (ctx or {}).get("project_dir") or self._conf("coding", "project_path", "") or str(Path.home())
        return Path(base).expanduser() / name

    @staticmethod
    def _project_name(description: str) -> str:
        """Derive a kebab-case-ish project name from the description."""
        m = re.search(r"(?:called|named)\s+([A-Za-z0-9_\- ]+)", description, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
        else:
            words = re.findall(r"[A-Za-z0-9]+", description.lower())
            stop = {"a", "an", "the", "create", "make", "build", "new", "project", "with", "and", "app"}
            keep = [w for w in words if w not in stop][:3]
            raw = "-".join(keep) or "viyon-project"
        return re.sub(r"\s+", "-", raw.strip()).lower()

    @staticmethod
    def _render_tree(name: str, tree: list[str]) -> str:
        """Render the planned paths as an indented tree string."""
        lines = [f"{name}/"]
        for rel in tree:
            depth = rel.count("/")
            lines.append("  " * (depth + 1) + Path(rel).name)
        return "\n".join(lines)

    def _starter(self, rel: str, stack: str, name: str) -> str:
        """Minimal starter content for a scaffolded file."""
        base = Path(rel).name
        if base == "README.md":
            return f"# {name}\n\nScaffolded by VIYON FORGE 🏗️ ({stack}).\n"
        if base == ".gitignore":
            return "__pycache__/\n*.pyc\n.env\n.venv/\nnode_modules/\ndist/\n.next/\n"
        if base == ".env.example":
            return "SECRET_KEY=change-me\n"
        if base == "requirements.txt":
            deps = ["fastapi", "uvicorn[standard]", "pydantic"]
            if stack == "fastapi":
                deps += ["python-jose[cryptography]", "passlib[bcrypt]"]
            return "\n".join(dict.fromkeys(deps)) + "\n"
        if base == "main.py" and stack == "fastapi":
            return (
                '"""FastAPI application entry point."""\n\n'
                "from fastapi import FastAPI\n\n"
                f'app = FastAPI(title="{name}")\n\n\n'
                '@app.get("/")\n'
                "async def root():\n"
                '    return {"status": "ok"}\n'
            )
        if base == "security.py":
            return (
                '"""JWT token creation and verification."""\n\n'
                "# TODO: implement create_access_token / decode_token using python-jose.\n"
            )
        if base == "auth.py":
            return (
                '"""Authentication routes (login, token issuance)."""\n\n'
                "# TODO: implement /login and /token endpoints with JWT.\n"
            )
        if base.endswith(".py"):
            return f'"""{name} — {rel}."""\n\n# TODO: implement.\n'
        if base == "package.json":
            return (
                "{\n"
                f'  "name": "{name}",\n'
                '  "version": "0.1.0",\n'
                '  "scripts": {"dev": "next dev", "build": "next build", "start": "next start"}\n'
                "}\n"
            )
        if base == ".gitkeep":
            return ""
        return f"// {rel} — scaffolded by VIYON FORGE.\n"
