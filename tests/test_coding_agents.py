"""Tests for NOVA and FORGE. The MCP/LLM layer is mocked — no network, no server.

Key contracts: NOVA refuses to apply changes without approval (and doesn't call
the apply step), and FORGE plans a full file tree for a FastAPI + JWT project.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock


from agents.forge import ForgeAgent
from agents.nova import NovaAgent
from tools.approval import ApprovalGate


def fake_llm(text: str = "PLAN: read app.py, change line 10."):
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))


def gate(answer: str) -> ApprovalGate:
    return ApprovalGate(callback=lambda prompt: answer, config={})


def make_nova(answer: str, llm=None):
    return NovaAgent(
        llm=llm or fake_llm(),
        tools=None,
        log=None,
        approval=gate(answer),
        config={"coding": {"project_path": "/proj", "mcp_url": "http://localhost:3000/mcp"}},
    )


# -- NOVA --------------------------------------------------------------------

async def test_nova_refuses_to_write_without_approval(monkeypatch):
    """With the MCP up but approval denied, NOVA aborts and never runs the apply step."""
    monkeypatch.setattr("agents.nova.claude_code_mcp.health_check", AsyncMock(return_value=True))
    nova = make_nova("no")

    result = await nova.run("fix the off-by-one bug in app.py", ctx={})

    assert result.ok is False
    assert "approved" in result.summary.lower() or "aborted" in result.summary.lower()
    # only the planning think() ran — the apply think() was gated out
    assert nova.llm.messages.create.call_count == 1


async def test_nova_applies_when_approved(monkeypatch):
    monkeypatch.setattr("agents.nova.claude_code_mcp.health_check", AsyncMock(return_value=True))
    nova = make_nova("yes")

    result = await nova.run("refactor the auth module", ctx={})

    assert result.ok is True
    assert "applied" in result.summary.lower()
    assert nova.llm.messages.create.call_count == 2  # plan + apply


async def test_nova_review_task_does_not_gate(monkeypatch):
    """A non-mutating task (explain) returns a plan without any approval prompt."""
    monkeypatch.setattr("agents.nova.claude_code_mcp.health_check", AsyncMock(return_value=True))
    nova = make_nova("no")  # would deny — but should never be asked

    result = await nova.run("explain how the login flow works", ctx={})

    assert result.ok is True
    assert nova.llm.messages.create.call_count == 1


async def test_nova_degrades_when_mcp_offline(monkeypatch):
    monkeypatch.setattr("agents.nova.claude_code_mcp.health_check", AsyncMock(return_value=False))
    nova = make_nova("yes")

    result = await nova.run("fix the bug in app.py", ctx={})

    assert result.ok is True
    assert "offline" in result.summary.lower()
    assert nova.llm.messages.create.call_count == 1  # plain generation, no apply


async def test_nova_project_path_from_ctx(monkeypatch):
    monkeypatch.setattr("agents.nova.claude_code_mcp.health_check", AsyncMock(return_value=False))
    nova = make_nova("yes")
    assert nova._project_path({"project": "/work/repo"}) == "/work/repo"
    assert nova._project_path({}) == "/proj"  # falls back to config


# -- FORGE -------------------------------------------------------------------

def make_forge(answer: str, tmp_path=None):
    config = {"coding": {"editor": "Code"}}
    if tmp_path is not None:
        config["filesystem"] = {"allowed_paths": [str(tmp_path)]}
    return ForgeAgent(llm=fake_llm(), tools=None, log=None, approval=gate(answer), config=config)


def test_forge_detects_fastapi():
    forge = make_forge("no")
    assert forge.detect_stack("create a FastAPI project with JWT auth") == "fastapi"
    assert forge.detect_stack("build a Next.js dashboard") == "nextjs"
    assert forge.detect_stack("a little python utility") == "python"


def test_forge_plans_fastapi_jwt_tree():
    forge = make_forge("no")
    tree = forge.plan_tree("create a FastAPI project with JWT auth")
    assert "app/main.py" in tree
    assert "requirements.txt" in tree
    # JWT auth pulls in security/auth/user files
    assert "app/core/security.py" in tree
    assert "app/api/auth.py" in tree
    assert "app/models/user.py" in tree


async def test_forge_returns_plan_when_not_approved():
    """Denied scaffold still hands back the planned tree in detail."""
    forge = make_forge("no")
    result = await forge.run("create a FastAPI project with JWT auth", ctx={})
    assert result.ok is False
    assert "main.py" in result.detail
    assert "security.py" in result.detail


async def test_forge_scaffolds_files_when_approved(tmp_path):
    forge = make_forge("yes", tmp_path)
    result = await forge.run(
        "create a FastAPI project with JWT auth", ctx={"project_dir": str(tmp_path)}
    )
    assert result.ok is True
    target = tmp_path / "fastapi-jwt-auth"
    assert (target / "app" / "main.py").exists()
    assert (target / "requirements.txt").exists()
    assert (target / "app" / "core" / "security.py").exists()
    # main.py has real starter content, not just a stub
    assert "FastAPI" in (target / "app" / "main.py").read_text()
    assert "main.py" in result.detail
