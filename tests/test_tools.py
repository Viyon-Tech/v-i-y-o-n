"""Smoke tests for the Mac/file/terminal tools and ATLAS/GHOST agents.

GUI-free: GHOST snapshot returns real numbers; file_ops does a write→read
roundtrip in a temp dir and refuses delete without approval; terminal gating is
exercised with a harmless ``echo``. macOS-only paths are skipped elsewhere.
"""

from __future__ import annotations

import sys

import pytest

from agents.ghost import GhostAgent
from tools import file_ops, terminal
from tools.approval import ApprovalGate

IS_MAC = sys.platform == "darwin"


def gate(answer: str) -> ApprovalGate:
    """An ApprovalGate whose console callback always returns ``answer``."""
    return ApprovalGate(callback=lambda prompt: answer, config={})


def make_ghost() -> GhostAgent:
    return GhostAgent(llm=None, tools=None, log=None, approval=gate("no"), config={})


# -- GHOST -------------------------------------------------------------------

async def test_ghost_snapshot_returns_numbers():
    snap = await make_ghost().snapshot()
    for key in ("cpu_percent", "mem_percent", "disk_percent", "mem_total_gb"):
        assert isinstance(snap[key], (int, float))
    assert 0 <= snap["cpu_percent"] <= 100
    assert 0 <= snap["disk_percent"] <= 100


async def test_ghost_run_gives_summary_and_detail():
    result = await make_ghost().run("how is my system doing", ctx={})
    assert result.ok is True
    assert result.agent == "GHOST"
    assert "%" in result.summary
    assert result.detail and "CPU" in result.detail


async def test_ghost_top_processes():
    top = await make_ghost().top_processes(by="mem", n=3)
    assert 1 <= len(top) <= 3
    assert all("pid" in p and "name" in p for p in top)


# -- file_ops ----------------------------------------------------------------

async def test_write_read_roundtrip(tmp_path):
    target = tmp_path / "notes" / "hello.txt"
    ok, msg = await file_ops.write_file(
        target, "hello viyon", approval=gate("yes"), allowed_roots=[tmp_path]
    )
    assert ok is True, msg
    content = await file_ops.read_file(target, allowed_roots=[tmp_path])
    assert content == "hello viyon"


async def test_delete_refused_without_approval(tmp_path):
    target = tmp_path / "keep.txt"
    target.write_text("important")
    ok, msg = await file_ops.delete(target, approval=None, allowed_roots=[tmp_path])
    assert ok is False
    assert "approved" in msg
    assert target.exists()  # still there


async def test_delete_refused_when_denied(tmp_path):
    target = tmp_path / "keep.txt"
    target.write_text("important")
    ok, _ = await file_ops.delete(target, approval=gate("no"), allowed_roots=[tmp_path])
    assert ok is False
    assert target.exists()


async def test_delete_succeeds_when_approved(tmp_path):
    target = tmp_path / "trash.txt"
    target.write_text("bye")
    ok, _ = await file_ops.delete(target, approval=gate("yes"), allowed_roots=[tmp_path])
    assert ok is True
    assert not target.exists()


async def test_path_outside_allowed_roots_is_refused(tmp_path):
    """A temp path not in allowed_roots (and outside home) raises PermissionError."""
    with pytest.raises(PermissionError):
        await file_ops.read_file(tmp_path / "x.txt")  # no allowed_roots, outside home


async def test_find_in_temp(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    matches = await file_ops.find("*.py", root=tmp_path, allowed_roots=[tmp_path])
    assert any(m.endswith("a.py") for m in matches)


# -- terminal ----------------------------------------------------------------

async def test_terminal_runs_when_not_gated():
    code, out, err = await terminal.run(["echo", "hello"], approval_required=False)
    assert code == 0
    assert out.strip() == "hello"


async def test_terminal_refused_without_gate():
    code, out, err = await terminal.run(["echo", "x"], approval_required=True, approval=None)
    assert code == 126
    assert "approval" in err


async def test_terminal_refused_when_denied():
    code, out, err = await terminal.run(
        ["echo", "x"], approval_required=True, approval=gate("no")
    )
    assert code == 125
    assert "not approved" in err


async def test_terminal_runs_when_approved():
    code, out, _ = await terminal.run(
        ["echo", "ok"], approval_required=True, approval=gate("yes")
    )
    assert code == 0 and out.strip() == "ok"


async def test_terminal_missing_binary():
    code, _, err = await terminal.run(["definitely-not-a-real-binary-xyz"], approval_required=False)
    assert code == 127


# -- ATLAS (file paths, no GUI) ----------------------------------------------

def make_atlas(answer: str, tmp_path):
    from agents.atlas import AtlasAgent

    return AtlasAgent(
        llm=None,
        tools=None,
        log=None,
        approval=gate(answer),
        config={"filesystem": {"allowed_paths": [str(tmp_path)]}},
    )


def test_atlas_system_prompt_states_rules():
    atlas = make_atlas("no", "/tmp")
    prompt = atlas.system_prompt()
    assert "ATLAS" in prompt
    assert "delete" in prompt.lower() and "confirm" in prompt.lower()


async def test_atlas_reads_allowed_file(tmp_path):
    (tmp_path / "note.txt").write_text("hi there")
    atlas = make_atlas("no", tmp_path)
    result = await atlas.run(f"read the file {tmp_path}/note.txt", ctx={})
    assert result.ok is True
    assert result.detail == "hi there"


async def test_atlas_delete_blocked_when_denied(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("keep me")
    atlas = make_atlas("no", tmp_path)
    result = await atlas.run(f"delete {target}", ctx={})
    assert result.ok is False
    assert target.exists()


async def test_atlas_rejects_path_outside_allowed(tmp_path):
    """ATLAS surfaces a friendly refusal for paths outside the allowed roots."""
    atlas = make_atlas("yes", tmp_path)
    result = await atlas.run("read the file /etc/hosts", ctx={})
    assert result.ok is False
    assert "allowed" in result.summary.lower()


# -- mac_control (macOS only) ------------------------------------------------

@pytest.mark.skipif(not IS_MAC, reason="clipboard tools require macOS")
async def test_clipboard_roundtrip():
    from tools import mac_control

    assert await mac_control.clipboard_set("viyon-clip-test") is True
    assert await mac_control.clipboard_get() == "viyon-clip-test"
