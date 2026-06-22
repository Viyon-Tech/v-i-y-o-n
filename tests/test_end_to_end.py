"""End-to-end wiring smoke: build the real CORE (12 agents) and route 4 commands.

No external keys (LLM=None → keyword routing; web search empty). A recording,
denying approval gate proves the gate fires on ATLAS without launching anything.
Everything is logged to a temp database.
"""

from __future__ import annotations

import pytest

from logs.logger import VLogger
from main import build_core
from tools.approval import ApprovalGate


class _Silent:
    async def say(self, text):  # no audio in tests
        return None


@pytest.fixture()
def harness(tmp_path):
    prompts = []
    gate = ApprovalGate(callback=lambda p: (prompts.append(p), "no")[1], config={})
    vlog = VLogger(base_dir=tmp_path)
    core = build_core(vlog=vlog, approval=gate, speaker=_Silent(), llm=None)
    yield core, vlog, prompts
    vlog.close()


def _row_for(vlog, command):
    for row in vlog.get_recent(50):
        if row["raw_input"] == command:
            return row
    return None


async def test_full_text_smoke(harness):
    core, vlog, prompts = harness
    commands = [
        ("what's eating my CPU", {"GHOST"}),
        ("open Safari", {"ATLAS"}),
        ("research RAG and scaffold a demo", {"PULSE", "FORGE"}),
        ("I'm stressed about my deadline", {"LUNA"}),
    ]

    for text, _ in commands:
        await core.handle(text)

    # 1) every command routed to the expected agent(s)
    for text, expected in commands:
        row = _row_for(vlog, text)
        assert row is not None, f"no log row for {text!r}"
        assert set(row["agents"]) == expected, f"{text!r} → {row['agents']}"

    # 2) the approval gate fired on ATLAS (open) — and was denied (no Safari launched)
    assert any("open" in p.lower() for p in prompts), prompts
    # FORGE's scaffold also prompted
    assert any("scaffold" in p.lower() or "create" in p.lower() for p in prompts)

    # 3) all four commands are in the database
    logged = {r["raw_input"] for r in vlog.get_recent(50)}
    for text, _ in commands:
        assert text in logged


async def test_research_runs_in_parallel(harness):
    core, vlog, _ = harness
    await core.handle("research RAG and scaffold a demo")
    row = _row_for(vlog, "research RAG and scaffold a demo")
    assert set(row["agents"]) == {"PULSE", "FORGE"}


async def test_private_command_is_redacted(harness):
    core, vlog, _ = harness
    await core.handle("privately journal that I feel anxious about money")
    rows = vlog.get_recent(10)
    blob = " ".join(str(r["raw_input"]) + str(r["result"]) + str(r["steps"]) for r in rows)
    assert "anxious about money" not in blob
    assert any(r["raw_input"] == "private entry" for r in rows)
