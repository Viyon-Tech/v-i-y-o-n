"""Tests for the BaseAgent contract and AgentResult.

A DummyAgent proves the lifecycle works end to end and that guarded() blocks
when the Approval Gate denies. The Anthropic client is mocked — no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock


from agents.base_agent import AgentResult, BaseAgent
from tools.approval import ApprovalGate


class DummyAgent(BaseAgent):
    """Minimal concrete agent used to exercise the BaseAgent contract."""

    name = "DUMMY"
    emoji = "🧪"
    scope = "A test agent that echoes and performs one guarded action."

    def system_prompt(self) -> str:
        return f"You are {self.name}. Answer in one word."

    async def run(self, task: str, ctx: dict) -> AgentResult:
        # A real agent would think() then funnel side effects through guarded().
        return await self.guarded(
            action="write_file",
            detail=f"create file for: {task}",
            risk="high",
            fn=lambda: self.succeed("file written", artifacts=["/tmp/dummy.txt"]),
        )


def fake_llm(text: str):
    """A stand-in AsyncAnthropic returning ``text`` as one text block."""
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))


def make_agent(answer: str, llm_text: str = "ok", config=None):
    """Build a DummyAgent whose approval callback always returns ``answer``."""
    approval = ApprovalGate(callback=lambda prompt: answer, config=config or {})
    return DummyAgent(
        llm=fake_llm(llm_text),
        tools=SimpleNamespace(),
        log=SimpleNamespace(),
        approval=approval,
        config={"llm": {"model": "claude-sonnet-4-6", "max_tokens": 256}},
    )


# -- AgentResult -------------------------------------------------------------

def test_agent_result_defaults():
    r = AgentResult(agent="X")
    assert r.ok is True
    assert r.summary == ""
    assert r.detail is None
    assert r.artifacts == []
    assert r.handoff is None
    assert r.needs_confirm is False


# -- BaseAgent contract ------------------------------------------------------

def test_class_attributes_and_config_wiring():
    agent = make_agent("yes")
    assert agent.name == "DUMMY"
    assert agent.emoji == "🧪"
    assert "test agent" in agent.scope.lower()
    assert isinstance(agent.system_prompt(), str) and agent.system_prompt()
    # config nested-dict is read for model/max_tokens
    assert agent.model == "claude-sonnet-4-6"
    assert agent.max_tokens == 256


async def test_run_succeeds_when_approved():
    agent = make_agent("yes")
    result = await agent.run("make a thing", ctx={})
    assert isinstance(result, AgentResult)
    assert result.ok is True
    assert result.agent == "DUMMY"
    assert result.summary == "file written"
    assert result.artifacts == ["/tmp/dummy.txt"]


async def test_guarded_blocks_when_denied():
    """guarded() must not run fn when approval is denied, and returns an abort."""
    ran = {"called": False}

    def side_effect():
        ran["called"] = True
        return AgentResult(agent="DUMMY", ok=True, summary="should not happen")

    agent = make_agent("no")
    result = await agent.guarded("delete_all", "rm -rf everything", "high", side_effect)

    assert ran["called"] is False
    assert result.ok is False
    assert "Aborted" in result.summary
    assert result.detail == "rm -rf everything"


async def test_guarded_runs_async_fn_when_approved():
    """guarded() awaits an async fn when approved."""
    agent = make_agent("yes")

    async def do_work():
        return agent.succeed("done the work")

    result = await agent.guarded("send_email", "email Bob", "high", do_work)
    assert result.ok is True
    assert result.summary == "done the work"


async def test_dummy_run_blocked_end_to_end():
    """The full run() aborts cleanly when its guarded action is denied."""
    agent = make_agent("no")
    result = await agent.run("make a thing", ctx={})
    assert result.ok is False
    assert "Aborted" in result.summary


async def test_think_calls_llm_with_system_prompt():
    """think() sends the agent's system prompt and returns response text."""
    agent = make_agent("yes", llm_text="Pong")
    text = await agent.think("ping", ctx={"history": [("user", "hi"), ("assistant", "hello")]})
    assert text == "Pong"
    # the system prompt was passed through
    _, kwargs = agent.llm.messages.create.call_args
    assert kwargs["system"] == agent.system_prompt()
    assert kwargs["model"] == "claude-sonnet-4-6"
