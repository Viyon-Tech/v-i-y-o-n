"""VIYON CORE orchestrator: STT -> intent -> route -> parallel run -> merge -> TTS -> log.

VIYONCore is the heart of the system. It wires together voice I/O, the router,
the agent registry, the approval gate, session memory, and structured logging,
and exposes:

* :meth:`handle` — process one transcript end to end.
* :meth:`run_forever` — wake-word → listen → handle loop, with clean shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import time

from agents.base_agent import AgentResult
from core import config, events, parallel
from core.listener import Listener
from core.memory import SessionMemory
from core.router import RoutePlan, Router
from core.speaker import Speaker
from core.wake_word import WakeWord
from logs.logger import log
from tools.approval import ApprovalGate

logger = logging.getLogger("viyon.core")


class VIYONCore:
    """Central orchestrator that turns a spoken command into agent work and a reply.

    All collaborators are injectable so the core is testable without audio or a
    network; sensible defaults are constructed when omitted.
    """

    def __init__(
        self,
        *,
        agents: dict | None = None,
        listener: Listener | None = None,
        speaker: Speaker | None = None,
        router: Router | None = None,
        wake_word: WakeWord | None = None,
        approval: ApprovalGate | None = None,
        memory: SessionMemory | None = None,
        logger_=None,
    ) -> None:
        self.agents = agents or {}
        self.listener = listener or Listener()
        self.speaker = speaker or Speaker()
        self.router = router or Router()
        self.wake = wake_word or WakeWord()
        self.approval = approval or ApprovalGate()
        self.memory = memory or SessionMemory()
        self.log = logger_ if logger_ is not None else log

    async def handle(self, transcript: str) -> str | None:
        """Process one command: route → confirm → run → merge → speak → log.

        Returns the spoken reply, or None if the command was empty or aborted.
        """
        if not transcript or not transcript.strip():
            return None

        start = time.perf_counter()
        private = self._is_private(transcript)
        self.memory.add_user(transcript)
        cmd_id = self.log.log_command(
            {"raw_input": "private entry" if private else transcript, "status": "pending"}
        )
        ctx = {"history": self.memory.get_context(), "command_id": cmd_id, "private": private}
        events.emit_reset()
        events.emit_command("private entry" if private else transcript)

        # 1) Route.
        try:
            plan = await self.router.route(transcript, ctx)
        except Exception as exc:
            self.log.update_command(cmd_id, status="error", error=f"routing failed: {exc}")
            await self.speaker.say("Sorry, I couldn't work out what to do with that.")
            return None
        self.log.update_command(
            cmd_id,
            parsed_intent={"private": True} if private else plan.model_dump(),
            agents=[a.name for a in plan.agents],
        )

        # 2) Run the agents. Each agent gates its own destructive actions through
        #    the Approval Gate (which flashes the HUD amber while it asks), so CORE
        #    doesn't add a redundant plan-level prompt.
        for a in plan.agents:
            events.emit_agent(a.name, "working", active=True)
        results = await parallel.run_agents(plan, self.agents, ctx)
        for r in results:
            events.emit_agent(r.agent, "done" if r.ok else "idle")
        confirmed = any(not r.ok and "abort" in (r.summary or "").lower() for r in results)

        # 3) Compose a natural spoken reply.
        merged = await self._merge(transcript, plan, results)
        self.memory.add_assistant(merged)

        # 4) Speak and log.
        await self.speaker.say(merged)
        duration_ms = int((time.perf_counter() - start) * 1000)
        status = "ok" if results and all(r.ok for r in results) else "error"
        self.log.update_command(
            cmd_id,
            status=status,
            result="private entry" if private else merged,
            duration_ms=duration_ms,
            confirmed=not confirmed,
            steps=[] if private else [r.model_dump() for r in results],
        )
        return merged

    @staticmethod
    def _is_private(transcript: str) -> bool:
        """True if the user signalled this command should not be logged in full."""
        low = (transcript or "").lower()
        cues = ("private", "privately", "between us", "don't log", "do not log",
                "off the record", "keep this secret")
        return any(cue in low for cue in cues)

    async def _merge(self, transcript: str, plan: RoutePlan, results: list[AgentResult]) -> str:
        """Compose one spoken reply from the agents' structured outputs.

        Uses a single Claude call; if that's unavailable it gracefully falls back
        to concatenating the successful agent outputs.
        """
        if not results:
            return "I didn't have anything to run for that."

        successes = [r.summary for r in results if r.ok and r.summary]
        failures = [f"{r.agent} couldn't finish: {r.detail or r.summary}" for r in results if not r.ok]
        fallback = " ".join(successes + failures) or "Done."

        try:
            composed = await self._compose(transcript, plan, results)
            return composed or fallback
        except Exception as exc:
            logger.warning("Merge LLM call failed (%s); using concatenated outputs.", exc)
            return fallback

    async def _compose(self, transcript: str, plan: RoutePlan, results: list[AgentResult]) -> str:
        """Ask Claude to merge agent outputs into a natural spoken reply."""
        length = "one or two sentences" if plan.reply_style == "short" else "a few sentences"
        summary = "\n".join(
            f"- {r.agent}: {'OK' if r.ok else 'FAILED'} — {r.summary or r.detail or ''}"
            for r in results
        )
        system = (
            "You are VIYON, a voice assistant. Compose a single natural spoken reply "
            f"({length}) from the agent results below. Speak directly to the user, no "
            "markdown, no lists, no preamble like 'Here is'."
        )
        user = f"User said: {transcript}\n\nAgent results:\n{summary}"
        response = await self.router.client.messages.create(
            model=self.router.model,
            max_tokens=int(config.get("llm", "max_tokens", 1024)),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

    async def run_forever(self) -> None:
        """Loop: wait for wake word → listen → handle. Exits cleanly on Ctrl-C."""
        logger.info("VIYON online. Listening for the wake word.")
        try:
            while True:
                await self.wake.wait_for_wake()
                events.emit_listening(True)
                transcript = await self.listener.listen()
                events.emit_listening(False)
                if not transcript.strip():
                    continue
                await self.handle(transcript)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("VIYON shutting down.")
            try:
                await self.speaker.say("Goodbye.")
            except Exception:
                pass
