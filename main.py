"""VIYON entry point: boots CORE, the 12 agents, the HUD, and the voice loop.

Usage:
    python main.py                     # voice: wake "VIYON" → listen → handle, + HUD
    python main.py --text              # type commands through the same handle()
    python main.py --no-hud            # skip the HUD server
    python main.py --agent GHOST "..." # run one agent once and exit (for testing)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
from types import SimpleNamespace

from agents.atlas import AtlasAgent
from agents.echo import EchoAgent
from agents.forge import ForgeAgent
from agents.ghost import GhostAgent
from agents.luna import LunaAgent
from agents.nexus import NexusAgent
from agents.nova import NovaAgent
from agents.pulse import PulseAgent
from agents.sage import SageAgent
from agents.shield import ShieldAgent
from agents.tempo import TempoAgent
from agents.vista import VistaAgent
from core import config, events
from core.listener import Listener
from core.memory import SessionMemory
from core.orchestrator import VIYONCore
from core.router import Router
from core.speaker import Speaker
from core.wake_word import WakeWord
from logs.logger import VLogger
from tools import file_ops, mac_control, terminal, web
from tools.approval import ApprovalGate

logger = logging.getLogger("viyon.main")

_AGENT_CLASSES = {
    "NOVA": NovaAgent, "FORGE": ForgeAgent, "SHIELD": ShieldAgent, "PULSE": PulseAgent,
    "ATLAS": AtlasAgent, "ECHO": EchoAgent, "NEXUS": NexusAgent, "VISTA": VistaAgent,
    "TEMPO": TempoAgent, "SAGE": SageAgent, "LUNA": LunaAgent, "GHOST": GhostAgent,
}


def make_llm():
    """Return an AsyncAnthropic client if a key and the SDK are present, else None."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.info("ANTHROPIC_API_KEY not set — running with keyword routing only.")
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — LLM features disabled.")
        return None
    try:
        return anthropic.AsyncAnthropic()
    except Exception as exc:  # pragma: no cover - depends on local env
        logger.warning("Could not create Anthropic client: %s", exc)
        return None


def build_core(*, vlog=None, approval=None, speaker=None, llm="auto") -> VIYONCore:
    """Construct VIYON CORE and all 12 agents from config/.env.

    Collaborators are injectable so tests can supply a temp logger, a silent
    speaker, a recording approval gate, and a None LLM.
    """
    config.load_env()
    vlog = vlog or VLogger()
    approval = approval or ApprovalGate()  # console front-end
    speaker = speaker or Speaker()
    if llm == "auto":
        llm = make_llm()

    tools = SimpleNamespace(
        mac_control=mac_control, file_ops=file_ops, terminal=terminal, web=web
    )
    deps = dict(llm=llm, tools=tools, log=vlog, approval=approval, config=config)
    agents = {name: cls(**deps) for name, cls in _AGENT_CLASSES.items()}

    return VIYONCore(
        agents=agents,
        listener=Listener(),
        speaker=speaker,
        router=Router(client=llm),
        wake_word=WakeWord(),
        approval=approval,
        memory=SessionMemory(max_turns=10),
        logger_=vlog,
    )


async def _start_hud(core: VIYONCore):
    """Run the HUD server (uvicorn) as a coroutine; returns the server object."""
    import uvicorn

    from hud.server import create_app

    host = config.get("hud", "host", "127.0.0.1")
    port = int(config.get("hud", "port", 8765))
    server = uvicorn.Server(uvicorn.Config(create_app(core), host=host, port=port, log_level="warning"))
    asyncio.ensure_future(server.serve())
    await asyncio.sleep(0.3)  # let it bind
    print(f"  HUD ····· http://{host}:{port}/boot")
    return server


async def _text_loop(core: VIYONCore) -> None:
    """Read typed commands from stdin and route them through handle()."""
    print("Type commands (Ctrl-D or 'exit' to quit).")
    while True:
        try:
            line = (await asyncio.to_thread(input, "viyon› ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        reply = await core.handle(line)
        if reply:
            print(f"  ◈ {reply}")


async def _run_one_agent(core: VIYONCore, name: str, task: str) -> None:
    """Run a single agent once and print its result (for testing)."""
    agent = core.agents.get(name.upper())
    if not agent:
        print(f"Unknown agent {name!r}. Options: {', '.join(core.agents)}")
        return
    try:
        result = await agent.run(task, {"history": []})
    except Exception as exc:
        print(f"[{name.upper()}] failed: {type(exc).__name__}: {exc}")
        return
    print(f"[{result.agent}] ok={result.ok}\n{result.summary}")
    if result.detail:
        print("---\n" + result.detail)
    if result.artifacts:
        print("artifacts: " + ", ".join(result.artifacts))


async def amain(args) -> None:
    vlog = VLogger()
    approval = ApprovalGate()
    core = build_core(vlog=vlog, approval=approval)

    # One-shot agent run.
    if args.agent:
        try:
            await _run_one_agent(core, args.agent[0], args.agent[1])
        finally:
            vlog.close()
        return

    hud_server = None
    try:
        print("◈ VIYON booting…")
        if not args.no_hud:
            hud_server = await _start_hud(core)
        boot_line = "VIYON online. All systems nominal."
        print(f"◈ {boot_line}")
        with contextlib.suppress(Exception):
            await core.speaker.say(boot_line)

        if args.text:
            await _text_loop(core)
        else:
            await core.run_forever()
    finally:
        if hud_server is not None:
            hud_server.should_exit = True
        vlog.close()
        events.emit_reset()
        print("\n◈ VIYON offline.")


def main() -> None:
    parser = argparse.ArgumentParser(description="VIYON — voice-intelligent operational network")
    parser.add_argument("--text", action="store_true", help="type commands instead of voice")
    parser.add_argument("--no-hud", action="store_true", help="don't start the HUD server")
    parser.add_argument(
        "--agent", nargs=2, metavar=("NAME", "TASK"), help="run one agent once and exit"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\n◈ VIYON offline.")


if __name__ == "__main__":
    main()
