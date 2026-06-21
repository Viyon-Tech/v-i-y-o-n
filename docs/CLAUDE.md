# VIYON — Voice-Intelligent Operational Network

## What this is
VIYON is a local-first, voice-controlled, multi-agent AI operating layer for macOS (Apple M4).
A central orchestrator (VIYON CORE) receives every command, routes it to one or more named
sub-agents, runs them — in parallel when possible — and speaks the merged result back.
It opens with a Stark-style HUD ("the VIYON HUD").

## Absolute principles (never violate)
1. VIYON NEVER acts autonomously. Every action runs only on an explicit user command.
2. Any destructive or state-changing action (delete, overwrite, git push, kill process,
   send email, run shell) MUST pass through the Approval Gate and get a verbal/typed "yes".
3. Everything is logged to ~/.viyon/logs/ (SQLite + JSON). Every command is replayable.
4. The user can always say "go back" / "undo last" — implement reversible actions where possible.
5. Secrets live in .env only. Never hardcode keys. Never print keys to logs.

## The agents
- VIYON CORE — orchestrator: STT → intent → route → parallel run → merge → TTS → log
- NOVA   💻 coding (writes/edits/reviews/debugs, git) — uses claude-code MCP
- FORGE  🏗️ project scaffolding (full project structures, opens in editor)
- SHIELD 🛡️ cybersecurity (vuln scans, dependency audit, network analysis)
- PULSE  🔬 research (web search, paper/URL summarization)
- ATLAS  🖥️ Mac control (apps, files, windows, clipboard, system settings)
- ECHO   📡 communications (Mail, Calendar, Messages — drafts before sending)
- NEXUS  📊 data & analytics (CSV/Excel/JSON, charts, reports)
- VISTA  🎨 design (Canva MCP, Figma MCP, mockups)
- TEMPO  📋 PA & scheduler (tasks, reminders, planning) — Notion MCP
- SAGE   💬 knowledge & chat (explanations, tutoring, knowledge base) — Notion MCP
- LUNA   🌙 emotional support (wellness, journaling) — warm, validates first — Notion MCP
- GHOST  👁️ system monitor (CPU/mem/disk/processes)

## Tech stack (do not substitute without asking)
- Language: Python 3.11+, fully async (asyncio)
- Package manager: uv
- STT: mlx-whisper (Apple Silicon native, runs on M4 Neural Engine) — fallback: faster-whisper
- Wake word: pvporcupine (Picovoice), wake word "VIYON"
- TTS: ElevenLabs API — fallback: macOS `say`
- LLM brain: Anthropic Python SDK (anthropic), model "claude-sonnet-4-20250514"
- Coding engine for NOVA/FORGE: claude-code MCP server (codeaashu/claude-code, mcp-server/)
- Mac control: AppleScript via osascript, pyobjc, pyautogui
- System monitor: psutil
- Logs: sqlite3 (stdlib) + JSON files
- HUD: local web app (HTML/CSS/Canvas/JS) served by FastAPI + WebSocket for live data
- Config: config.yaml + python-dotenv

## Repo layout (target)
viyon/
  main.py
  config.yaml
  .env.example
  pyproject.toml
  core/        orchestrator.py listener.py wake_word.py speaker.py router.py parallel.py memory.py
  agents/      base_agent.py nova.py forge.py shield.py pulse.py atlas.py echo.py
               nexus.py vista.py tempo.py sage.py luna.py ghost.py
  tools/       mac_control.py file_ops.py terminal.py web.py approval.py
  integrations/ claude_code_mcp.py canva_mcp.py notion_mcp.py
  logs/        logger.py history.py
  hud/         server.py  static/(index.html style.css hud.js)
  tests/       test_router.py test_approval.py test_agents.py

## Coding conventions
- Type hints everywhere. Docstrings on every public function/class.
- Each agent subclasses BaseAgent and implements async def run(self, task: str, ctx: dict) -> AgentResult.
- No agent calls another agent directly — only CORE coordinates. Agents may *request* a handoff
  by returning AgentResult.handoff = "NOVA" (CORE decides).
- All shell/file/network side effects go through tools/ — agents never call subprocess directly.
- Use structured logging via logs/logger.py. Never use bare print() in library code.
- Keep functions small. Fail loudly with clear messages. Validate inputs at boundaries.

## Build order (follow strictly)
Phase 0 scaffold → 1 logging+approval → 2 voice I/O → 3 CORE+router → 4 base_agent →
5 ATLAS+GHOST → 6 NOVA+FORGE (claude-code MCP) → 7 PULSE+SAGE → 8 ECHO+TEMPO+NEXUS →
9 VISTA+LUNA → 10 HUD → 11 wire main.py + end-to-end test.

## Definition of done for any module
Compiles, has a docstring, has at least one test or a __main__ smoke test, is logged,
and (if it has side effects) is gated by the Approval Gate.