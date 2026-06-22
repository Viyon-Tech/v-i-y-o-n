# VIYON — Voice-Intelligent Operational Network

## What this is
VIYON is a local-first, voice-controlled, multi-agent AI operating layer for macOS (Apple
Silicon). A central orchestrator (VIYON CORE) receives every command, routes it to one or more
named sub-agents, runs them — in parallel when possible — and speaks the merged result back.
It opens with a Stark-style HUD ("the VIYON HUD").

> Design, principles, agent roster, and build order live in [`docs/CLAUDE.md`](docs/CLAUDE.md).

**Absolute rule:** VIYON never acts autonomously. Every destructive or state-changing action
(delete, overwrite, send, run shell, git push, kill process) passes through the Approval Gate and
needs an explicit "yes". Everything is logged to `~/.viyon/logs/` (SQLite + JSON) and is replayable.

## Requirements
- macOS on Apple Silicon (built/tested on an M-series)
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- Node.js (only for NOVA/FORGE's claude-code MCP server — see below)

## Setup (uv)

```bash
# 1. Create the virtual environment (Python 3.11)
uv venv --python 3.11

# 2. Activate it
source .venv/bin/activate

# 3. Install the project plus dev tooling (editable)
uv pip install -e ".[dev]"

# 4. Configure secrets — copy the template and fill in your keys
cp .env.example .env
#   ANTHROPIC_API_KEY    — the LLM brain (router + agents). Without it, VIYON
#                          falls back to keyword routing and skips LLM features.
#   ELEVENLABS_API_KEY   — voice output (falls back to macOS `say`)
#   PICOVOICE_ACCESS_KEY — wake-word "VIYON" (falls back to press-Enter-to-talk)
#   BRAVE_API_KEY        — web research for PULSE (or set SERPER_API_KEY)
```

Non-secret settings (wake word, model, enabled agents, HUD host/port, coding MCP URL, logging)
live in [`config.yaml`](config.yaml). Secrets live only in `.env` — never commit it.

> **Note on the model:** `config.yaml` pins `llm.model`. If it's an older/retired model id, update
> it to a current one (e.g. `claude-sonnet-4-6`).

## Run

```bash
python main.py                      # voice: say "VIYON" → speak a command. Opens the HUD too.
python main.py --text               # type commands through the same pipeline (no mic needed)
python main.py --no-hud             # skip the HUD server
python main.py --agent GHOST "snapshot my system"   # run one agent once and exit (testing)
```

On boot VIYON prints the HUD URL and says "VIYON online". Ctrl-C shuts down cleanly (stops the
HUD, closes the database).

## The HUD

A Stark/JARVIS-style web app: a multi-ring reactor with VIYON CORE at the center and the 12 agents
as orbiting nodes, plus live system vitals, a clock, and a command log. It connects to the backend
over WebSocket and reacts in real time — agent nodes pulse cyan while working, the core breathes
while listening, and the ring flashes amber during an approval prompt.

```bash
python main.py                      # the HUD starts automatically
# then open the full-screen boot view:
open http://127.0.0.1:8765/boot
```

- **Mock mode** (no backend needed): `http://127.0.0.1:8765/boot?mock=1` drives the HUD with
  synthetic data so you can see it animate even before wiring real data.
- Run the HUD on its own: `uvicorn hud.server:app` (or `python -m hud.server`).
- Type a command into the HUD's command bar to route it through the orchestrator (`POST /command`).

## Coding engine — claude-code MCP (for NOVA & FORGE)

NOVA (coding) and FORGE (scaffolding) read/edit/run your project through the
[`codeaashu/claude-code`](https://github.com/codeaashu/claude-code) MCP server.

```bash
# one-time setup
git clone https://github.com/codeaashu/claude-code ~/viyon/claude-code
cd ~/viyon/claude-code/mcp-server
npm install && npm run build

# start it (or let VIYON launch it via integrations.claude_code_mcp.start_server)
node ~/viyon/claude-code/mcp-server/dist/index.js
```

It listens at `coding.mcp_url` in `config.yaml` (default `http://localhost:3000/mcp`). If the
server isn't running, NOVA/FORGE **degrade gracefully** to plain Claude code generation (no file
tools) and tell you.

## Voice commands by agent

Say "VIYON" (or press Enter in fallback mode), then speak. Examples:

| Agent | What it does | Try saying |
|-------|--------------|------------|
| **GHOST** 👁️ | System monitor (CPU/mem/disk/battery, top processes) | "what's eating my CPU", "show top processes", "kill process 1234" |
| **ATLAS** 🖥️ | Mac control (apps, files, clipboard, settings) | "open Safari", "turn on dark mode", "what's running", "read the file ~/notes.txt" |
| **NOVA** 💻 | Coding — read/edit/debug, git (claude-code MCP) | "fix the bug in app.py", "add tests for the router", "explain the auth module" |
| **FORGE** 🏗️ | Project scaffolding | "scaffold a FastAPI project with JWT auth", "create a Next.js dashboard" |
| **SHIELD** 🛡️ | Security — secret scans, dependency audits, ports | "scan my project for secrets", "audit my dependencies", "what's listening on the network" |
| **PULSE** 🔬 | Research — web search + summarize | "research vector databases", "look up the latest on RAG" |
| **ECHO** 📡 | Mail / Calendar / Messages (always drafts first) | "check my unread mail", "what's on my calendar", "email Bob about lunch" |
| **NEXUS** 📊 | Data — profile/chart/report CSV·Excel·JSON | "profile sales.csv", "chart revenue.csv", "build a dashboard for data.csv" (→ NOVA) |
| **VISTA** 🎨 | Design via Canva/Figma | "make a poster for the hackathon", "design an Instagram post" |
| **TEMPO** 📋 | PA & scheduler (tasks, reminders, plan day) | "plan my day", "add a task to write the spec", "remind me to call at 3pm" |
| **SAGE** 💬 | Knowledge & tutoring | "explain how transformers work", "what is recursion" |
| **LUNA** 🌙 | Emotional support & private journaling | "I'm stressed about my deadline", "privately journal that…" |

VIYON routes to one or more agents automatically (in parallel when independent), e.g.
*"research RAG and scaffold a demo"* → PULSE + FORGE together. Anything destructive prompts for
approval first. Say *"privately…"* / *"off the record"* to keep a command's content out of the log.

## Project layout

```
viyon/
  main.py            entry point (boots CORE + 12 agents + HUD + voice loop)
  config.yaml        non-secret config
  .env.example       secret key template
  pyproject.toml     deps + tooling (uv)
  core/              orchestrator, router, parallel, memory, events, listener, wake_word, speaker, config
  agents/            base_agent + NOVA FORGE SHIELD PULSE ATLAS ECHO NEXUS VISTA TEMPO SAGE LUNA GHOST
  tools/             mac_control, file_ops, terminal, web, approval
  integrations/      claude_code_mcp, canva_mcp, notion_mcp
  logs/              logger (SQLite + JSON), history
  hud/               server + static/ (index.html, style.css, hud.js)
  tests/             pytest suite
```

## Tests

```bash
pytest
```
