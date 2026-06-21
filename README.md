# VIYON — Voice-Intelligent Operational Network

## What this is
VIYON is a local-first, voice-controlled, multi-agent AI operating layer for macOS (Apple M4).
A central orchestrator (VIYON CORE) receives every command, routes it to one or more named
sub-agents, runs them — in parallel when possible — and speaks the merged result back.
It opens with a Stark-style HUD ("the VIYON HUD").

> Design, principles, agent roster, and build order live in [`docs/CLAUDE.md`](docs/CLAUDE.md).
> **Status:** Phase 0 — project scaffold only. No logic implemented yet.

## Requirements
- macOS on Apple Silicon (built/tested on an M4)
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management

## Setup

```bash
# 1. Create the virtual environment (Python 3.11)
uv venv --python 3.11

# 2. Activate it
source .venv/bin/activate

# 3. Install the project plus dev tooling (editable)
uv pip install -e ".[dev]"

# 4. Configure secrets — copy the template and fill in your keys
cp .env.example .env
#   then edit .env: ANTHROPIC_API_KEY, ELEVENLABS_API_KEY,
#   PICOVOICE_ACCESS_KEY, BRAVE_API_KEY
```

Non-secret settings (wake word, models, enabled agents, HUD host/port, logging)
live in [`config.yaml`](config.yaml). Secrets live only in `.env` — never commit it.

## Verify the scaffold

```bash
python -c "import core, agents, tools, integrations, logs"
```

## Run (once implemented)

```bash
python main.py        # or: viyon
```

## Project layout

```
viyon/
  main.py            entry point
  config.yaml        non-secret config
  .env.example       secret key template
  pyproject.toml     deps + tooling (uv)
  core/              orchestrator, listener, wake_word, speaker, router, parallel, memory
  agents/            base_agent + NOVA FORGE SHIELD PULSE ATLAS ECHO NEXUS VISTA TEMPO SAGE LUNA GHOST
  tools/             mac_control, file_ops, terminal, web, approval
  integrations/      claude_code_mcp, canva_mcp, notion_mcp
  logs/              logger, history
  hud/               server + static/ (index.html, style.css, hud.js)
  tests/             test_router, test_approval, test_agents
```

## Tests

```bash
pytest
```
