# JARVIS

A personal, voice-first AI assistant, built incrementally and running entirely on
local hardware. Notes and conversations stay on this machine.

## Status

**v0.1 (Phase 1):** text chat with a local model, tool calling, SQLite notes,
persistent conversation history.

Planned: voice interaction, computer control, email/calendar, long-term memory,
smart home. See the roadmap in this project's design notes.

## Architecture

    User input -> Agent -> LLMProvider (Ollama) -> tool request
                     |                                  |
                     |<-------- result <---- ToolRegistry (validates, confirms, logs)
                     v
                  reply

- `core/`: agent loop, provider interface, prompts
- `tools/`: tool registry (the only path from model requests to execution) and tools
- `memory/`: SQLite stores for notes and conversation history
- `security/`: reserved for permission policies (arrives with the first sensitive tool)

The model only *requests* tools by name. Python validates arguments, asks for
confirmation on sensitive tools, executes, and logs every call.

## Requirements

- Python 3.12
- [Ollama](https://ollama.com) with a tool-capable model (developed with `gpt-oss:20b`)

## Setup

    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    pip install -e .
    copy .env.example .env
    ollama pull gpt-oss:20b

## Run

    python -m jarvis

Commands: `/clear` forgets the conversation (notes are kept), `exit` quits.

## Test

    pytest -v
    python scripts/smoke_test.py   # checks real-model tool-calling behaviour

## Privacy

`.env` and everything in `data/` (notes, conversation history) are git-ignored
and never leave the machine.