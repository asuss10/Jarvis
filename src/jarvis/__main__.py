"""Command-line chat interface. Run with: python -m jarvis"""

import logging
from typing import Any

from jarvis.config import load_settings
from jarvis.core.agent import Agent
from jarvis.core.llm import LLMError
from jarvis.core.ollama_provider import OllamaProvider
from jarvis.core.prompts import build_system_prompt
from jarvis.logging_config import setup_logging
from jarvis.memory.database import connect
from jarvis.memory.notes import NotesStore
from jarvis.tools.notes_tools import build_note_tools
from jarvis.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def ask_confirmation(tool_name: str, arguments: dict[str, Any]) -> bool:
    answer = input(f"  Allow {tool_name}({arguments})? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    conn = connect(settings.data_dir / "jarvis.db")
    registry = ToolRegistry(confirm=ask_confirmation)
    for tool in build_note_tools(NotesStore(conn)):
        registry.register(tool)

    provider = OllamaProvider(
        model=settings.model, host=settings.ollama_host, num_ctx=settings.num_ctx
    )
    agent = Agent(provider, registry, build_system_prompt())

    print(f"JARVIS v0.1 ({settings.model}). Type 'exit' to quit.\n")
    try:
        while True:
            try:
                user_text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break
            try:
                print(f"jarvis> {agent.respond(user_text)}\n")
            except LLMError as exc:
                print(f"[error] {exc}\n")
    finally:
        conn.close()
        print("\nGoodbye.")


if __name__ == "__main__":
    main()