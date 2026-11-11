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
from jarvis.memory.history import HistoryStore
from jarvis.memory.notes import NotesStore
from jarvis.tools.notes_tools import build_note_tools
from jarvis.tools.registry import ToolRegistry
from jarvis.voice.tts import Speaker, TTSError, ensure_voice

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
        model=settings.model,
        host=settings.ollama_host,
        num_ctx=settings.num_ctx,
        keep_alive=settings.keep_alive,
    )
    agent = Agent(
        provider,
        registry,
        build_system_prompt(),
        history=HistoryStore(conn),
        # Roughly half the context window; the rest is for system prompt, tools and reply.
        max_history_chars=settings.num_ctx * 2,
    )

    speaker: Speaker | None = None  # created only when voice is switched on

    print(
        f"JARVIS v0.1 ({settings.model}). "
        "Commands: /voice (speech on/off), /clear (forget chat), exit.\n"
    )
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
            if user_text.lower() == "/clear":
                agent.reset()
                print("jarvis> Conversation cleared. Your notes are untouched.\n")
                continue
            if user_text.lower() == "/voice":
                if speaker is not None:
                    speaker.unload()  # free the voice model's memory
                    speaker = None
                    print("jarvis> Voice off.\n")
                else:
                    try:
                        model_path = ensure_voice(settings.voice, settings.data_dir / "voices")
                        candidate = Speaker(model_path)
                        candidate.speak("Voice enabled.")
                        speaker = candidate
                        print(f"jarvis> Voice on ({settings.voice}).\n")
                    except TTSError as exc:
                        print(f"[voice error] {exc}\n")
                continue

            try:
                reply = agent.respond(user_text)
            except LLMError as exc:
                print(f"[error] {exc}\n")
                continue
            print(f"jarvis> {reply}\n")

            if speaker is not None:
                try:
                    speaker.speak(reply)
                except TTSError as exc:
                    print(f"[voice error] {exc}. Voice switched off.\n")
                    speaker.unload()
                    speaker = None
    finally:
        conn.close()
        print("\nGoodbye.")


if __name__ == "__main__":
    main()