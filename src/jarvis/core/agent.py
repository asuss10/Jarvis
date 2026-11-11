"""The agent loop: model <-> tools, with persistent history and context trimming."""

import logging
from typing import Any

from jarvis.core.llm import LLMError, LLMProvider, LLMResponse
from jarvis.memory.history import HistoryStore
from jarvis.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

GIVE_UP_MESSAGE = "I couldn't complete that request. Please try rephrasing it."
DEFAULT_MAX_HISTORY_CHARS = 12000


def _assistant_message(response: LLMResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.text}
    if response.tool_calls:
        message["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in response.tool_calls
        ]
    return message


def _size(message: dict[str, Any]) -> int:
    return len(message.get("content") or "") + len(str(message.get("tool_calls", "")))


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        system_prompt: str,
        max_tool_rounds: int = 5,
        history: HistoryStore | None = None,
        max_history_chars: int = DEFAULT_MAX_HISTORY_CHARS,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._system_prompt = system_prompt
        self._max_tool_rounds = max_tool_rounds
        self._history = history
        self._max_history_chars = max_history_chars
        self._messages: list[dict[str, Any]] = history.load_recent() if history else []
        if self._messages:
            logger.info("Resumed %d messages of saved history", len(self._messages))

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def reset(self) -> None:
        """Forget the conversation (notes are untouched)."""
        self._messages = []
        if self._history:
            self._history.clear()

    def respond(self, user_text: str) -> str:
        checkpoint = len(self._messages)
        self._messages.append({"role": "user", "content": user_text})
        try:
            reply = self._run_turn()
        except LLMError:
            # Don't leave a half-finished turn in memory (nothing was saved yet).
            del self._messages[checkpoint:]
            raise
        if self._history:
            self._history.append_many(self._messages[checkpoint:])
        return reply

    def _window(self) -> list[dict[str, Any]]:
        """The newest whole turns that fit the budget. Always starts at a user message."""
        turns: list[list[dict[str, Any]]] = []
        for message in self._messages:
            if message["role"] == "user" or not turns:
                turns.append([message])
            else:
                turns[-1].append(message)

        selected: list[list[dict[str, Any]]] = []
        used = 0
        for turn in reversed(turns):
            size = sum(_size(m) for m in turn)
            if selected and used + size > self._max_history_chars:
                break  # the newest turn is always kept, even if it alone is too big
            selected.insert(0, turn)
            used += size
        return [m for turn in selected for m in turn]

    def _run_turn(self) -> str:
        for round_number in range(1, self._max_tool_rounds + 1):
            messages = [{"role": "system", "content": self._system_prompt}, *self._window()]
            response = self._provider.chat(messages, tools=self._registry.schemas())
            self._messages.append(_assistant_message(response))

            if not response.tool_calls:
                return response.text.strip() or "(no response)"

            logger.debug("Round %d: %d tool call(s)", round_number, len(response.tool_calls))
            for call in response.tool_calls:
                result = self._registry.execute(call)
                self._messages.append(
                    {"role": "tool", "tool_name": call.name, "content": result}
                )

        logger.warning("Gave up after %d tool rounds", self._max_tool_rounds)
        return GIVE_UP_MESSAGE