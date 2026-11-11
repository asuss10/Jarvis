"""The agent loop: model <-> tools, with conversation history."""

import logging
from typing import Any

from jarvis.core.llm import LLMError, LLMProvider, LLMResponse
from jarvis.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

GIVE_UP_MESSAGE = "I couldn't complete that request. Please try rephrasing it."


def _assistant_message(response: LLMResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.text}
    if response.tool_calls:
        message["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in response.tool_calls
        ]
    return message


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        system_prompt: str,
        max_tool_rounds: int = 5,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._system_prompt = system_prompt
        self._max_tool_rounds = max_tool_rounds
        self._messages: list[dict[str, Any]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def respond(self, user_text: str) -> str:
        checkpoint = len(self._messages)
        self._messages.append({"role": "user", "content": user_text})
        try:
            return self._run_turn()
        except LLMError:
            # Don't leave a half-finished turn in the history.
            del self._messages[checkpoint:]
            raise

    def _run_turn(self) -> str:
        for round_number in range(1, self._max_tool_rounds + 1):
            messages = [{"role": "system", "content": self._system_prompt}, *self._messages]
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