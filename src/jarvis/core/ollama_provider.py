"""LLMProvider implementation backed by a local Ollama server."""

import logging
from typing import Any

import ollama

from jarvis.core.llm import LLMError, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class OllamaProvider:
    def __init__(self, model: str, host: str, num_ctx: int) -> None:
        self._model = model
        self._num_ctx = num_ctx
        self._client = ollama.Client(host=host)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        logger.debug("Calling %s with %d messages", self._model, len(messages))
        try:
            response = self._client.chat(
                model=self._model,
                messages=messages,
                tools=tools,
                options={"num_ctx": self._num_ctx},
            )
        except ConnectionError as exc:
            raise LLMError("Cannot reach Ollama. Is it running?") from exc
        except ollama.ResponseError as exc:
            raise LLMError(f"Ollama error: {exc}") from exc

        tool_calls = [
            ToolCall(name=call.function.name, arguments=dict(call.function.arguments))
            for call in (response.message.tool_calls or [])
        ]
        return LLMResponse(text=response.message.content or "", tool_calls=tool_calls)