"""Provider-neutral types. The agent depends on these, not on any vendor SDK."""

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    """Raised when the model backend is unreachable or returns an error."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...