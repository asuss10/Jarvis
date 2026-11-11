"""Tool registry: the single choke point between model requests and Python execution.

The model only ever *requests* a tool by name. Everything that happens next
(lookup, argument validation, confirmation, execution, logging) happens here,
in Python, under our control.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jarvis.core.llm import ToolCall

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], str]
ConfirmFn = Callable[[str, dict[str, Any]], bool]

_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "boolean": bool,
    "number": (int, float),
}
MAX_LOG_VALUE = 200


class ToolError(Exception):
    """Raised by a handler for an expected problem. The message is shown to the model."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict[str, dict[str, Any]]
    required: tuple[str, ...]
    handler: Handler
    requires_confirmation: bool = False

    def schema(self) -> dict[str, Any]:
        """The JSON Schema description the model sees."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                },
            },
        }


def _validate_arguments(tool: Tool, arguments: dict[str, Any]) -> str | None:
    """Return a problem description, or None if the arguments are acceptable."""
    for name in tool.required:
        if name not in arguments:
            return f"missing required argument '{name}'"
    for name, value in arguments.items():
        spec = tool.properties.get(name)
        if spec is None:
            return f"unexpected argument '{name}'"
        json_type = spec["type"]
        # bool is a subclass of int in Python; don't accept True as an integer.
        if isinstance(value, bool) and json_type != "boolean":
            return f"argument '{name}' must be of type {json_type}"
        if not isinstance(value, _PY_TYPES[json_type]):
            return f"argument '{name}' must be of type {json_type}"
    return None


def _summarize(arguments: dict[str, Any]) -> str:
    text = repr(arguments)
    return text if len(text) <= MAX_LOG_VALUE else text[:MAX_LOG_VALUE] + "..."


def _deny_all(tool_name: str, arguments: dict[str, Any]) -> bool:
    return False  # fail closed: without an explicit confirm function, nothing sensitive runs


class ToolRegistry:
    def __init__(self, confirm: ConfirmFn = _deny_all) -> None:
        self._tools: dict[str, Tool] = {}
        self._confirm = confirm

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> str:
        """Run a model-requested tool call. Never raises; always returns text for the model."""
        tool = self._tools.get(call.name)
        if tool is None:
            logger.warning("Model requested unknown tool %r", call.name)
            return f"Error: unknown tool '{call.name}'."

        problem = _validate_arguments(tool, call.arguments)
        if problem:
            logger.warning("Rejected call to %s: %s", call.name, problem)
            return f"Error: {problem}."

        if tool.requires_confirmation and not self._confirm(call.name, call.arguments):
            logger.info("User declined tool call: %s", call.name)
            return "Error: the user declined this action."

        logger.info("Tool call: %s%s", call.name, _summarize(call.arguments))
        try:
            return tool.handler(call.arguments)
        except ToolError as exc:
            logger.info("Tool %s reported an error: %s", call.name, exc)
            return f"Error: {exc}"
        except Exception:
            # Details go to the log, not to the model.
            logger.exception("Tool %s crashed", call.name)
            return "Error: the tool failed unexpectedly."