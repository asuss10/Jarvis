import pytest

from jarvis.core.llm import ToolCall
from jarvis.tools.registry import Tool, ToolError, ToolRegistry


def make_tool(handler=None, requires_confirmation=False) -> Tool:
    return Tool(
        name="echo",
        description="Echo text back.",
        properties={"text": {"type": "string"}, "count": {"type": "integer"}},
        required=("text",),
        handler=handler or (lambda args: f"echo: {args['text']}"),
        requires_confirmation=requires_confirmation,
    )


def registry_with(tool: Tool, **kwargs) -> ToolRegistry:
    registry = ToolRegistry(**kwargs)
    registry.register(tool)
    return registry


def test_valid_call_runs_handler():
    registry = registry_with(make_tool())
    assert registry.execute(ToolCall("echo", {"text": "hi"})) == "echo: hi"


def test_unknown_tool_returns_error():
    registry = registry_with(make_tool())
    assert registry.execute(ToolCall("delete_everything", {})).startswith("Error: unknown tool")


def test_missing_required_argument():
    registry = registry_with(make_tool())
    assert "missing required argument 'text'" in registry.execute(ToolCall("echo", {}))


def test_unexpected_argument_rejected():
    registry = registry_with(make_tool())
    result = registry.execute(ToolCall("echo", {"text": "hi", "path": "C:/"}))
    assert "unexpected argument 'path'" in result


def test_wrong_type_rejected():
    registry = registry_with(make_tool())
    assert "must be of type string" in registry.execute(ToolCall("echo", {"text": 5}))


def test_bool_is_not_accepted_as_integer():
    registry = registry_with(make_tool())
    result = registry.execute(ToolCall("echo", {"text": "hi", "count": True}))
    assert "must be of type integer" in result


def test_tool_error_message_is_passed_to_model():
    def handler(args):
        raise ToolError("that note is too long")

    registry = registry_with(make_tool(handler))
    assert registry.execute(ToolCall("echo", {"text": "x"})) == "Error: that note is too long"


def test_unexpected_exception_does_not_leak_details():
    def handler(args):
        raise RuntimeError("secret internal detail")

    registry = registry_with(make_tool(handler))
    result = registry.execute(ToolCall("echo", {"text": "x"}))
    assert "secret internal detail" not in result
    assert result.startswith("Error:")


def test_confirmation_denied_by_default():
    registry = registry_with(make_tool(requires_confirmation=True))
    result = registry.execute(ToolCall("echo", {"text": "x"}))
    assert "declined" in result


def test_confirmation_granted():
    registry = registry_with(
        make_tool(requires_confirmation=True), confirm=lambda name, args: True
    )
    assert registry.execute(ToolCall("echo", {"text": "x"})) == "echo: x"


def test_duplicate_registration_rejected():
    registry = registry_with(make_tool())
    with pytest.raises(ValueError):
        registry.register(make_tool())


def test_schema_shape():
    schema = make_tool().schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"]["required"] == ["text"]