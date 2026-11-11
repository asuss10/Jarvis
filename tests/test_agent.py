import pytest

from jarvis.core.agent import GIVE_UP_MESSAGE, Agent
from jarvis.core.llm import LLMError, LLMResponse, ToolCall
from jarvis.memory.database import connect
from jarvis.memory.notes import NotesStore
from jarvis.tools.notes_tools import build_note_tools
from jarvis.tools.registry import ToolRegistry


class FakeProvider:
    """Plays back scripted responses and records what the agent sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(list(messages))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def store():
    conn = connect(":memory:")
    yield NotesStore(conn)
    conn.close()


@pytest.fixture
def registry(store):
    reg = ToolRegistry()
    for tool in build_note_tools(store):
        reg.register(tool)
    return reg


def make_agent(provider, registry, max_tool_rounds=5):
    return Agent(provider, registry, system_prompt="test prompt", max_tool_rounds=max_tool_rounds)


def test_plain_answer_needs_no_tools(registry):
    provider = FakeProvider([LLMResponse(text="Paris.")])
    agent = make_agent(provider, registry)
    assert agent.respond("Capital of France?") == "Paris."
    assert len(provider.calls) == 1


def test_tool_round_trip_saves_note(registry, store):
    provider = FakeProvider([
        LLMResponse(text="", tool_calls=[ToolCall("save_note", {"text": "Dentist Friday 10:00"})]),
        LLMResponse(text="Done, I saved that."),
    ])
    agent = make_agent(provider, registry)

    assert agent.respond("Remember my dentist appointment") == "Done, I saved that."
    assert store.list_recent()[0].text == "Dentist Friday 10:00"

    # The second model call must have seen the tool result.
    last_message = provider.calls[1][-1]
    assert last_message["role"] == "tool"
    assert "Saved note" in last_message["content"]


def test_unknown_tool_request_does_not_crash(registry):
    provider = FakeProvider([
        LLMResponse(text="", tool_calls=[ToolCall("run_shell", {"cmd": "format C:"})]),
        LLMResponse(text="I can't do that."),
    ])
    agent = make_agent(provider, registry)
    assert agent.respond("do something bad") == "I can't do that."
    assert provider.calls[1][-1]["content"].startswith("Error: unknown tool")


def test_gives_up_after_max_rounds(registry):
    looping = LLMResponse(text="", tool_calls=[ToolCall("list_recent_notes", {})])
    provider = FakeProvider([looping] * 10)
    agent = make_agent(provider, registry, max_tool_rounds=3)
    assert agent.respond("hello") == GIVE_UP_MESSAGE
    assert len(provider.calls) == 3


def test_llm_error_rolls_back_history(registry):
    provider = FakeProvider([LLMError("Cannot reach Ollama.")])
    agent = make_agent(provider, registry)
    with pytest.raises(LLMError):
        agent.respond("hello")
    assert agent.history == []


def test_history_carries_across_turns(registry):
    provider = FakeProvider([LLMResponse(text="Hi."), LLMResponse(text="Fine.")])
    agent = make_agent(provider, registry)
    agent.respond("hello")
    agent.respond("how are you?")
    sent_second = [m["content"] for m in provider.calls[1] if m["role"] == "user"]
    assert sent_second == ["hello", "how are you?"]