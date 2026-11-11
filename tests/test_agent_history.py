import pytest

from jarvis.core.agent import Agent
from jarvis.core.llm import LLMError, LLMResponse
from jarvis.memory.database import connect
from jarvis.memory.history import HistoryStore
from jarvis.tools.registry import ToolRegistry


class FakeProvider:
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
def history():
    conn = connect(":memory:")
    yield HistoryStore(conn)
    conn.close()


def test_old_turns_are_trimmed_on_turn_boundaries():
    provider = FakeProvider([LLMResponse(text="a" * 40) for _ in range(4)])
    agent = Agent(provider, ToolRegistry(), "sys", max_history_chars=100)
    for i in range(4):
        agent.respond(f"question {i}")

    sent = provider.calls[-1][1:]  # skip the system message
    assert sent[0]["role"] == "user"
    assert sent[-1]["content"] == "question 3"
    assert len(sent) == 3  # one older turn plus the current question


def test_latest_turn_is_kept_even_if_over_budget():
    provider = FakeProvider([LLMResponse(text="ok")])
    agent = Agent(provider, ToolRegistry(), "sys", max_history_chars=1)
    agent.respond("a very long question")
    assert provider.calls[0][-1]["content"] == "a very long question"


def test_completed_turn_is_persisted(history):
    agent = Agent(FakeProvider([LLMResponse(text="Hi.")]), ToolRegistry(), "sys", history=history)
    agent.respond("hello")
    assert [m["role"] for m in history.load_recent()] == ["user", "assistant"]


def test_failed_turn_is_not_persisted(history):
    agent = Agent(FakeProvider([LLMError("down")]), ToolRegistry(), "sys", history=history)
    with pytest.raises(LLMError):
        agent.respond("hello")
    assert history.load_recent() == []


def test_new_agent_resumes_saved_history(history):
    first = Agent(FakeProvider([LLMResponse(text="Hi.")]), ToolRegistry(), "sys", history=history)
    first.respond("hello")

    provider = FakeProvider([LLMResponse(text="Fine.")])
    second = Agent(provider, ToolRegistry(), "sys", history=history)
    second.respond("how are you?")

    sent = [m["content"] for m in provider.calls[0] if m["role"] == "user"]
    assert sent == ["hello", "how are you?"]