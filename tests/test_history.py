import pytest

from jarvis.memory.database import connect
from jarvis.memory.history import HistoryStore


@pytest.fixture
def history():
    conn = connect(":memory:")
    yield HistoryStore(conn)
    conn.close()


def test_tool_calls_roundtrip(history):
    messages = [
        {"role": "user", "content": "remember x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "save_note", "arguments": {"text": "x"}}}],
        },
        {"role": "tool", "tool_name": "save_note", "content": "Saved note #1."},
        {"role": "assistant", "content": "Done."},
    ]
    history.append_many(messages)
    assert history.load_recent() == messages


def test_leading_non_user_messages_are_dropped(history):
    history.append_many([
        {"role": "tool", "tool_name": "x", "content": "orphan"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert [m["role"] for m in history.load_recent()] == ["user", "assistant"]


def test_limit_returns_newest_messages(history):
    for i in range(3):
        history.append_many([
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ])
    assert [m["content"] for m in history.load_recent(limit=2)] == ["q2", "a2"]


def test_clear_removes_everything(history):
    history.append_many([{"role": "user", "content": "hi"}])
    history.clear()
    assert history.load_recent() == []


def test_history_persists_across_connections(tmp_path):
    db_file = tmp_path / "test.db"
    first = connect(db_file)
    HistoryStore(first).append_many([{"role": "user", "content": "remember me"}])
    first.close()

    second = connect(db_file)
    assert HistoryStore(second).load_recent()[0]["content"] == "remember me"
    second.close()