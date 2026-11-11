import pytest

from jarvis.memory.database import connect
from jarvis.memory.notes import MAX_NOTE_LENGTH, NotesStore, NoteValidationError


@pytest.fixture
def store():
    conn = connect(":memory:")
    yield NotesStore(conn)
    conn.close()


def test_add_and_list(store):
    note = store.add("Dentist on Friday at 10:00")
    assert note.id == 1
    assert store.list_recent()[0].text == "Dentist on Friday at 10:00"


def test_add_strips_whitespace(store):
    assert store.add("  hello  ").text == "hello"


def test_empty_note_rejected(store):
    with pytest.raises(NoteValidationError):
        store.add("   ")


def test_too_long_note_rejected(store):
    with pytest.raises(NoteValidationError):
        store.add("x" * (MAX_NOTE_LENGTH + 1))


def test_list_recent_is_newest_first(store):
    store.add("first")
    store.add("second")
    assert [n.text for n in store.list_recent()] == ["second", "first"]


def test_search_is_case_insensitive(store):
    store.add("Buy MILK tomorrow")
    store.add("Call mom")
    results = store.search("milk")
    assert [n.text for n in results] == ["Buy MILK tomorrow"]


def test_search_treats_wildcards_literally(store):
    store.add("100% done")
    store.add("fully done")
    assert [n.text for n in store.search("%")] == ["100% done"]


def test_empty_search_rejected(store):
    with pytest.raises(NoteValidationError):
        store.search("")


def test_sql_injection_attempt_is_stored_as_plain_text(store):
    evil = "'); DROP TABLE notes; --"
    store.add(evil)
    assert store.list_recent()[0].text == evil  # table still exists


def test_notes_persist_across_connections(tmp_path):
    db_file = tmp_path / "test.db"
    first = connect(db_file)
    NotesStore(first).add("remember me")
    first.close()

    second = connect(db_file)
    assert NotesStore(second).list_recent()[0].text == "remember me"
    second.close()