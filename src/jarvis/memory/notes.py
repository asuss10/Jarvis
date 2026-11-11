"""Personal notes: save, list and search."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

MAX_NOTE_LENGTH = 2000
MAX_RESULTS = 50
ESCAPE_CHAR = "!"


class NoteValidationError(ValueError):
    """Raised when note text or a search query is not acceptable."""


@dataclass(frozen=True)
class Note:
    id: int
    text: str
    created_at: str


class NotesStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, text: str) -> Note:
        text = text.strip()
        if not text:
            raise NoteValidationError("Note text must not be empty.")
        if len(text) > MAX_NOTE_LENGTH:
            raise NoteValidationError(
                f"Note is too long ({len(text)} characters, max {MAX_NOTE_LENGTH})."
            )
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn:  # commits on success, rolls back on error
            cursor = self._conn.execute(
                "INSERT INTO notes (text, created_at) VALUES (?, ?)",
                (text, created_at),
            )
        return Note(id=cursor.lastrowid, text=text, created_at=created_at)

    def list_recent(self, limit: int = 10) -> list[Note]:
        limit = max(1, min(limit, MAX_RESULTS))
        rows = self._conn.execute(
            "SELECT id, text, created_at FROM notes ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Note(**dict(row)) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[Note]:
        query = query.strip()
        if not query:
            raise NoteValidationError("Search query must not be empty.")
        limit = max(1, min(limit, MAX_RESULTS))
        # Escape LIKE wildcards so the query is matched literally.
        escaped = (
            query.replace(ESCAPE_CHAR, ESCAPE_CHAR * 2)
            .replace("%", ESCAPE_CHAR + "%")
            .replace("_", ESCAPE_CHAR + "_")
        )
        rows = self._conn.execute(
            "SELECT id, text, created_at FROM notes "
            f"WHERE text LIKE ? ESCAPE '{ESCAPE_CHAR}' ORDER BY id DESC LIMIT ?",
            (f"%{escaped}%", limit),
        ).fetchall()
        return [Note(**dict(row)) for row in rows]