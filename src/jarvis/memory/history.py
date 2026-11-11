"""Persistent conversation history."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

Message = dict[str, Any]


class HistoryStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append_many(self, messages: list[Message]) -> None:
        """Save messages in one transaction (all or nothing)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [
            (
                m["role"],
                m.get("content") or "",
                json.dumps(m["tool_calls"]) if m.get("tool_calls") else None,
                m.get("tool_name"),
                now,
            )
            for m in messages
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT INTO messages (role, content, tool_calls, tool_name, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def load_recent(self, limit: int = 200) -> list[Message]:
        """Newest `limit` messages, oldest first, starting at a user message."""
        rows = self._conn.execute(
            "SELECT role, content, tool_calls, tool_name FROM messages "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        messages = [self._to_message(row) for row in reversed(rows)]
        # The cut may have landed mid-turn; never start with an orphaned reply.
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        return messages

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM messages")

    @staticmethod
    def _to_message(row: sqlite3.Row) -> Message:
        message: Message = {"role": row["role"], "content": row["content"]}
        if row["tool_calls"]:
            message["tool_calls"] = json.loads(row["tool_calls"])
        if row["tool_name"]:
            message["tool_name"] = row["tool_name"]
        return message