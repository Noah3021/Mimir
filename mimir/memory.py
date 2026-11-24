"""会话持久化：LangGraph SqliteSaver 保存对话状态，另建一张表保存会话元信息（标题等）。"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


@dataclass
class SessionInfo:
    thread_id: str
    title: str
    model: str
    created_at: str
    updated_at: str


class SessionStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)
        self.checkpointer.setup()
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS agent_sessions (
                   thread_id  TEXT PRIMARY KEY,
                   title      TEXT,
                   model      TEXT,
                   created_at TEXT,
                   updated_at TEXT
               )"""
        )
        self.conn.commit()

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex[:12]

    def touch(self, thread_id: str, title: str, model: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute(
            """INSERT INTO agent_sessions (thread_id, title, model, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(thread_id) DO UPDATE SET model = excluded.model, updated_at = excluded.updated_at""",
            (thread_id, title, model, now, now),
        )
        self.conn.commit()

    def list(self, limit: int = 20) -> list[SessionInfo]:
        rows = self.conn.execute(
            "SELECT thread_id, title, model, created_at, updated_at FROM agent_sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [SessionInfo(*r) for r in rows]

    def find(self, key: str) -> list[SessionInfo]:
        """按 ID 或 ID 前缀查找。"""
        rows = self.conn.execute(
            "SELECT thread_id, title, model, created_at, updated_at FROM agent_sessions "
            "WHERE thread_id LIKE ? ORDER BY updated_at DESC",
            (key.strip() + "%",),
        ).fetchall()
        return [SessionInfo(*r) for r in rows]

    def delete(self, thread_id: str) -> None:
        try:
            self.checkpointer.delete_thread(thread_id)
        finally:
            self.conn.execute("DELETE FROM agent_sessions WHERE thread_id = ?", (thread_id,))
            self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
