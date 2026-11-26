"""Token 用量统计：从 AIMessage.usage_metadata 汇总，并持久化到 SQLite，可按会话 / 日期 / 模型查询。

只统计用量，不计算价格。说明：
- input 含缓存命中部分（LangChain 已把各家接口统一成这种口径）
- reasoning（推理 token）包含在 output 中
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, fields
from datetime import datetime
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output

    @property
    def cache_rate(self) -> float:
        """缓存命中率：命中 token / 输入 token（input 已包含命中部分）。"""
        return self.cache_read / self.input if self.input else 0.0

    def add(self, other: Usage) -> Usage:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self


def usage_of(msg: BaseMessage) -> Usage | None:
    um = getattr(msg, "usage_metadata", None)
    if not isinstance(msg, AIMessage) or not um:
        return None
    itd = um.get("input_token_details") or {}
    otd = um.get("output_token_details") or {}
    return Usage(
        input=int(um.get("input_tokens") or 0),
        output=int(um.get("output_tokens") or 0),
        reasoning=int(otd.get("reasoning") or 0),
        cache_read=int(itd.get("cache_read") or 0),
        cache_write=int(itd.get("cache_creation") or 0),
        calls=1,
    )


def summarize(messages: Sequence[BaseMessage]) -> Usage:
    total = Usage()
    for m in messages:
        u = usage_of(m)
        if u:
            total.add(u)
    return total


def fmt_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def format_usage_line(u: Usage, elapsed: float | None = None) -> str:
    inp = f"↑ {fmt_tokens(u.input)}"
    if u.cache_read:
        inp += f"（缓存 {fmt_tokens(u.cache_read)}，{u.cache_rate:.0%}）"
    out = f"↓ {fmt_tokens(u.output)}"
    if u.reasoning:
        out += f"（推理 {fmt_tokens(u.reasoning)}）"
    parts = [inp, out, f"{u.calls} 次调用"]
    if elapsed is not None:
        parts.append(f"{elapsed:.1f}s")
    return "  " + " · ".join(parts)


@dataclass
class UsageRow:
    profile: str
    model: str
    usage: Usage


class UsageStore:
    """每次模型调用一行，message_id 去重。删除会话不会删除用量记录。"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.execute(
            """CREATE TABLE IF NOT EXISTS usage_log (
                   message_id       TEXT PRIMARY KEY,
                   ts               TEXT,
                   day              TEXT,
                   thread_id        TEXT,
                   profile          TEXT,
                   model            TEXT,
                   input_tokens     INTEGER,
                   output_tokens    INTEGER,
                   reasoning_tokens INTEGER,
                   cache_read       INTEGER,
                   cache_write      INTEGER
               )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_day ON usage_log(day)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_thread ON usage_log(thread_id)")
        conn.commit()

    def record(self, thread_id: str, messages: Sequence[BaseMessage], *, profile: str, model_id: str) -> int:
        now = datetime.now()
        rows = []
        for i, m in enumerate(messages):
            u = usage_of(m)
            if u is None:
                continue
            meta = m.response_metadata or {}
            mid = m.id or f"{thread_id}:{now.timestamp()}:{i}"
            rows.append((
                mid, now.isoformat(timespec="seconds"), now.date().isoformat(), thread_id,
                meta.get("agent_profile") or profile,
                meta.get("model_name") or meta.get("model") or model_id,
                u.input, u.output, u.reasoning, u.cache_read, u.cache_write,
            ))
        if not rows:
            return 0
        cur = self.conn.executemany("INSERT OR IGNORE INTO usage_log VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return cur.rowcount

    def by_model(self, *, thread_id: str | None = None, day: str | None = None) -> list[UsageRow]:
        where, args = [], []
        if thread_id:
            where.append("thread_id = ?")
            args.append(thread_id)
        if day:
            where.append("day = ?")
            args.append(day)
        sql = (
            "SELECT profile, model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(reasoning_tokens), "
            "SUM(cache_read), SUM(cache_write) FROM usage_log "
            + (("WHERE " + " AND ".join(where)) if where else "")
            + " GROUP BY profile, model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC"
        )
        result = []
        for p, mdl, n, i, o, r, cr, cw in self.conn.execute(sql, args).fetchall():
            result.append(UsageRow(p or "-", mdl or "-", Usage(
                input=i or 0, output=o or 0, reasoning=r or 0, cache_read=cr or 0, cache_write=cw or 0, calls=n or 0)))
        return result
