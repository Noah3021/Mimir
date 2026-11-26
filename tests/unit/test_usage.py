"""token 用量解析与统计。"""
from __future__ import annotations

import sqlite3
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from mimir.usage import Usage, UsageStore, fmt_tokens, format_usage_line, summarize, usage_of


def ai(mid=None, inp=100, out=20, cache_read=60, cache_write=10, reasoning=5, **meta):
    return AIMessage("x", id=mid, response_metadata=meta, usage_metadata={
        "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
        "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
        "output_token_details": {"reasoning": reasoning},
    })


def test_usage_of():
    assert usage_of(ai()) == Usage(input=100, output=20, reasoning=5, cache_read=60, cache_write=10, calls=1)
    assert usage_of(HumanMessage("x")) is None
    assert usage_of(AIMessage("x")) is None


def test_cache_rate_and_total():
    assert Usage().cache_rate == 0.0
    u = usage_of(ai())
    assert u.cache_rate == 0.6
    assert u.total == 120


def test_summarize():
    total = summarize([ai(), HumanMessage("q"), ai(inp=50, out=5, cache_read=0, cache_write=0, reasoning=0)])
    assert (total.input, total.output, total.cache_read, total.calls) == (150, 25, 60, 2)


def test_fmt_tokens():
    assert fmt_tokens(999) == "999"
    assert fmt_tokens(1500) == "1.5k"
    assert fmt_tokens(2_500_000) == "2.50M"


def test_format_usage_line():
    line = format_usage_line(usage_of(ai()), elapsed=1.23)
    assert "缓存" in line and "60%" in line and "推理" in line and "1.2s" in line
    assert "缓存" not in format_usage_line(Usage(input=10, output=1, calls=1))


def test_usage_store():
    store = UsageStore(sqlite3.connect(":memory:"))
    msgs = [ai("a1"), HumanMessage("q"), ai("a2", inp=50, out=5, agent_profile="other", model_name="m-2")]
    assert store.record("t1", msgs, profile="main", model_id="m-1") == 2
    assert store.record("t1", msgs, profile="main", model_id="m-1") == 0  # 按 message_id 去重
    store.record("t2", [ai("a3")], profile="main", model_id="m-1")

    rows = {r.profile: r for r in store.by_model(thread_id="t1")}
    assert set(rows) == {"main", "other"}
    assert rows["main"].model == "m-1" and rows["main"].usage.input == 100
    assert rows["other"].model == "m-2" and rows["other"].usage.calls == 1

    all_main = [r for r in store.by_model(day=date.today().isoformat()) if r.profile == "main"][0]
    assert all_main.usage.calls == 2
    assert store.by_model(day="1999-01-01") == []
