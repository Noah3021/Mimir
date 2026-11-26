"""token 估算与压缩点计算。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mimir.context import estimate_message_tokens, estimate_text_tokens, history_budget, plan_context


def length(m):
    return len(m.content) if isinstance(m.content, str) else 0


def turns(n, size=10):
    """n 轮对话，每条消息 size 个字符（按 length 计 size token）。用户消息下标为 0, 2, 4, ..."""
    msgs = []
    for _ in range(n):
        msgs += [HumanMessage("h" * size), AIMessage("a" * size)]
    return msgs


def test_estimate_text_tokens():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("中文") == 2
    assert estimate_text_tokens("abc") == 1
    assert estimate_text_tokens("abcdef") == 2
    assert estimate_text_tokens("中a") == 2


def test_message_tokens_include_tool_calls():
    plain = AIMessage("x")
    with_call = AIMessage("x", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "1"}])
    assert estimate_message_tokens(with_call) > estimate_message_tokens(plain)


def test_below_high_unchanged():
    msgs = turns(3)
    p = plan_context(msgs, high=100, low=50, cut=0, clear_before=0, cost=length)
    assert (p.cut, p.clear_before, p.tokens, p.total_tokens) == (0, 0, 60, 60)


def test_existing_state_kept_below_high():
    msgs = turns(5)
    p = plan_context(msgs, high=1000, low=500, cut=4, clear_before=3, cost=length)
    assert (p.cut, p.clear_before) == (4, 3)


def test_cut_moves_to_user_boundary_until_low():
    msgs = turns(5)  # 100 tokens，用户消息在 0/2/4/6/8
    p = plan_context(msgs, high=50, low=30, cost=length)
    assert p.cut == 8
    assert isinstance(msgs[p.cut], HumanMessage)
    assert p.tokens == 20
    assert p.before == 100


def test_cut_stops_at_first_boundary_that_fits():
    msgs = turns(5)
    p = plan_context(msgs, high=50, low=65, cost=length)
    assert p.cut == 4  # 剩余 60 <= 65
    assert p.tokens == 60


def test_summary_reserve_counted():
    msgs = turns(5)
    p = plan_context(msgs, high=50, low=30, summary_reserve=15, cost=length)
    assert p.tokens == 15 + 100 - sum(length(m) for m in msgs[:p.cut])


def test_current_turn_never_compressed():
    msgs = [*turns(1), HumanMessage("h" * 10), AIMessage("x" * 1000)]
    p = plan_context(msgs, high=50, low=30, cost=length)
    assert p.cut == 2  # 只能压到最后一条用户消息之前


def test_single_turn_not_compressed():
    msgs = [HumanMessage("h" * 500), AIMessage("a" * 500)]
    p = plan_context(msgs, high=50, low=30, cost=length)
    assert p.cut == 0


def test_summary_tokens_in_before():
    msgs = turns(2)
    p = plan_context(msgs, high=1000, low=500, summary_tokens=7, cost=length)
    assert p.before == p.tokens == 47


def test_tool_clearing_before_summary():
    call = AIMessage("", tool_calls=[{"name": "read_file", "args": {}, "id": "c1"}])
    msgs = [HumanMessage("h" * 10), call, ToolMessage("t" * 1000, tool_call_id="c1"),
            HumanMessage("h" * 10), AIMessage("a" * 10)]
    p = plan_context(
        msgs, high=500, low=300, cost=length,
        cleared_cost=lambda m: 5 if isinstance(m, ToolMessage) else length(m),
        clear_boundary=lambda end: 3,
    )
    assert p.clear_before == 3
    assert p.cut == 0  # 清理后已低于低水位，不需要摘要
    assert p.tokens == 35
    assert p.before == 1030


def test_cut_and_clear_never_move_backwards():
    msgs = turns(5)
    p = plan_context(msgs, high=50, low=30, cut=6, clear_before=6, cost=length,
                     cleared_cost=length, clear_boundary=lambda end: 2)
    assert p.cut >= 6
    assert p.clear_before >= 6


def test_history_budget():
    assert history_budget(100_000, 0.8, 0.5, reserve=16_000, overhead=2_000) == (78_000, 48_000)
    # 输出预留更紧时以预留为准
    high, _ = history_budget(100_000, 0.9, 0.5, reserve=20_000, overhead=0)
    assert high == 80_000
    # 窗口很小时有下限
    high, low = history_budget(1_000, 0.8, 0.5, reserve=16_000, overhead=5_000)
    assert high == 1000
    assert low == 500
