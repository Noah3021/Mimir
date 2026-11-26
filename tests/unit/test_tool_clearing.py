"""旧工具输出清理。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mimir.tool_clearing import clear_boundary, clear_old_tool_outputs, cleared_chars, is_cleared


def test_clear_boundary():
    msgs = [HumanMessage("1"), AIMessage("a"), HumanMessage("2"), AIMessage("b"), HumanMessage("3"), AIMessage("c")]
    assert clear_boundary(msgs, 2) == 2
    assert clear_boundary(msgs, 1) == 4
    assert clear_boundary(msgs, 5) == 0
    assert clear_boundary(msgs, 0) == len(msgs)


def build():
    return [
        HumanMessage("q"),
        ToolMessage("x" * 1000, tool_call_id="c1", name="read_file", status="success", id="m1"),
        ToolMessage("short", tool_call_id="c2", name="list_directory"),
        HumanMessage("q2"),
        ToolMessage("y" * 1000, tool_call_id="c3", name="read_file"),
    ]


def test_only_long_outputs_before_boundary_cleared():
    msgs = build()
    out = clear_old_tool_outputs(msgs, 3, min_chars=500)
    assert is_cleared(out[1])
    assert "read_file" in out[1].content and "1000" in out[1].content
    assert (out[1].tool_call_id, out[1].name, out[1].status, out[1].id) == ("c1", "read_file", "success", "m1")
    assert out[2] is msgs[2]  # 短输出不动
    assert out[4] is msgs[4]  # 边界之后不动
    assert out[0] is msgs[0]


def test_original_messages_not_modified():
    msgs = build()
    clear_old_tool_outputs(msgs, 3)
    assert msgs[1].content == "x" * 1000


def test_already_cleared_not_reprocessed():
    once = clear_old_tool_outputs(build(), 3)
    twice = clear_old_tool_outputs(once, 3)
    assert twice[1] is once[1]


def test_list_content_counted():
    m = ToolMessage([{"type": "text", "text": "z" * 600}], tool_call_id="c1", name="t")
    out = clear_old_tool_outputs([m], 1, min_chars=500)
    assert is_cleared(out[0])


def test_boundary_clamped():
    msgs = build()
    assert clear_old_tool_outputs(msgs, -5) == msgs
    assert len(clear_old_tool_outputs(msgs, 99)) == len(msgs)


def test_cleared_chars():
    msgs = build()
    out = clear_old_tool_outputs(msgs, 3)
    assert cleared_chars(msgs, out) == 1000 - len(out[1].content)
