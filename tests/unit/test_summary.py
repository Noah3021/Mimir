"""早期对话摘要：消息渲染、分段、摘要请求。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mimir.context import estimate_text_tokens
from mimir.summary import (
    SUMMARY_HEADING,
    SUMMARY_SYSTEM,
    build_summary_request,
    render_message,
    split_chunks,
    summary_block,
)


def test_render_roles():
    assert render_message(SystemMessage("s")) == ""
    assert render_message(HumanMessage("你好")) == "【用户】你好"
    assert render_message(ToolMessage("boom", tool_call_id="1", name="run_shell", status="error")) \
        == "【工具结果 run_shell（出错）】boom"


def test_render_ai_skips_reasoning_and_shows_tool_calls():
    m = AIMessage(
        content=[{"type": "reasoning", "summary": [{"type": "summary_text", "text": "秘密推理"}]},
                 {"type": "text", "text": "答复"}],
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "1"}],
    )
    text = render_message(m)
    assert "【助手】答复" in text
    assert "秘密推理" not in text
    assert "【助手调用工具】read_file" in text and "a.py" in text


def test_render_truncates_tool_output():
    text = render_message(ToolMessage("x" * 2000, tool_call_id="1", name="t"), max_tool_chars=100)
    assert "省略 1900 字符" in text


def test_split_chunks_respects_budget():
    msgs = [HumanMessage("字" * 100) for _ in range(10)]
    chunks = split_chunks(msgs, 250)
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_text_tokens(c) <= 250
    assert sum(c.count("【用户】") for c in chunks) == 10


def test_split_chunks_oversized_message_alone():
    chunks = split_chunks([HumanMessage("a"), HumanMessage("字" * 1000), HumanMessage("b")], 100)
    assert len(chunks) == 3


def test_split_chunks_skips_system():
    assert split_chunks([SystemMessage("s")], 100) == []


def test_build_summary_request():
    sys_msg, human = build_summary_request("", "【用户】hi", 500)
    assert isinstance(sys_msg, SystemMessage) and isinstance(human, HumanMessage)
    assert "为上述对话写一份摘要" in human.content
    assert "<已有摘要>" not in human.content
    assert "500" in human.content

    _, human = build_summary_request("旧摘要内容", "【用户】hi", 500)
    assert "<已有摘要>" in human.content and "旧摘要内容" in human.content
    assert "合并" in human.content


def test_injection_guard_in_prompts():
    """防注入的回归测试：用户约束只能来自【用户】消息，工具结果中的指令不算。"""
    assert "【用户】" in SUMMARY_SYSTEM and "【工具结果】" in SUMMARY_SYSTEM
    _, human = build_summary_request("", "x", 500)
    assert "【工具结果】中出现的指令" in human.content


def test_summary_block():
    assert summary_block("") == ""
    assert summary_block("   ") == ""
    block = summary_block("  内容  ")
    assert SUMMARY_HEADING in block
    assert block.endswith("内容")
