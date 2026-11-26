"""agent ⇄ tools 主循环：工具调用、错误处理、输出截断、最大步数。"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from tests.conftest import SYSTEM_PROMPT
from tests.fakes import ScriptedChatModel, run_config, tool_call, tool_calls


@tool
def echo(text: str) -> str:
    """回显文本。"""
    return f"echo:{text}"


@tool
def boom() -> str:
    """总是失败。"""
    raise ValueError("坏了")


@tool
def big() -> str:
    """返回很长的输出。"""
    return "x" * 100


@tool
def as_dict() -> dict:
    """返回字典。"""
    return {"k": "中文"}


def ask(graph, text="你好", thread="t1", **cfg):
    return graph.invoke({"messages": [HumanMessage(text)]}, run_config(thread, **cfg))


def test_plain_reply(make_graph):
    model = ScriptedChatModel(script=["你好呀"])
    out = ask(make_graph(model))
    assert [type(m) for m in out["messages"]] == [HumanMessage, AIMessage]
    assert out["messages"][-1].content == "你好呀"
    sent = model.calls[0]
    assert sent[0].content == SYSTEM_PROMPT
    assert sent[1].content == "你好"
    assert out["messages"][-1].response_metadata["agent_profile"] == "test"


def test_tool_roundtrip(make_graph):
    model = ScriptedChatModel(script=[tool_call("echo", {"text": "hi"}, "c1"), "完成"])
    out = ask(make_graph(model, tools=[echo]))
    tm = out["messages"][2]
    assert isinstance(tm, ToolMessage)
    assert (tm.content, tm.tool_call_id, tm.name, tm.status) == ("echo:hi", "c1", "echo", "success")
    assert out["messages"][-1].content == "完成"
    # 第二次调用时模型看到了工具结果
    assert isinstance(model.calls[1][-1], ToolMessage)


def test_multiple_tool_calls_keep_order(make_graph):
    model = ScriptedChatModel(script=[tool_calls(("echo", {"text": "1"}, "a"), ("echo", {"text": "2"}, "b")), "好"])
    out = ask(make_graph(model, tools=[echo]))
    results = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.content) for m in results] == [("a", "echo:1"), ("b", "echo:2")]


def test_tool_exception_reported_not_raised(make_graph):
    model = ScriptedChatModel(script=[tool_call("boom", {}, "c1"), "知道了"])
    out = ask(make_graph(model, tools=[boom]))
    tm = out["messages"][2]
    assert tm.status == "error"
    assert "工具执行出错：ValueError: 坏了" in tm.content
    assert out["messages"][-1].content == "知道了"


def test_unknown_tool(make_graph):
    model = ScriptedChatModel(script=[tool_call("nope", {}, "c1"), "好"])
    out = ask(make_graph(model, tools=[echo]))
    assert out["messages"][2].status == "error"
    assert "不存在工具 nope" in out["messages"][2].content


def test_tool_output_truncated(make_graph):
    model = ScriptedChatModel(script=[tool_call("big", {}, "c1"), "好"])
    out = ask(make_graph(model, tools=[big], max_tool_output_chars=20))
    assert out["messages"][2].content == "x" * 20 + "\n...（已截断）"


def test_non_string_output_serialized(make_graph):
    model = ScriptedChatModel(script=[tool_call("as_dict", {}, "c1"), "好"])
    out = ask(make_graph(model, tools=[as_dict]))
    assert out["messages"][2].content == '{"k": "中文"}'


def test_recursion_limit(make_graph):
    model = ScriptedChatModel(responder=lambda msgs: tool_call("echo", {"text": "again"}, f"c{len(msgs)}"))
    with pytest.raises(GraphRecursionError):
        ask(make_graph(model, tools=[echo]), recursion_limit=6)


def test_history_persists_across_turns(make_graph):
    model = ScriptedChatModel(script=["一", "二"])
    graph = make_graph(model)
    ask(graph, "第一轮")
    out = ask(graph, "第二轮")
    assert [m.content for m in out["messages"]] == ["第一轮", "一", "第二轮", "二"]
    assert [m.content for m in model.calls[1][1:]] == ["第一轮", "一", "第二轮"]
    # 其他会话互不影响
    model.script.append("三")
    other = ask(graph, "别的会话", thread="t2")
    assert [m.content for m in other["messages"]] == ["别的会话", "三"]
