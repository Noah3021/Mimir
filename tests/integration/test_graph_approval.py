"""人工确认：interrupt / resume。核心是“副作用只发生一次”和多个确认的顺序。"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from mimir.approval import ApprovalPolicy
from tests.fakes import ScriptedChatModel, pending_interrupts, run_config, tool_call, tool_calls


@pytest.fixture
def side_effects():
    return []


@pytest.fixture
def tools(side_effects):
    @tool
    def danger(n: int = 0) -> str:
        """有副作用的操作。"""
        side_effects.append(n)
        return f"done {n}"

    @tool
    def safe() -> str:
        """无需确认的操作。"""
        side_effects.append("safe")
        return "safe ok"

    return [danger, safe]


@pytest.fixture
def approval():
    return ApprovalPolicy(["danger"])


CFG = run_config("t1")


def results(graph):
    msgs = graph.get_state(CFG).values["messages"]
    return {m.tool_call_id: m for m in msgs if isinstance(m, ToolMessage)}


def test_interrupt_then_approve_runs_once(make_graph, tools, approval, side_effects):
    model = ScriptedChatModel(script=[tool_call("danger", {"n": 1}, "c1"), "完成"])
    graph = make_graph(model, tools=tools, approval=approval)
    graph.invoke({"messages": [HumanMessage("做")]}, CFG)

    ints = pending_interrupts(graph, CFG)
    assert len(ints) == 1
    assert ints[0].value == {"type": "approval", "tool": "danger", "args": {"n": 1}, "id": "c1"}
    assert side_effects == []  # 确认前不执行

    out = graph.invoke(Command(resume={"approved": True}), CFG)
    assert side_effects == [1]  # 恢复重放节点后只执行一次
    assert results(graph)["c1"].content == "done 1"
    assert out["messages"][-1].content == "完成"
    assert pending_interrupts(graph, CFG) == []
    assert len(model.calls) == 2  # 恢复时不会重新调用模型


@pytest.mark.parametrize("decision, expected", [
    ({"approved": False}, "不要重复尝试"),
    ({"approved": False, "feedback": "换个文件名"}, "用户反馈：换个文件名"),
    ("n", "不要重复尝试"),
])
def test_reject(make_graph, tools, approval, side_effects, decision, expected):
    model = ScriptedChatModel(script=[tool_call("danger", {"n": 1}, "c1"), "好的"])
    graph = make_graph(model, tools=tools, approval=approval)
    graph.invoke({"messages": [HumanMessage("做")]}, CFG)
    graph.invoke(Command(resume=decision), CFG)
    assert side_effects == []
    tm = results(graph)["c1"]
    assert tm.status == "error"
    assert "用户拒绝执行此操作" in tm.content and expected in tm.content


def test_multiple_calls_in_order(make_graph, tools, approval, side_effects):
    calls = tool_calls(("danger", {"n": 1}, "c1"), ("safe", {}, "c2"), ("danger", {"n": 2}, "c3"))
    model = ScriptedChatModel(script=[calls, "好"])
    graph = make_graph(model, tools=tools, approval=approval)
    graph.invoke({"messages": [HumanMessage("做")]}, CFG)

    assert [i.value["id"] for i in pending_interrupts(graph, CFG)] == ["c1"]
    graph.invoke(Command(resume={"approved": True}), CFG)
    assert [i.value["id"] for i in pending_interrupts(graph, CFG)] == ["c3"]
    assert side_effects == []  # 全部确认完才开始执行

    graph.invoke(Command(resume={"approved": False, "feedback": "第二个不要"}), CFG)
    assert side_effects == [1, "safe"]
    r = results(graph)
    assert r["c1"].content == "done 1"
    assert r["c2"].content == "safe ok"
    assert r["c3"].status == "error" and "第二个不要" in r["c3"].content
    # ToolMessage 顺序与调用顺序一致
    msgs = graph.get_state(CFG).values["messages"]
    assert [m.tool_call_id for m in msgs if isinstance(m, ToolMessage)] == ["c1", "c2", "c3"]


def test_allowed_tool_skips_interrupt(make_graph, tools, approval, side_effects):
    approval.allow("danger")
    model = ScriptedChatModel(script=[tool_call("danger", {"n": 5}, "c1"), "好"])
    graph = make_graph(model, tools=tools, approval=approval)
    graph.invoke({"messages": [HumanMessage("做")]}, CFG)
    assert pending_interrupts(graph, CFG) == []
    assert side_effects == [5]


def test_approval_disabled(make_graph, tools, side_effects):
    model = ScriptedChatModel(script=[tool_call("danger", {"n": 7}, "c1"), "好"])
    graph = make_graph(model, tools=tools, approval=ApprovalPolicy(["danger"], enabled=False))
    graph.invoke({"messages": [HumanMessage("做")]}, CFG)
    assert side_effects == [7]
