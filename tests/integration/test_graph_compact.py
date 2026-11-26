"""上下文压缩：摘要、工具输出清理、失败降级、缓存前缀稳定性、项目上下文注入。

用很小的窗口触发压缩：window=4000、high=0.5、low=0.25、output_reserve=0，
历史预算约为 高水位 ~1990 / 低水位 ~990 token。每条消息 300 个汉字（约 304 token），
因此前 3 轮不压缩，第 4 轮开始时越过高水位。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from mimir.config import ContextConfig
from mimir.project_context import ProjectContext
from mimir.summary import SUMMARY_HEADING, SUMMARY_SYSTEM
from mimir.tool_clearing import PLACEHOLDER_PREFIX
from tests.fakes import ScriptedChatModel, run_config, tool_call

CTX = ContextConfig(window=4000, high=0.5, low=0.25, output_reserve=0, clear_keep_turns=0, summary_chars=200)
TEXT = "字" * 300
CFG = run_config()


def is_summary_request(msgs) -> bool:
    return isinstance(msgs[0], SystemMessage) and msgs[0].content == SUMMARY_SYSTEM


class Responder:
    """普通请求回复 300 字；摘要请求回复固定摘要（或抛出异常）。"""

    def __init__(self, summary="【摘要】用户在讨论测试", fail=None):
        self.summary, self.fail = summary, fail
        self.summary_calls: list = []

    def __call__(self, msgs):
        if is_summary_request(msgs):
            self.summary_calls.append(msgs)
            return self.fail or self.summary
        return TEXT


def agent_calls(model):
    return [c for c in model.calls if not is_summary_request(c)]


def contents(msgs):
    return [(type(m).__name__, m.content) for m in msgs]


def chat(graph, n, start=0):
    for i in range(start, start + n):
        graph.invoke({"messages": [HumanMessage(f"第{i + 1}轮" + TEXT)]}, CFG)


@pytest.fixture
def events():
    return []


def chat_streaming(graph, n, events):
    for i in range(n):
        for mode, data in graph.stream({"messages": [HumanMessage(f"第{i + 1}轮" + TEXT)]}, CFG,
                                       stream_mode=["custom", "values"]):
            if mode == "custom":
                events.append(data)


def test_no_compaction_below_high(make_graph):
    resp = Responder()
    model = ScriptedChatModel(responder=resp)
    graph = make_graph(model, context=CTX)
    chat(graph, 3)
    values = graph.get_state(CFG).values
    assert resp.summary_calls == []
    assert values.get("summary_cut", 0) == 0 and not values.get("summary")


def test_prefix_stable_between_compactions(make_graph):
    """缓存前缀稳定性：两次压缩之间，每次请求都以上一次请求为前缀。"""
    model = ScriptedChatModel(responder=Responder())
    chat(make_graph(model, context=CTX), 3)
    calls = agent_calls(model)
    assert len(calls) == 3
    for prev, cur in zip(calls, calls[1:], strict=False):
        assert contents(cur)[:len(prev)] == contents(prev)


def test_summary_compaction(make_graph, events):
    resp = Responder()
    model = ScriptedChatModel(responder=resp)
    saver = InMemorySaver()
    graph = make_graph(model, context=CTX, checkpointer=saver)
    chat_streaming(graph, 4, events)

    assert len(resp.summary_calls) == 1
    values = graph.get_state(CFG).values
    cut = values["summary_cut"]
    msgs = values["messages"]
    assert cut > 0 and isinstance(msgs[cut], HumanMessage)
    assert values["summary"] == resp.summary
    assert len(msgs) == 8  # 完整历史仍在状态里

    # 摘要请求包含被压缩的消息，不含当前这一轮
    transcript = resp.summary_calls[0][1].content
    assert "第1轮" in transcript and "第4轮" not in transcript

    # 压缩后的请求：系统提示词 + 摘要，后面只有 cut 之后的消息
    last = agent_calls(model)[-1]
    assert SUMMARY_HEADING in last[0].content and resp.summary in last[0].content
    assert contents(last[1:]) == contents(msgs[cut:-1])

    compact = [e for e in events if e.get("type") == "compact"]
    assert [e["stage"] for e in compact] == ["start", "done"]
    assert compact[1]["after"] < compact[1]["before"]

    # 压缩状态随 Checkpointer 持久化：换一个图（如切换模型）读取仍然存在
    other = make_graph(ScriptedChatModel(), context=CTX, checkpointer=saver)
    assert other.get_state(CFG).values["summary_cut"] == cut


def test_prefix_stable_after_compaction(make_graph):
    model = ScriptedChatModel(responder=Responder())
    graph = make_graph(model, context=CTX)
    chat(graph, 5)
    calls = agent_calls(model)
    # 第 4 轮压缩后，第 5 轮的请求以第 4 轮的请求为前缀
    assert contents(calls[4])[:len(calls[3])] == contents(calls[3])


def test_summary_failure_degrades(make_graph, events):
    resp = Responder(fail=RuntimeError("摘要服务挂了"))
    model = ScriptedChatModel(responder=resp)
    graph = make_graph(model, context=CTX)
    chat_streaming(graph, 4, events)

    values = graph.get_state(CFG).values
    assert values["summary_cut"] > 0
    assert "摘要生成失败而被省略" in values["summary"]
    assert values["messages"][-1].content == TEXT  # 本轮照常完成
    failed = [e for e in events if e.get("type") == "compact" and e["stage"] == "failed"]
    assert len(failed) == 1 and "摘要服务挂了" in failed[0]["error"]


def test_tool_output_clearing_only(make_graph):
    @tool
    def read_big() -> str:
        """读取一个大文件。"""
        return "字" * 2500

    ctx = ContextConfig(window=4000, high=0.5, low=0.25, output_reserve=0,
                        clear_keep_turns=1, clear_min_chars=100, summary_chars=200)
    resp = Responder()
    model = ScriptedChatModel(script=[tool_call("read_big", {}, "c1"), "读完了"], responder=resp)
    graph = make_graph(model, tools=[read_big], context=ctx)
    graph.invoke({"messages": [HumanMessage("读文件")]}, CFG)
    graph.invoke({"messages": [HumanMessage("继续")]}, CFG)

    assert resp.summary_calls == []  # 清理后已低于低水位，不需要摘要
    values = graph.get_state(CFG).values
    assert values["clear_before"] > 0 and values.get("summary_cut", 0) == 0
    sent_tool = [m for m in agent_calls(model)[-1] if isinstance(m, ToolMessage)][0]
    assert sent_tool.content.startswith(PLACEHOLDER_PREFIX)
    assert sent_tool.tool_call_id == "c1"
    stored_tool = [m for m in values["messages"] if isinstance(m, ToolMessage)][0]
    assert stored_tool.content == "字" * 2500  # 状态里的原始输出不变


def test_project_context_injection(make_graph):
    ctx_obj = ProjectContext(project_file=Path("MIMIR.md"), project_text="项目规则：用中文")
    model = ScriptedChatModel(responder=Responder())
    graph = make_graph(model, context=CTX, project_context=lambda: ctx_obj)
    chat(graph, 4)

    calls = agent_calls(model)
    first_human = calls[0][1]
    assert first_human.content.startswith("<project-context>") and "项目规则：用中文" in first_human.content
    values = graph.get_state(CFG).values
    assert all("<project-context>" not in m.content for m in values["messages"])  # 不写入状态

    # 压缩后，项目上下文拼到剩下的第一条用户消息前面，且只出现一次
    last = calls[-1]
    humans = [m for m in last if isinstance(m, HumanMessage)]
    assert humans[0].content.startswith("<project-context>")
    assert sum("<project-context>" in m.content for m in humans) == 1
