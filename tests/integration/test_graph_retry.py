"""模型请求连接中断时的自动重试。"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests.fakes import ScriptedChatModel, run_config


class APIConnectionError(Exception):
    """与 openai SDK 同名，按类名识别为可重试错误。"""


@pytest.fixture
def delays(monkeypatch):
    seen = []
    monkeypatch.setattr("mimir.graph.time.sleep", seen.append)
    return seen


def stream(graph, text="你好"):
    events, final = [], None
    for mode, data in graph.stream({"messages": [HumanMessage(text)]}, run_config(), stream_mode=["custom", "values"]):
        if mode == "custom":
            events.append(data)
        else:
            final = data
    return events, final


def test_retry_then_success(make_graph, delays):
    model = ScriptedChatModel(script=[APIConnectionError("断了"), APIConnectionError("又断了"), "终于好了"])
    events, final = stream(make_graph(model, max_stream_retries=5))
    assert final["messages"][-1].content == "终于好了"
    assert [type(m) for m in final["messages"]] == [HumanMessage, AIMessage]  # 失败的请求不留消息
    assert delays == [1, 2]
    retries = [e for e in events if e.get("type") == "retry"]
    assert [(e["attempt"], e["max"], e["delay"]) for e in retries] == [(1, 5, 1), (2, 5, 2)]
    assert "APIConnectionError" in retries[0]["error"]
    # 每次重发的请求内容一致
    assert [m.content for m in model.calls[0]] == [m.content for m in model.calls[2]]


def test_non_transient_error_not_retried(make_graph, delays):
    model = ScriptedChatModel(script=[ValueError("参数错误")])
    with pytest.raises(ValueError, match="参数错误"):
        stream(make_graph(model))
    assert len(model.calls) == 1
    assert delays == []


def test_retries_exhausted(make_graph, delays):
    model = ScriptedChatModel(script=[APIConnectionError(str(i)) for i in range(3)])
    with pytest.raises(APIConnectionError):
        stream(make_graph(model, max_stream_retries=2))
    assert len(model.calls) == 3
    assert delays == [1, 2]


def test_retry_disabled(make_graph, delays):
    model = ScriptedChatModel(script=[APIConnectionError("断了")])
    with pytest.raises(APIConnectionError):
        stream(make_graph(model, max_stream_retries=0))
    assert len(model.calls) == 1


def test_backoff_capped(make_graph, delays):
    model = ScriptedChatModel(script=[*[APIConnectionError(str(i)) for i in range(6)], "好"])
    stream(make_graph(model, max_stream_retries=6))
    assert delays == [1, 2, 4, 8, 8, 8]
