"""AgentRuntime：组装、中断修复、切换模型、MIMIR.md 变化检测。create_llm 替换为假模型。"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mimir.agent import AgentRuntime
from mimir.config import load_settings
from tests.fakes import ScriptedChatModel, tool_call

CONFIG = """
agent:
  default_model: m1
  system_prompt: "当前模型：{model}"
  project:
    git_status: false
models:
  m1: {provider: openai, model: model-one, api_key: x}
  m2: {provider: openai, model: model-two, api_key: x}
tools:
  require_approval: [write_file]
memory:
  db_path: data/sessions.db
  transcripts: {enabled: true, dir: data/transcripts}
cli:
  stream: false
logging:
  file: ""
"""


@pytest.fixture
def models():
    return {"m1": ScriptedChatModel(responder=lambda m: "来自 m1"),
            "m2": ScriptedChatModel(responder=lambda m: "来自 m2")}


@pytest.fixture
def rt(tmp_path, workspace, models, monkeypatch):
    monkeypatch.setattr("mimir.agent.create_llm", lambda mc, **kw: models[mc.name])
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    runtime = AgentRuntime(load_settings(cfg, workspace=workspace))
    yield runtime
    runtime.close()


def turn(rt, text):
    rt.begin_turn()
    rt.graph.invoke({"messages": [HumanMessage(text)]}, rt.run_config())
    rt.sync_transcript()
    return rt.finish_turn()


def test_basic_turn(rt, models):
    turn(rt, "你好")
    assert [m.content for m in rt.messages()] == ["你好", "来自 m1"]
    assert models["m1"].calls[0][0].content == "当前模型：m1（model-one）"
    assert rt.transcript.path(rt.thread_id).is_file()


def test_switch_model_keeps_history(rt, models):
    turn(rt, "第一轮")
    rt.switch_model("m2")
    turn(rt, "第二轮")
    assert [m.content for m in rt.messages()] == ["第一轮", "来自 m1", "第二轮", "来自 m2"]
    sent = models["m2"].calls[0]
    assert sent[0].content == "当前模型：m2（model-two）"  # 切换模型后重新渲染系统提示词
    assert [m.content for m in sent[1:]] == ["第一轮", "来自 m1", "第二轮"]


def test_system_prompt_stable_within_session(rt, models):
    turn(rt, "一")
    turn(rt, "二")
    assert models["m1"].calls[0][0].content == models["m1"].calls[1][0].content


def test_unknown_model(rt):
    with pytest.raises(Exception, match="未知模型"):
        rt.switch_model("m9")
    assert rt.model_name == "m1"


def test_repair_dangling_tool_calls(rt, models):
    assert rt.repair_dangling_tool_calls() == 0
    rt.graph.update_state(rt.run_config(), {"messages": [HumanMessage("写文件"), tool_call("write_file", {}, "c1")]},
                          as_node="agent")
    assert rt.repair_dangling_tool_calls() == 1
    last = rt.messages()[-1]
    assert isinstance(last, ToolMessage) and last.tool_call_id == "c1" and last.status == "error"
    assert rt.repair_dangling_tool_calls() == 0

    turn(rt, "算了")  # 修复后下一轮请求格式合法：每个工具调用都有结果
    sent = models["m1"].calls[-1]
    assert isinstance(sent[2], AIMessage) and isinstance(sent[3], ToolMessage)


def test_new_session(rt):
    turn(rt, "旧会话")
    old = rt.thread_id
    assert rt.new_session() != old
    assert rt.messages() == []


def test_check_project_update(rt, workspace):
    mimir_md = workspace / "MIMIR.md"
    mimir_md.write_text("规则 v1", encoding="utf-8")
    rt.new_session()
    assert rt.project_info().project_text == "规则 v1"
    assert rt.check_project_update() is None

    mimir_md.write_text("规则 v2", encoding="utf-8")
    assert rt.check_project_update() == {"file": "MIMIR.md", "text": "规则 v2", "truncated": False}
    assert rt.project_info().project_text == "规则 v1"  # 会话开始时的快照不变

    mimir_md.unlink()
    assert rt.check_project_update()["text"] == ""


def test_usage_recorded(rt, models):
    models["m1"].responder = lambda m: AIMessage("好", usage_metadata={
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
    u = turn(rt, "你好")
    assert (u.input, u.output, u.calls) == (10, 2, 1)
    assert rt.session_usage().input == 10
    rows = rt.usage.by_model(thread_id=rt.thread_id)
    assert rows[0].profile == "m1" and rows[0].usage.calls == 1
