"""TurnRunner：一轮对话产生的事件、危险操作确认回调、中断和出错。"""
from __future__ import annotations

import pytest

from mimir import turn
from mimir.agent import AgentRuntime
from mimir.config import load_settings
from tests.fakes import ScriptedChatModel, tool_call

CONFIG = """
agent:
  name: Mimir
  default_model: m1
  system_prompt: "测试"
  project:
    git_status: false
models:
  m1: {provider: openai, model: model-one, api_key: x}
tools:
  require_approval: [write_file]
memory:
  db_path: data/sessions.db
  transcripts: {enabled: true, dir: data/transcripts}
cli:
  history_file: ""
logging:
  file: ""
"""


@pytest.fixture
def model():
    return ScriptedChatModel(responder=lambda m: "你好呀")


@pytest.fixture
def rt(tmp_path, workspace, model, monkeypatch):
    monkeypatch.setattr("mimir.agent.create_llm", lambda mc, **kw: model)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    runtime = AgentRuntime(load_settings(cfg, workspace=workspace))
    yield runtime
    runtime.close()


def run(rt, text="你好", *, ask=None, streaming=False, setup=None):
    events: list = []
    asked: list[dict] = []

    def default_ask(value):
        asked.append(value)
        return {"approved": True}, False

    runner = turn.TurnRunner(rt, events.append, ask or default_ask, streaming=streaming)
    if setup:
        setup(runner)
    end = runner.run(text)
    return end, events, asked


def texts(events) -> str:
    return "".join(e.text for e in events if isinstance(e, turn.TextDelta))


@pytest.mark.parametrize("streaming", [False, True])
def test_plain_answer(rt, streaming):
    end, events, _ = run(rt, streaming=streaming)
    assert isinstance(events[0], turn.Status) and events[0].text == turn.THINKING
    assert texts(events) == "你好呀"
    assert events[-1] is end
    assert end.error is None and not end.interrupted
    assert end.usage is not None


def test_tool_call_with_approval(rt, model, workspace):
    model.script = [tool_call("write_file", {"path": "a.txt", "content": "hi"}), "写好了"]
    end, events, asked = run(rt, "写个文件")
    assert [v["tool"] for v in asked] == ["write_file"]
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hi"
    kinds = [type(e) for e in events]
    assert kinds.index(turn.ToolCall) < kinds.index(turn.ToolResult)
    assert any(isinstance(e, turn.Status) and e.text == turn.RUNNING_TOOLS for e in events)
    assert texts(events) == "写好了"
    assert end.error is None


def test_always_allow_applies_after_turn(rt, model):
    model.script = [tool_call("write_file", {"path": "a.txt", "content": "1"}), "ok"]
    seen = []

    def ask(value):
        seen.append(rt.approval.needs("write_file"))
        return {"approved": True}, True

    run(rt, ask=ask)
    assert seen == [True]  # 本轮内策略不变
    assert "write_file" in rt.approval.session_allowed


def test_cancel_before_run(rt, model):
    end, events, _ = run(rt, setup=lambda r: r.cancel())
    assert end.interrupted
    assert model.calls == []
    assert events == [end]


def test_cancel_during_approval_skips_tool(rt, model, workspace):
    model.script = [tool_call("write_file", {"path": "a.txt", "content": "hi"}), "写好了"]
    holder = {}

    def ask(value):
        holder["runner"].cancel()  # 例如用户在确认框里按了 Esc
        return {"approved": True}, True

    end, _, _ = run(rt, ask=ask, setup=lambda r: holder.setdefault("runner", r))
    assert end.interrupted
    assert not (workspace / "a.txt").exists()
    assert "write_file" not in rt.approval.session_allowed


def test_keyboard_interrupt_counts_as_interrupted(rt, model):
    model.script = [KeyboardInterrupt()]
    end, _, _ = run(rt)
    assert end.interrupted and end.error is None


def test_model_error_is_reported(rt, model):
    model.script = [RuntimeError("接口挂了")]
    end, _, _ = run(rt)
    assert isinstance(end.error, RuntimeError)
    assert not end.interrupted
