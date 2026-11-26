"""AgentCLI：斜杠命令分发、一轮对话（含工具确认）、main() 入口和 `python -m mimir` 冒烟测试。

模型换成假模型，Console 输出写进 StringIO，不用真实终端。
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

from mimir import cli as cli_mod
from mimir.agent import AgentRuntime
from mimir.cli import AgentCLI, main
from mimir.config import ConfigError, load_settings
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
  m2: {provider: openai, model: model-two, api_key: x}
tools:
  require_approval: [write_file]
memory:
  db_path: data/sessions.db
  transcripts: {enabled: true, dir: data/transcripts}
cli:
  stream: false
  history_file: ""
logging:
  file: ""
"""

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def models():
    return {"m1": ScriptedChatModel(responder=lambda m: "来自 m1"),
            "m2": ScriptedChatModel(responder=lambda m: "来自 m2")}


@pytest.fixture
def config_file(tmp_path, workspace, models, monkeypatch):
    monkeypatch.setattr("mimir.agent.create_llm", lambda mc, **kw: models[mc.name])
    # 测试里没有真实终端，不创建 prompt_toolkit 会话
    monkeypatch.setattr(AgentCLI, "_make_prompt_session", lambda self: None)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    return cfg


@pytest.fixture
def app(config_file, workspace):
    rt = AgentRuntime(load_settings(config_file, workspace=workspace))
    console = Console(file=io.StringIO(), width=120, force_terminal=False, color_system=None)
    yield AgentCLI(rt, console)
    rt.close()


def out(app: AgentCLI) -> str:
    return app.console.file.getvalue()


def answers(monkeypatch, app, *replies):
    """依次回答 console.input 的提问。"""
    it = iter(replies)
    monkeypatch.setattr(app.console, "input", lambda prompt="": next(it))


# ---------------- 斜杠命令 ----------------
def test_unknown_command(app):
    app.handle_command("/nope")
    assert "未知命令 /nope" in out(app)


@pytest.mark.parametrize("cmd", ["/help", "/h", "/?"])
def test_help_lists_commands(app, cmd):
    app.handle_command(cmd)
    text = out(app)
    for c in ("/new", "/resume", "/usage", "/exit"):
        assert c in text


@pytest.mark.parametrize("cmd", ["/tools", "/config", "/usage", "/model", "/think", "/sessions", "/history"])
def test_info_commands_do_not_fail(app, cmd):
    app.handle_command(cmd)
    assert "命令执行失败" not in out(app)


def test_command_exception_is_reported(app, monkeypatch):
    def boom(_):
        raise RuntimeError("坏了")

    monkeypatch.setattr(app, "cmd_tools", boom)
    app.handle_command("/tools")
    assert "命令执行失败：坏了" in out(app)


def test_model_switch(app):
    app.handle_command("/model m2")
    assert app.rt.model_name == "m2"
    assert "已切换到" in out(app)
    app.handle_command("/model m2")
    assert "当前已是 m2" in out(app)


def test_model_unknown(app):
    app.handle_command("/model nope")
    assert app.rt.model_name == "m1"


def test_think(app):
    app.handle_command("/think high")
    assert app.rt.reasoning_level == "high"
    app.handle_command("/think extreme")
    assert app.rt.reasoning_level == "high"


def test_approve_toggle(app):
    app.handle_command("/approve off")
    assert app.rt.approval.enabled is False
    app.handle_command("/approve on")
    assert app.rt.approval.enabled is True
    app.handle_command("/approve")
    assert "当前为：开启" in out(app)


@pytest.mark.parametrize("arg", ["0", "-1", "abc"])
def test_history_rejects_bad_count(app, arg):
    app.handle_command(f"/history {arg}")
    assert "条数必须是正整数" in out(app)


def test_new_session_changes_thread(app):
    before = app.rt.thread_id
    app.handle_command("/new")
    assert app.rt.thread_id != before


# ---------------- 一轮对话 ----------------
def test_run_turn_prints_answer_and_saves_session(app):
    app.run_turn("你好")
    assert "来自 m1" in out(app)
    assert [s.thread_id for s in app.rt.store.list(10)] == [app.rt.thread_id]


def test_sessions_resume_and_delete(app, monkeypatch):
    app.run_turn("第一个会话")
    first = app.rt.thread_id
    app.handle_command("/new")
    app.run_turn("第二个会话")

    app.handle_command("/sessions")
    assert len(app.listing) == 2
    idx = [s.thread_id for s in app.listing].index(first) + 1
    app.handle_command(f"/resume {idx}")
    assert app.rt.thread_id == first
    assert "第一个会话" in out(app)

    answers(monkeypatch, app, "y")
    app.handle_command(f"/delete {first}")
    assert first not in [s.thread_id for s in app.rt.store.list(10)]
    assert app.rt.thread_id != first  # 删的是当前会话，自动开新会话


def test_run_turn_approval_allow(app, models, monkeypatch, workspace):
    models["m1"].script = [tool_call("write_file", {"path": "a.txt", "content": "hi"}), "写好了"]
    answers(monkeypatch, app, "y")
    app.run_turn("写个文件")
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hi"
    assert "需要确认：write_file" in out(app)
    assert "写好了" in out(app)


def test_run_turn_approval_deny_with_feedback(app, models, monkeypatch, workspace):
    models["m1"].script = [tool_call("write_file", {"path": "a.txt", "content": "hi"}), "好的，不写了"]
    answers(monkeypatch, app, "别写这个文件")
    app.run_turn("写个文件")
    assert not (workspace / "a.txt").exists()
    tool_msg = models["m1"].calls[-1][-1]
    assert "别写这个文件" in str(tool_msg.content)


def test_run_turn_always_allow(app, models, monkeypatch):
    models["m1"].script = [tool_call("write_file", {"path": "a.txt", "content": "1"}), "ok"]
    answers(monkeypatch, app, "a")
    app.run_turn("写")
    assert "write_file" in app.rt.approval.session_allowed


def test_run_turn_model_error_is_reported(app, models):
    models["m1"].script = [RuntimeError("接口挂了")]
    app.run_turn("你好")
    assert "出错：RuntimeError: 接口挂了" in out(app)


# ---------------- main() ----------------
def test_main_single_prompt(config_file, workspace, capsys):
    assert main(["-c", str(config_file), "-w", str(workspace), "-p", "你好"]) == 0
    assert "来自 m1" in capsys.readouterr().out


def test_main_model_option(config_file, workspace, capsys):
    assert main(["-c", str(config_file), "-w", str(workspace), "-m", "m2", "-p", "你好"]) == 0
    assert "来自 m2" in capsys.readouterr().out


def test_main_config_error(tmp_path, capsys):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("models:\n  m1: {provider: openai, model: x}\nagent: {default_model: m1}\n", encoding="utf-8")
    assert main(["-c", str(cfg), "-w", str(tmp_path), "-p", "hi"]) == 1
    assert "配置错误" in capsys.readouterr().out


def test_main_falls_back_to_available_model(tmp_path, workspace, models, monkeypatch, capsys):
    def fake_create_llm(mc, **kw):  # 和真实的 create_llm 一样：配置不完整时报 ConfigError
        if not mc.available:
            raise ConfigError(f"模型 '{mc.name}' 配置不完整")
        return models[mc.name]

    monkeypatch.setattr("mimir.agent.create_llm", fake_create_llm)
    monkeypatch.setattr(AgentCLI, "_make_prompt_session", lambda self: None)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG.replace("m1: {provider: openai, model: model-one, api_key: x}",
                                  "m1: {provider: openai, model: model-one}"), encoding="utf-8")
    assert main(["-c", str(cfg), "-w", str(workspace), "-p", "你好"]) == 0
    printed = capsys.readouterr().out
    assert "已改用模型 m2" in printed and "来自 m2" in printed


# ---------------- 冒烟测试：真实入口 ----------------
@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_module_entrypoint(flag):
    proc = subprocess.run([sys.executable, "-m", "mimir", flag], cwd=ROOT, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"mimir" in proc.stdout.lower()


def test_cli_module_imports():
    assert callable(cli_mod.main)
