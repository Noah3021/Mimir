"""输入框固定在底部的界面：用管道输入模拟按键，记录界面写到终端的内容。"""
from __future__ import annotations

import io
import re
import threading

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from mimir.agent import AgentRuntime
from mimir.cli import AgentCLI, banner_renderable
from mimir.config import load_settings
from mimir.tui import FixedInputUI, context_bar
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
  transcripts: {enabled: false}
cli:
  stream: true
  history_file: ""
logging:
  file: ""
"""

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


class RecordingOutput(DummyOutput):
    """不连终端的输出：记下界面打印到上方的内容。"""

    def __init__(self) -> None:
        self.raw: list[str] = []

    def write_raw(self, data: str) -> None:
        self.raw.append(data)

    def get_size(self) -> Size:
        return Size(rows=30, columns=100)

    def text(self) -> str:
        return ANSI_RE.sub("", "".join(self.raw))


@pytest.fixture
def model():
    return ScriptedChatModel(responder=lambda m: "来自 m1")


@pytest.fixture
def cli(tmp_path, workspace, model, monkeypatch):
    monkeypatch.setattr("mimir.agent.create_llm", lambda mc, **kw: model)
    monkeypatch.setattr(AgentCLI, "_make_prompt_session", lambda self: None)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    rt = AgentRuntime(load_settings(cfg, workspace=workspace))
    console = Console(file=io.StringIO(), width=100, force_terminal=False, color_system=None)
    yield AgentCLI(rt, console)
    rt.close()


def drive(cli, *steps, timeout=20):
    """启动界面，在另一个线程里依次执行 steps（每步拿到 (ui, pipe)），最后退出界面。返回界面输出。"""
    out = RecordingOutput()
    errors: list[BaseException] = []
    with create_pipe_input() as pipe:
        ui = FixedInputUI(cli, input=pipe, output=out)

        def script():
            try:
                assert ui._started.wait(10), "界面没有启动"
                for step in steps:
                    step(ui, pipe)
            except BaseException as e:  # 断言失败也要退出界面，再在主线程里抛出
                errors.append(e)
            finally:
                ui.exit_threadsafe()

        t = threading.Thread(target=script, daemon=True)
        t.start()
        ui.run()
        t.join(timeout)
    if errors:
        raise errors[0]
    return out.text()


def send(text, n):
    """输入 text 并回车，等前 n 条输入都处理完。"""
    def step(ui, pipe):
        pipe.send_text(text + "\r")
        assert ui.wait_idle(n), f"第 {n} 条输入没有处理完"
    return step


# ---------------- 纯函数 ----------------
def test_context_bar_levels():
    def plain(frags):
        return "".join(t for _, t, *_ in frags)

    assert plain(context_bar(None)) == "上下文 —"
    low = context_bar((10_000, 100_000), width=10)
    assert plain(low) == "上下文 █░░░░░░░░░ 10% · 10.0k/100.0k"
    assert low[1][0] == "class:ctx-low"
    assert context_bar((60_000, 100_000))[1][0] == "class:ctx-mid"
    assert context_bar((95_000, 100_000))[1][0] == "class:ctx-high"
    assert plain(context_bar((200_000, 100_000), width=4)).startswith("上下文 ████ 200%")


def render(renderable, width):
    c = Console(file=io.StringIO(), width=width, force_terminal=False, color_system=None)
    c.print(renderable)
    return c.file.getvalue()


def test_banner_shows_logo_when_wide():
    out = render(banner_renderable("Mimir", "0.2.0", "model-one", "high", "~\\proj", width=100), 100)
    lines = out.splitlines()
    assert len(lines) == 6
    assert lines[0].startswith("███╗")
    assert "Mimir v0.2.0" in out and "model-one · 推理 high" in out and "~\\proj" in out


@pytest.mark.parametrize("width,unicode", [(50, True), (100, False)])
def test_banner_falls_back_to_text(width, unicode):
    out = render(banner_renderable("Mimir", "0.2.0", "model-one", "high", "~", width=width, unicode=unicode), width)
    assert "█" not in out
    assert out.splitlines()[0] == "Mimir v0.2.0"


# ---------------- 界面 ----------------
def test_message_is_echoed_and_answered(cli):
    text = drive(cli, send("你好", 1))
    assert "› 你好" in text
    assert "● 来自 m1" in text
    assert text.index("› 你好") < text.index("来自 m1")
    assert "再见" in cli.console.file.getvalue()  # 退出后恢复原来的 Console


def test_queued_messages_run_in_order(cli, model):
    gate = threading.Event()

    def slow(messages):
        gate.wait(10)
        return "第一条的回答"

    model.script = [slow, "第二条的回答"]

    def step(ui, pipe):
        pipe.send_text("第一条\r")
        pipe.send_text("第二条\r")
        gate.set()
        assert ui.wait_idle(2)

    text = drive(cli, step)
    assert text.index("第一条的回答") < text.index("› 第二条") < text.index("第二条的回答")
    assert len(model.calls) == 2


def test_slash_command_runs_in_worker(cli):
    text = drive(cli, send("/approve off", 1))
    assert cli.rt.approval.enabled is False
    assert "已关闭危险操作确认" in text


def test_escape_interrupts_turn(cli, model):
    started, gate = threading.Event(), threading.Event()

    def slow(messages):
        started.set()
        gate.wait(10)
        return "不该显示"

    model.script = [slow]

    def step(ui, pipe):
        pipe.send_text("你好\r")
        assert started.wait(10)
        pipe.send_text("\x1b")  # Esc
        for _ in range(100):
            if ui.cli._runner is not None and ui.cli._runner.cancelled:
                break
            threading.Event().wait(0.05)
        gate.set()
        assert ui.wait_idle(1)

    text = drive(cli, step)
    assert "已中断本轮回答" in text


def test_approval_prompt_in_terminal(cli, model, workspace, monkeypatch):
    model.script = [tool_call("write_file", {"path": "a.txt", "content": "hi"}), "写好了"]
    monkeypatch.setattr(cli.console, "input", lambda prompt="", **kw: "y")
    text = drive(cli, send("写个文件", 1))
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hi"
    assert "需要确认：write_file" in text
    assert "写好了" in text


def test_exit_word_quits(cli):
    out = RecordingOutput()
    with create_pipe_input() as pipe:
        ui = FixedInputUI(cli, input=pipe, output=out)
        threading.Thread(target=lambda: (ui._started.wait(10), pipe.send_text("/exit\r")), daemon=True).start()
        ui.run()  # 不退出的话测试会超时
    assert "再见" in cli.console.file.getvalue()
