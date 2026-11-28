"""输入框固定在底部的界面：用管道输入模拟按键，记录界面写到终端的内容。"""
from __future__ import annotations

import io
import re
import threading
import time

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from mimir.agent import AgentRuntime
from mimir.cli import AgentCLI, banner_renderable
from mimir.config import load_settings
from mimir.tui import FixedInputUI, cache_text, context_bar
from mimir.usage import Usage
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



def plain(frags) -> str:
    return "".join(t for _, t, *_ in frags)


def test_cache_text():
    assert plain(cache_text(None)) == "缓存 —"
    assert plain(cache_text(Usage(input=0))) == "缓存 —"
    assert plain(cache_text(Usage(input=4000, cache_read=3000))) == "缓存 75% · 命中 3.0k"
    assert plain(cache_text(Usage(input=1600))) == "缓存 0% · 命中 0"


def test_cache_text_levels():
    """命中率 ≥60% 青色，30%~60% 浅灰，更低黄色。"""
    assert cache_text(Usage(input=100, cache_read=60))[1][0] == "class:cache-high"
    assert cache_text(Usage(input=100, cache_read=59))[1][0] == "class:cache-mid"
    assert cache_text(Usage(input=100, cache_read=30))[1][0] == "class:cache-mid"
    assert cache_text(Usage(input=100, cache_read=29))[1][0] == "class:cache-low"
    assert cache_text(Usage(input=100))[1][0] == "class:cache-low"


def test_cache_text_is_session_cumulative():
    """会话累计：一次没命中不会把整体拉到 0%。"""
    session = Usage()
    for u in (Usage(input=4000, cache_read=3000), Usage(input=4000), Usage(input=4000, cache_read=3500)):
        session.add(u)
    assert plain(cache_text(session)) == "缓存 54% · 命中 6.5k"


def test_footer_order(cli):
    """底部一行从左到右：审批 │ 上下文 │ 缓存，紧挨着排，不再把进度条推到最右边。"""
    ui = FixedInputUI(cli, output=RecordingOutput())
    ui.footer_state = (True, (16_000, 128_000), Usage(input=16_000, cache_read=12_000))
    text = plain(ui._footer_fragments())
    assert text.startswith("  审批：开启  │  上下文 ")
    assert text.index("上下文") < text.index("缓存 75% · 命中 12.0k")
    assert "   " not in text.split("│")[-1].strip()  # 末尾没有填充空格
    ui.footer_state = (False, None, None)
    assert plain(ui._footer_fragments()) == "  审批：关闭  │  上下文 —  │  缓存 —"


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


SHIFT_TAB = "\x1b[Z"


def test_shift_tab_toggles_approval(cli):
    """空闲时 Shift+Tab 开关审批，底部状态栏和提示同步变化；开启时清掉“始终允许”。"""
    approval = cli.rt.approval
    approval.allow("write_file")
    seen = []

    def press(ui, pipe):
        pipe.send_text(SHIFT_TAB)
        assert wait_for(lambda: ui.cli.rt.approval.enabled is not seen[-1])
        seen.append(ui.cli.rt.approval.enabled)

    def check_off(ui, pipe):
        assert wait_for(lambda: ui.footer_state[0] is False)
        assert "审批：关闭" in plain(ui._footer_fragments())
        assert "已关闭危险操作确认" in ui.hint
        assert "write_file" in approval.session_allowed  # 关闭时不动“始终允许”

    def check_on(ui, pipe):
        assert wait_for(lambda: ui.footer_state[0] is True)
        assert "审批：开启" in plain(ui._footer_fragments())
        assert "已开启危险操作确认" in ui.hint
        assert not approval.session_allowed

    seen.append(True)
    drive(cli, press, check_off, press, check_on)
    assert approval.enabled is True


def test_shift_tab_refused_while_busy(cli, model):
    """回答期间同一轮内策略不能变，Shift+Tab 只提示不切换。"""
    started, gate = threading.Event(), threading.Event()

    def slow(messages):
        started.set()
        gate.wait(10)
        return "好了"

    model.script = [slow]

    def step(ui, pipe):
        pipe.send_text("你好\r")
        assert started.wait(10)
        pipe.send_text(SHIFT_TAB)
        assert wait_for(lambda: "回答期间不能切换审批" in ui.hint)
        assert cli.rt.approval.enabled is True
        gate.set()
        assert ui.wait_idle(1)

    drive(cli, step)
    assert cli.rt.approval.enabled is True


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


# ---------------- 危险操作确认 ----------------
WRITE = tool_call("write_file", {"path": "a.txt", "content": "hi"})


def wait_for(cond, timeout=10) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def answer_approval(keys, wait_done=True):
    """发一条会触发确认的消息，等底部出现选择框后按下 keys。"""
    def step(ui, pipe):
        pipe.send_text("写个文件\r")
        assert wait_for(lambda: ui.approval is not None), "没有出现确认框"
        if keys:
            pipe.send_text(keys)
        if wait_done:
            assert ui.wait_idle(1)
    return step


@pytest.mark.parametrize("keys", ["1", "\r"])  # 回车 = 选中的第一项“允许”
def test_approval_allow(cli, model, workspace, keys):
    model.script = [WRITE, "写好了"]
    text = drive(cli, answer_approval(keys))
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hi"
    assert "需要确认：write_file" in text  # 详情打印在消息区
    assert "└ 已允许" in text
    assert "写好了" in text
    assert "write_file" not in cli.rt.approval.session_allowed


def test_approval_always(cli, model):
    model.script = [WRITE, "ok"]
    text = drive(cli, answer_approval("2"))
    assert "本次运行不再询问 write_file" in text
    assert "write_file" in cli.rt.approval.session_allowed


def test_approval_reject_with_arrow_keys(cli, model, workspace):
    model.script = [WRITE, "好的"]
    text = drive(cli, answer_approval("\x1b[B\x1b[B\x1b[B\x1b[A\r"))  # 下 3 次回到第 1 项，再上 1 次绕到最后的“拒绝”
    assert not (workspace / "a.txt").exists()
    assert "└ 已拒绝" in text


def test_approval_feedback(cli, model, workspace):
    model.script = [WRITE, "好的，不写了"]
    text = drive(cli, answer_approval("别写这个文件\r"))
    assert not (workspace / "a.txt").exists()
    assert "已拒绝，反馈：别写这个文件" in text
    assert "别写这个文件" in str(model.calls[-1][-1].content)


def test_approval_escape_interrupts(cli, model, workspace):
    model.script = [WRITE]
    text = drive(cli, answer_approval("\x1b"))
    assert not (workspace / "a.txt").exists()
    assert "└ 已中断" in text and "已中断本轮回答" in text
    assert len(model.calls) == 1


def test_approval_keeps_draft(cli, model):
    """确认框出现前输入框里没发出去的内容，确认完要还回来，不能被当成反馈。"""
    gate = threading.Event()

    def slow(messages):
        gate.wait(10)
        return WRITE

    model.script = [slow, "写好了"]
    seen = {}

    def step(ui, pipe):
        pipe.send_text("写个文件\r")
        pipe.send_text("草稿")
        assert wait_for(lambda: ui.buffer.text == "草稿")
        gate.set()
        assert wait_for(lambda: ui.approval is not None)
        assert ui.buffer.text == ""
        pipe.send_text("1")
        assert ui.wait_idle(1)
        seen["draft"] = ui.buffer.text

    text = drive(cli, step)
    assert seen["draft"] == "草稿"
    assert "└ 已允许" in text


def test_approval_letters_go_to_feedback(cli, model, workspace):
    """英文反馈的首字母不能被当成快捷键：a 开头的反馈不能变成“始终允许”。"""
    model.script = [WRITE, "好的"]
    text = drive(cli, answer_approval("actually no\r"))
    assert not (workspace / "a.txt").exists()
    assert "已拒绝，反馈：actually no" in text
    assert "write_file" not in cli.rt.approval.session_allowed


def test_exit_while_waiting_for_approval(cli, model, workspace):
    model.script = [WRITE]
    drive(cli, answer_approval("", wait_done=False))  # 确认框出现后直接退出界面
    assert not (workspace / "a.txt").exists()


def test_exit_word_quits(cli):
    out = RecordingOutput()
    with create_pipe_input() as pipe:
        ui = FixedInputUI(cli, input=pipe, output=out)
        threading.Thread(target=lambda: (ui._started.wait(10), pipe.send_text("/exit\r")), daemon=True).start()
        ui.run()  # 不退出的话测试会超时
    assert "再见" in cli.console.file.getvalue()
