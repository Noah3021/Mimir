"""输入框固定在终端底部的交互界面（参考 Claude Code）。

画面分两块：
- 上面是固定区：说完的内容（用户消息、回答、工具调用、命令输出）打印成普通的终端输出，之后不再改动，
  终端自带的滚动和复制都能用。打印时先擦掉底部的界面，打印完再重画（prompt_toolkit 的 run_in_terminal）。
- 下面是动态区：正在流式输出的内容、等待动画和状态、排队的消息、输入框、底部状态栏，一直在原地刷新。
  界面不是全屏的：它从光标位置一直铺到终端底部，顶部用一块空白把输入框压在最下面。

线程：
- 主线程跑 prompt_toolkit 的事件循环，负责输入和绘制。界面状态只在这个线程里修改。
- 后台线程按顺序处理提交的消息和命令（回答期间提交的会排队），一轮对话由 TurnRunner 执行。
  它要显示的东西都通过 call() 交给主线程：rich 的输出先写进 _AppWriter，flush 时整段交给主线程打印。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import io
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.completion import generate_completions
from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style
from rich.console import Console, ConsoleDimensions
from rich.padding import Padding
from rich.text import Text

from .cli import EXIT_WORDS, AgentCLI, stream_renderable
from .turn import THINKING
from .usage import Usage, fmt_tokens

logger = logging.getLogger("mimir")

SPINNER = "·✢✳✻✽✻✳✢"
HINT = "/help 查看命令 · Esc+Enter 换行 · Esc 中断回答 · 连按两次 Ctrl+C 退出"
INTERRUPTING = "正在中断…（当前步骤结束后停止）"

STYLE = Style.from_dict({
    "rule": "#5f5f5f",
    "prompt": "bold ansicyan",
    "dim": "#7a7a7a",
    "spinner": "bold #00afff",
    "status": "#00afff",
    "queued": "#8a8a8a",
    "approval": "#8a8a8a",
    "approval-off": "ansiyellow",
    "ctx-low": "#8a8a8a",
    "ctx-mid": "ansiyellow",
    "ctx-high": "ansired",
    "ctx-empty": "#444444",
    "cache": "#5fafaf",
})


def context_bar(usage: tuple[int, int] | None, width: int = 20) -> StyleAndTextTuples:
    """上下文用量进度条：上下文 ███████░░░░░ 34% · 43.5k/128k。

    usage 是 (已用 token, 上下文窗口)，还没调用过模型时为 None。占用低于 50% 灰色，50%~80% 黄色，更高红色。
    """
    if not usage or usage[1] <= 0:
        return [("class:dim", "上下文 —")]
    used, window = usage
    ratio = used / window
    filled = max(0, min(width, round(ratio * width)))
    level = "low" if ratio < 0.5 else "mid" if ratio < 0.8 else "high"
    return [
        ("class:dim", "上下文 "),
        (f"class:ctx-{level}", "█" * filled),
        ("class:ctx-empty", "░" * (width - filled)),
        ("class:dim", f" {ratio:.0%} · {fmt_tokens(used)}/{fmt_tokens(window)}"),
    ]


def cache_text(last: Usage | None) -> StyleAndTextTuples:
    """最近一次模型调用的缓存命中：缓存 75% · 1.2k。还没调用过时为“缓存 —”。"""
    if last is None or last.input <= 0:
        return [("class:dim", "缓存 —")]
    return [("class:dim", "缓存 "), ("class:cache", f"{last.cache_rate:.0%}"),
            ("class:dim", f" · {fmt_tokens(last.cache_read)}")]


# 底部状态栏的数据：(审批是否开启, (上下文已用, 窗口), 最近一次调用的用量)
FooterState = tuple[bool, "tuple[int, int] | None", "Usage | None"]


class _Segment:
    """动态区里的一段流式内容（思考过程或正文），缓存渲染结果。"""

    __slots__ = ("mode", "buf", "_key", "_lines")

    def __init__(self, mode: str):
        self.mode, self.buf = mode, ""
        self._key: tuple[int, int] | None = None
        self._lines: list[str] = []


class _AppWriter(io.TextIOBase):
    """给 rich Console 用的输出文件：攒下一次 print 的内容，flush 时交给界面线程打印到输入框上方。"""

    encoding = "utf-8"

    def __init__(self, ui: FixedInputUI):
        super().__init__()
        self.ui = ui
        self._parts: list[str] = []
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        with self._lock:
            self._parts.append(s)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            text = "".join(self._parts)
            self._parts.clear()
        if text:
            self.ui.call(lambda: self.ui.print_ansi(text))


class _TuiConsole(Console):
    """界面运行期间 AgentCLI 使用的 Console：输出打印到输入框上方，宽度跟随终端，input 暂停界面后再读。"""

    def __init__(self, ui: FixedInputUI, **kw: Any):
        self._ui = ui  # 先设置：Console.__init__ 里可能就会读 size
        super().__init__(file=_AppWriter(ui), **kw)

    @property
    def size(self) -> ConsoleDimensions:
        cols, rows = self._ui.term_size()
        return ConsoleDimensions(cols, rows)

    @size.setter
    def size(self, new_size: tuple[int, int]) -> None:  # 尺寸始终跟随终端
        pass

    def input(self, prompt: Any = "", **kw: Any) -> str:  # type: ignore[override]
        real = self._ui.real_console
        return self._ui.run_in_terminal_sync(lambda: real.input(prompt, **kw))


class DynamicPrinter:
    """StreamPrinter 的替代：流式内容显示在动态区，一段结束后打印到上方。

    接口和 StreamPrinter 一样（wait / feed / discard / end），由后台线程调用，实际修改在界面线程里进行。
    """

    def __init__(self, ui: FixedInputUI, show_reasoning: bool = True):
        self.ui = ui
        self.show_reasoning = show_reasoning

    def wait(self, msg: str = THINKING) -> None:
        self.ui.call(self.ui.end_stream)
        self.ui.call(lambda: self.ui.set_status(msg))

    def feed(self, kind: str, s: str) -> None:
        if not s or (kind == "reasoning" and not self.show_reasoning):
            return
        self.ui.call(lambda: self.ui.feed(kind, s))

    def discard(self) -> None:
        self.ui.call(self.ui.discard_stream)

    def end(self) -> None:
        self.ui.call(self.ui.end_stream)


class FixedInputUI:
    def __init__(self, cli: AgentCLI, *, input: Any = None, output: Any = None):  # noqa: A002
        self.cli = cli
        self.real_console: Console = cli.console
        cs = self.real_console.color_system
        self.color_system = "truecolor" if cs == "windows" else cs  # 输出经 prompt_toolkit 走 VT 序列
        self.loop: asyncio.AbstractEventLoop | None = None
        self._started = threading.Event()
        self._ctx: contextvars.Context | None = None

        # 界面状态（只在界面线程里修改）
        self.current: _Segment | None = None  # 正在流式输出的一段
        self.flushing: list[_Segment] = []  # 已经说完、正在打印到上方的段落（打印完之前仍显示在动态区）
        self.pending: deque[str] = deque()  # 排队等待处理的输入
        self.status = THINKING
        self.running = False
        self.exiting = False
        self.turn_started = 0.0
        self.hint, self.hint_until = "", 0.0
        self._last_ctrl_c = 0.0
        self.footer_state = self._footer_state()
        self._prints: set[asyncio.Future] = set()

        # 后台线程：按顺序处理输入
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.worker = threading.Thread(target=self._work, name="mimir-agent", daemon=True)
        self._cv = threading.Condition()
        self._submitted = 0
        self._outstanding = 0
        self._waiting: set[concurrent.futures.Future] = set()

        session = cli.session
        self.buffer = Buffer(
            multiline=True,
            history=getattr(session, "history", None) or InMemoryHistory(),
            completer=WordCompleter(cli.completion_words(), ignore_case=True, sentence=True),
            # 只在输入斜杠命令时边输边补全
            complete_while_typing=Condition(lambda: self.buffer.text.startswith("/")),
        )
        self.input_window = Window(
            BufferControl(self.buffer),
            height=Dimension(min=1, max=10),
            dont_extend_height=True,
            wrap_lines=True,
            get_line_prefix=lambda line, wrap: [("class:prompt", "› ")] if line == 0 and wrap == 0 else [("", "  ")],
        )
        self.app: Application = Application(
            layout=Layout(self._layout(), focused_element=self.input_window),
            key_bindings=self._key_bindings(),
            style=STYLE,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
            min_redraw_interval=0.05,
            input=input,
            output=output,
        )
        self.app.timeoutlen = 0.3  # Esc 后等多久判断是不是 Esc+Enter
        self.app.ttimeoutlen = 0.05

    # ====================== 运行 ======================
    def run(self) -> None:
        self.cli.banner(hint=HINT)
        old_printer = self.cli.printer
        self.cli.console = _TuiConsole(self, force_terminal=True, color_system=self.color_system,
                                       legacy_windows=False)
        self.cli.printer = DynamicPrinter(self, self.cli.settings.cli.show_reasoning)
        try:
            self.app.run(pre_run=self._pre_run)
        finally:
            self.loop = None
            self.cli.console, self.cli.printer = self.real_console, old_printer
            for f in list(self._waiting):
                if not f.done():
                    f.set_exception(EOFError("界面已退出"))
        self.real_console.print("[dim]再见！[/]")

    def _pre_run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._ctx = contextvars.copy_context()  # 后台线程交过来的回调在界面线程的上下文里执行（get_app 要用）
        self.buffer.load_history_if_not_yet_loaded()
        self.worker.start()
        self.app.create_background_task(self._tick())
        self._started.set()

    def call(self, fn: Callable[[], Any]) -> None:
        """在界面线程里执行 fn 并重画，可以从任何线程调用。界面已退出时忽略。"""
        loop = self.loop
        if loop is None:
            return

        def run() -> None:
            try:
                fn()
            except Exception:
                logger.exception("界面更新失败")
            self.app.invalidate()

        try:
            ctx = self._ctx
            loop.call_soon_threadsafe(run, context=ctx.copy() if ctx is not None else None)
        except RuntimeError:  # 事件循环已关闭
            pass

    def term_size(self) -> tuple[int, int]:
        try:
            size = self.app.output.get_size()
            return max(size.columns, 20), max(size.rows, 5)
        except Exception:
            return 80, 24

    async def _tick(self) -> None:
        """回答期间定时重画，让等待动画转起来。"""
        while True:
            await asyncio.sleep(0.12)
            if self.hint and time.monotonic() >= self.hint_until:
                self.hint = ""
                self.app.invalidate()
            if self.running or self.exiting or self.hint:
                self.app.invalidate()

    # ====================== 打印到上方 ======================
    def print_ansi(self, text: str, done: Callable[[], None] | None = None) -> None:
        """（界面线程）暂时擦掉界面，把 text 打印到上方，再重画界面。按调用顺序依次打印。"""
        def write() -> None:
            out = self.app.output
            out.write_raw(text.replace("\r\n", "\n").replace("\n", "\r\n"))  # 不依赖终端把 LF 当作换行 + 回车
            out.flush()
            if done is not None:
                done()

        self._track(run_in_terminal(write))

    def run_in_terminal_sync(self, func: Callable[[], Any]) -> Any:
        """（后台线程）暂停界面，在终端里执行 func（例如读一行确认输入），返回它的结果或抛出它的异常。"""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._waiting.add(fut)

        def body() -> None:
            if fut.done():
                return
            try:
                fut.set_result(func())
            except BaseException as e:  # 包括 Ctrl+C 的 KeyboardInterrupt，交给后台线程处理
                fut.set_exception(e)

        self.call(lambda: self._track(run_in_terminal(body)))
        try:
            return fut.result()
        finally:
            self._waiting.discard(fut)

    def _track(self, task: asyncio.Future) -> None:
        self._prints.add(task)
        task.add_done_callback(self._prints.discard)

    def _render_ansi(self, renderable: Any, width: int) -> str:
        c = Console(file=io.StringIO(), force_terminal=True, color_system=self.color_system,
                    width=width, legacy_windows=False)
        c.print(renderable)
        return c.file.getvalue()  # type: ignore[attr-defined]

    # ====================== 流式内容（界面线程） ======================
    def set_status(self, text: str) -> None:
        runner = self.cli._runner
        if runner is not None and runner.cancelled:
            return  # 已经在中断，保持“正在中断”的提示
        self.status = text

    def feed(self, kind: str, s: str) -> None:
        if self.current is None or self.current.mode != kind:
            self.end_stream()
            self.current = _Segment(kind)
        self.current.buf += s

    def end_stream(self) -> None:
        seg, self.current = self.current, None
        if seg is None or not seg.buf.strip():
            return
        self.flushing.append(seg)
        text = self._render_ansi(stream_renderable(seg.mode, seg.buf), self.term_size()[0])
        self.print_ansi(text, done=lambda: self.flushing.remove(seg))

    def discard_stream(self) -> None:
        self.current = None

    def _stream_lines(self) -> list[str]:
        width, rows = self.term_size()
        segs = [*self.flushing, *([self.current] if self.current and self.current.buf.strip() else [])]
        lines: list[str] = []
        for seg in segs:
            key = (len(seg.buf), width)
            if seg._key != key:
                seg._lines = self._render_ansi(stream_renderable(seg.mode, seg.buf), width).rstrip("\n").split("\n")
                seg._key = key
            lines.extend(seg._lines)
        # 动态区不能超过一屏，否则顶部滚出屏幕后无法擦除
        limit = max(3, rows - 6 - len(self.pending) - self.buffer.document.line_count)
        return lines[-limit:]

    # ====================== 布局 ======================
    def _layout(self) -> FloatContainer:
        rule = lambda: Window(char="─", height=1, style="class:rule")  # noqa: E731
        body = HSplit([
            Window(),  # 空白，把下面的内容压到终端底部
            ConditionalContainer(
                Window(FormattedTextControl(lambda: ANSI("\n".join(self._stream_lines()))),
                       height=lambda: Dimension.exact(len(self._stream_lines())), wrap_lines=False),
                filter=Condition(lambda: bool(self._stream_lines())),
            ),
            ConditionalContainer(Window(FormattedTextControl(self._status_fragments), height=1),
                                 filter=Condition(lambda: self.running or self.exiting)),
            ConditionalContainer(Window(FormattedTextControl(self._queued_fragments),
                                        height=lambda: Dimension.exact(len(self.pending))),
                                 filter=Condition(lambda: bool(self.pending))),
            rule(),
            self.input_window,
            rule(),
            Window(FormattedTextControl(self._footer_fragments), height=1),
        ])
        return FloatContainer(body, floats=[
            Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=8, scroll_offset=1)),
        ])

    def _status_fragments(self) -> StyleAndTextTuples:
        if self.exiting:
            return [("class:status", "  正在退出（等待当前步骤结束）…")]
        now = time.monotonic()
        frame = SPINNER[int(now * 8) % len(SPINNER)]
        return [("class:spinner", f"{frame} "), ("class:status", self.status),
                ("class:dim", f"  ({int(now - self.turn_started)}s · Esc 中断)")]

    def _queued_fragments(self) -> StyleAndTextTuples:
        width = self.term_size()[0]
        out: StyleAndTextTuples = []
        for i, item in enumerate(self.pending):
            if i:
                out.append(("", "\n"))
            first = item.split("\n")[0]
            out += [("class:dim", "  ⧗ 排队 "), ("class:queued", first[: max(width - 12, 10)])]
        return out

    def _footer_fragments(self) -> StyleAndTextTuples:
        """底部一行，从左到右：审批状态 │ 上下文进度条 │ 缓存命中（│ 临时提示）。"""
        enabled, usage, last = self.footer_state
        sep = ("class:rule", "  │  ")
        out: StyleAndTextTuples = [("class:approval", "  审批：开启")] if enabled \
            else [("class:approval-off", "  审批：关闭")]
        out += [sep, *context_bar(usage), sep, *cache_text(last)]
        if self.hint:
            out += [sep, ("class:ctx-mid", self.hint)]
        return out

    # ====================== 按键 ======================
    def _key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter", eager=True)
        def _(event):
            self.submit()

        @kb.add("escape", "enter")  # Esc+Enter（或 Alt+Enter）换行
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("escape")
        def _(event):
            if self.buffer.complete_state:
                self.buffer.cancel_completion()
            elif self.running:
                self.interrupt()

        @kb.add("c-c")
        def _(event):
            now = time.monotonic()
            if self.running:
                self.interrupt()
            elif self.buffer.text:
                self.buffer.reset()
            elif now - self._last_ctrl_c < 2:
                self.request_exit()
            else:
                self._last_ctrl_c = now
                self.hint, self.hint_until = "再按一次 Ctrl+C 退出", now + 2

        @kb.add("c-d")
        def _(event):
            if not self.buffer.text:
                self.request_exit()

        kb.add("tab")(generate_completions)
        return kb

    def submit(self) -> None:
        line = self.buffer.text.strip()
        if not line:
            return
        self.buffer.reset(append_to_history=True)
        if line.lower() in EXIT_WORDS:
            self.request_exit()
            return
        self.pending.append(line)
        with self._cv:
            self._submitted += 1
            self._outstanding += 1
        self.queue.put(line)

    def interrupt(self) -> None:
        runner = self.cli._runner
        if runner is not None and not runner.cancelled:
            runner.cancel()
            self.status = INTERRUPTING

    def request_exit(self) -> None:
        """（界面线程）退出：中断当前回答，丢弃排队的输入，等后台线程结束、输出打印完再退出界面。"""
        if self.exiting:
            return
        self.exiting = True
        self.interrupt()
        self.pending.clear()
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                with self._cv:
                    self._outstanding -= 1
                    self._cv.notify_all()
        self.queue.put(None)
        self.app.create_background_task(self._shutdown())

    async def _shutdown(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.worker.join, 60)
        await asyncio.sleep(0)  # 让后台线程最后提交的打印先排进去
        while self._prints:
            await asyncio.gather(*list(self._prints), return_exceptions=True)
        self.app.exit()

    def exit_threadsafe(self) -> None:
        self._started.wait(10)
        self.call(self.request_exit)

    def wait_idle(self, submitted: int, timeout: float = 10) -> bool:
        """等到至少提交了 submitted 条输入、并且都处理完。测试用。"""
        with self._cv:
            return self._cv.wait_for(lambda: self._submitted >= submitted and self._outstanding == 0, timeout)

    # ====================== 后台线程 ======================
    def _work(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            self.call(lambda item=item: self._begin(item))
            try:
                if item.startswith("/"):
                    self.cli.handle_command(item)
                else:
                    self._echo(item)
                    self.cli.run_turn(item)
            except Exception as e:
                logger.exception("处理输入失败：%s", item)
                self.cli.console.print(f"[red]出错：{type(e).__name__}: {e}[/]", markup=True)
            state = self._footer_state()
            self.call(lambda state=state: self._end(state))
            with self._cv:
                self._outstanding -= 1
                self._cv.notify_all()

    def _begin(self, item: str) -> None:
        if self.pending:
            self.pending.popleft()
        if not item.startswith("/"):
            self.running, self.turn_started, self.status = True, time.monotonic(), THINKING

    def _end(self, state: FooterState) -> None:
        self.running = False
        self.footer_state = state

    def _echo(self, text: str) -> None:
        """把用户消息打印到消息区：灰底整行，› 开头。"""
        first, *rest = text.split("\n")
        body = Text.assemble(("› ", "bold cyan"), first)
        for line in rest:
            body.append("\n  " + line)
        self.cli.console.print(Padding(body, (0, 1), style="on #303030", expand=True))

    def _footer_state(self) -> FooterState:
        """（后台线程）底部状态栏要显示的数据：读会话状态比较慢，只在每次处理完输入后算一次。"""
        try:
            last = self.cli.last_call_usage()
            usage = self.cli.context_usage(last)
        except Exception:
            logger.exception("计算上下文占用失败")
            last = usage = None
        return self.cli.rt.approval.enabled, usage, last
