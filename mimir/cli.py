"""富文本命令行界面：流式输出、Markdown 渲染、工具调用展示、危险操作确认、斜杠命令。"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from rich.console import Console, Group
from rich.live import Live
from rich.segment import Segment
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .agent import AgentRuntime
from .config import REASONING_LEVELS, ConfigError, Settings, load_settings
from .graph import extract_chunk, message_text
from .usage import Usage, fmt_tokens, format_usage_line

logger = logging.getLogger("mimir")

COMMANDS = [
    ("/help", "显示帮助"),
    ("/new", "开始新会话"),
    ("/sessions", "列出历史会话"),
    ("/resume <ID|序号>", "恢复历史会话（序号来自 /sessions）"),
    ("/delete <ID|序号>", "删除历史会话"),
    ("/history [条数]", "查看当前会话的消息记录（如 /history 20 只看最近 20 条，省略则显示全部）"),
    ("/model [名称]", "查看或切换模型"),
    ("/think [级别]", "查看或设置推理强度（off/low/medium/high/xhigh）"),
    ("/usage", "查看 token 用量（当前会话 / 今天 / 全部）"),
    ("/tools", "查看可用工具"),
    ("/approve on|off", "开启/关闭危险操作确认"),
    ("/config", "查看当前配置"),
    ("/clear", "清屏"),
    ("/exit", "退出（也可输入“退出”或按 Ctrl+D）"),
]
EXIT_WORDS = {"/exit", "/quit", "/q", "退出", "exit", "quit"}


# ====================== 输出渲染 ======================
def short(obj: Any, limit: int = 160) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "…"


def clip_lines(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n…（共 {len(lines)} 行，仅显示前 {max_lines} 行）"


class _Tail:
    """只渲染内容的最后若干行，保证 Live 区域始终不超过终端高度。

    Live 刷新时需要把光标移回上一帧顶部重绘；内容一旦高于终端，顶部已滚出屏幕、
    无法擦除，旧帧就会残留在滚动区里，看起来像重复输出。
    """

    def __init__(self, renderable: Any):
        self.renderable = renderable

    def __rich_console__(self, console: Console, options):
        max_lines = max(console.size.height - 2, 3)
        lines = console.render_lines(self.renderable, options, pad=False)
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        for line in lines:
            yield from line
            yield Segment.line()


class StreamPrinter:
    """管理流式输出：等待动画 / 推理过程 / 正文（Markdown 实时渲染）。

    流式过程中只在 Live 区域显示末尾一屏；结束时清除 Live 区域，再完整打印一次，
    终端中最终只保留一份完整内容。
    """

    def __init__(self, console: Console, show_reasoning: bool = True):
        self.console = console
        self.show_reasoning = show_reasoning
        self.live: Live | None = None
        self.mode: str | None = None
        self.buf = ""

    def wait(self, msg: str = "思考中…") -> None:
        self.end()
        self.mode = "wait"
        self.live = Live(Spinner("dots", text=Text(msg, style="dim")), console=self.console,
                         transient=True, refresh_per_second=12)
        self.live.start()

    def feed(self, kind: str, s: str) -> None:
        if not s or (kind == "reasoning" and not self.show_reasoning):
            return
        if self.mode != kind:
            self.end()
            self.mode, self.buf = kind, ""
            self.live = Live(_Tail(self._render()), console=self.console, refresh_per_second=10,
                             transient=True, vertical_overflow="crop")
            self.live.start()
        self.buf += s
        self.live.update(_Tail(self._render()))

    def _render(self):
        if self.mode == "reasoning":  # 标题放在 Live 内，重试丢弃时一并清除
            return Group(Text("思考过程", style="dim bold"), Text(self.buf.strip(), style="dim italic"))
        return Markdown(self.buf.strip())

    def discard(self) -> None:
        """丢弃当前流式区域的内容（不打印），用于请求中断后重试。"""
        if self.live is not None:
            try:
                self.live.stop()  # transient=True：清除已显示的半截内容
            finally:
                self.live = None
        self.mode, self.buf = None, ""

    def end(self) -> None:
        if self.live is not None:
            try:
                self.live.stop()  # transient=True：清除流式区域
            finally:
                self.live = None
            if self.mode in ("text", "reasoning") and self.buf.strip():
                self.console.print(self._render())  # 最终只完整打印一次
        self.mode, self.buf = None, ""


# ====================== CLI 主体 ======================
class AgentCLI:
    def __init__(self, runtime: AgentRuntime, console: Console):
        self.rt = runtime
        self.console = console
        self.settings: Settings = runtime.settings
        self.printer = StreamPrinter(console, self.settings.cli.show_reasoning)
        self.listing: list = []  # 最近一次 /sessions 的结果，用于按序号恢复
        self.session = self._make_prompt_session()

    # ---------- 输入 ----------
    def _make_prompt_session(self):
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.key_binding import KeyBindings

            words = [c.split()[0] for c, _ in COMMANDS] + ["/quit", "/approve on", "/approve off"]
            words += [f"/model {n}" for n in self.settings.models]
            words += [f"/think {lv}" for lv in REASONING_LEVELS]
            kb = KeyBindings()

            @kb.add("escape", "enter")  # Esc+Enter（或 Alt+Enter）换行
            def _(event):
                event.current_buffer.insert_text("\n")

            hist = self.settings.cli.history_file
            history = None
            if hist:
                hist.parent.mkdir(parents=True, exist_ok=True)
                history = FileHistory(str(hist))
            return PromptSession(
                history=history,
                completer=WordCompleter(words, ignore_case=True, sentence=True),
                key_bindings=kb,
                complete_while_typing=True,
            )
        except Exception as e:  # 例如 Git Bash(mintty) 中没有 Windows 控制台
            logger.warning("prompt_toolkit 不可用，改用简单输入：%s", e)
            return None

    def read_input(self) -> str:
        if self.session is not None:
            from prompt_toolkit.formatted_text import HTML
            try:
                return self.session.prompt(HTML("<ansigreen><b>你 › </b></ansigreen>"))
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception as e:
                logger.warning("prompt_toolkit 输入失败，改用简单输入：%s", e)
                self.session = None
        return self.console.input("[bold green]你 › [/]")

    # ---------- 主循环 ----------
    def banner(self) -> None:
        mc = self.rt.model_config
        info = Text.assemble(
            (f"{self.settings.agent.name}", "bold cyan"), f"  v{__version__}\n",
            ("模型    ", "dim"), f"{mc.name} ({mc.model})\n",
            ("推理    ", "dim"), f"{self.rt.reasoning_level}\n",
            ("工作目录 ", "dim"), f"{self.settings.tools.workspace}\n",
            ("会话    ", "dim"), f"{self.rt.thread_id}\n",
            ("输入 /help 查看命令，Esc+Enter 换行，Ctrl+C 中断回答", "dim"),
        )
        self.console.print(Panel(info, border_style="cyan", expand=False))

    def loop(self) -> None:
        self.banner()
        while True:
            try:
                line = self.read_input().strip()
            except KeyboardInterrupt:
                self.console.print("[dim]（输入 /exit 退出）[/]")
                continue
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in EXIT_WORDS:
                break
            if line.startswith("/"):
                self.handle_command(line)
            else:
                self.run_turn(line)
        self.console.print("[dim]再见！[/]")

    # ---------- 一轮对话 ----------
    def run_turn(self, user_input: str) -> None:
        config = self.rt.run_config()
        started = time.perf_counter()
        self.rt.begin_turn()
        try:
            self.rt.repair_dangling_tool_calls()
        except Exception:
            logger.exception("修复中断的工具调用失败")

        payload: Any = {"messages": [HumanMessage(user_input)]}
        always_allow: set[str] = set()
        self.console.print(Text(f"{self.settings.agent.name} ›", style="bold cyan"))
        try:
            while True:
                self._stream(payload, config)
                interrupts = self.rt.pending_interrupts()
                if not interrupts:
                    break
                value = interrupts[0].value or {}
                decision, always = self._ask_approval(value)
                if always:
                    always_allow.add(value.get("tool", ""))
                payload = Command(resume=decision)
        except KeyboardInterrupt:
            self.printer.end()
            self.console.print("[yellow]已中断本轮回答[/]")
        except GraphRecursionError:
            self.printer.end()
            self.console.print(f"[yellow]已达到单轮最大步数（{self.settings.agent.recursion_limit}），已停止。"
                               "可在 config.yaml 调大 agent.recursion_limit[/]")
        except Exception as e:
            self.printer.end()
            logger.exception("对话出错")
            self.console.print(f"[red]出错：{type(e).__name__}: {escape(str(e))}[/]")
            from .graph import is_transient_error
            if is_transient_error(e):
                self.console.print(f"[yellow]连接多次中断（已自动重试 {self.settings.agent.stream_retries} 次），"
                                   "请检查网络或代理后重新发送；可在 config.yaml 调整 agent.stream_retries[/]")
        finally:
            self.printer.end()
            for name in always_allow:  # 一轮结束后再更新策略，避免中断恢复时重放不一致
                self.rt.approval.allow(name)
            try:
                self.rt.record_turn(user_input)
            except Exception:
                logger.exception("保存会话信息失败")
            self.rt.sync_transcript()  # 兜底：中断/出错时也把已产生的消息写入明文记录
            self._print_turn_usage(time.perf_counter() - started)
        self.console.print()

    def _print_turn_usage(self, elapsed: float) -> None:
        try:
            u = self.rt.finish_turn()
        except Exception:
            logger.exception("统计用量失败")
            return
        if self.settings.cli.show_usage and u.calls:
            self.console.print(Text(format_usage_line(u, elapsed), style="dim"))

    def _stream(self, payload: Any, config: dict) -> None:
        streaming = self.settings.cli.stream
        modes = ["messages", "updates", "custom"] if streaming else ["updates", "custom"]
        self.printer.wait()
        for mode, data in self.rt.graph.stream(payload, config, stream_mode=modes):
            if mode == "messages":
                chunk, meta = data
                if meta.get("langgraph_node") != "agent" or not isinstance(chunk, (AIMessageChunk, AIMessage)):
                    continue
                reasoning, text = extract_chunk(chunk)
                self.printer.feed("reasoning", reasoning)
                self.printer.feed("text", text)
            elif mode == "custom" and isinstance(data, dict) and data.get("type") == "retry":
                self._handle_retry(data)
            elif mode == "updates" and isinstance(data, dict):
                self._handle_update(data, streaming)
                self.rt.sync_transcript()  # 每个节点完成后立即追加，时间戳更准确
        self.printer.end()

    def _handle_retry(self, data: dict) -> None:
        """模型请求连接中断：清掉半截输出，提示并等待重试。"""
        self.printer.discard()
        attempt, total, delay = data.get("attempt"), data.get("max"), data.get("delay")
        self.console.print(Text.assemble(
            ("  ⟳ ", "yellow"), (f"连接中断，{delay}s 后重试（{attempt}/{total}）", "yellow"),
            " ", (short(str(data.get("error", "")), 100), "dim")))
        self.printer.wait("重新连接中…")

    def _handle_update(self, data: dict, streaming: bool) -> None:
        for node, value in data.items():
            if not isinstance(value, dict):
                continue
            msgs = value.get("messages") or []
            if node == "agent":
                for m in msgs:
                    if not isinstance(m, AIMessage):
                        continue
                    if not streaming:
                        reasoning, text = extract_chunk(m)
                        self.printer.feed("reasoning", reasoning)
                        self.printer.feed("text", text)
                    if m.tool_calls:
                        self.printer.end()
                        if self.settings.cli.show_tool_calls:
                            for tc in m.tool_calls:
                                self.console.print(Text.assemble(
                                    ("  ▶ ", "magenta"), (tc["name"], "bold magenta"), " ", (short(tc["args"]), "dim")))
                        self.printer.wait("执行工具…")
            elif node == "tools":
                self.printer.end()
                if self.settings.cli.show_tool_calls:
                    for m in msgs:
                        if isinstance(m, ToolMessage):
                            ok = m.status != "error"
                            self.console.print(Text.assemble(
                                ("  ✓ " if ok else "  ✗ ", "green" if ok else "red"),
                                (short(message_text(m) or str(m.content), 200), "dim")))
                self.printer.wait()

    # ---------- 危险操作确认 ----------
    def _format_args(self, tool: str, args: dict):
        if tool == "run_shell":
            return Group(Text("命令：", style="bold"), Syntax(str(args.get("command", "")), "bash", word_wrap=True))
        if tool == "write_file":
            path = str(args.get("path", ""))
            mode = "追加到" if args.get("append") else "写入（覆盖）"
            content = clip_lines(str(args.get("content", "")))
            try:
                lexer = Syntax.guess_lexer(path, content)
            except Exception:
                lexer = "text"
            return Group(Text(f"{mode}：{path}", style="bold"), Syntax(content, lexer, line_numbers=True, word_wrap=True))
        if tool == "edit_file":
            old = clip_lines(str(args.get("old_text", "")), 20)
            new = clip_lines(str(args.get("new_text", "")), 20)
            diff = Text()
            for line in old.splitlines():
                diff.append(f"- {line}\n", style="red")
            for line in new.splitlines():
                diff.append(f"+ {line}\n", style="green")
            title = f"修改：{args.get('path', '')}" + ("（全部替换）" if args.get("replace_all") else "")
            return Group(Text(title, style="bold"), diff)
        return Text(clip_lines(json.dumps(args, ensure_ascii=False, indent=2, default=str)))

    def _ask_approval(self, value: dict) -> tuple[dict, bool]:
        self.printer.end()
        tool = value.get("tool", "?")
        self.console.print(Panel(self._format_args(tool, value.get("args") or {}),
                                 title=f"[bold yellow]需要确认：{escape(tool)}[/]", border_style="yellow"))
        hint = "[yellow]是否允许？[/] y=允许  n=拒绝  a=本次运行始终允许该工具  其他内容=拒绝并反馈给助手 › "
        while True:
            ans = self.console.input(hint).strip()
            low = ans.lower()
            if not ans:
                continue
            if low in ("y", "yes", "是", "允许"):
                return {"approved": True}, False
            if low in ("a", "always"):
                return {"approved": True}, True
            if low in ("n", "no", "否", "拒绝"):
                return {"approved": False}, False
            return {"approved": False, "feedback": ans}, False

    # ---------- 斜杠命令 ----------
    def handle_command(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        handlers = {
            "/help": self.cmd_help, "/h": self.cmd_help, "/?": self.cmd_help,
            "/new": self.cmd_new,
            "/sessions": self.cmd_sessions, "/ls": self.cmd_sessions,
            "/resume": self.cmd_resume,
            "/delete": self.cmd_delete,
            "/history": self.cmd_history,
            "/model": self.cmd_model, "/models": self.cmd_model,
            "/think": self.cmd_think, "/reasoning": self.cmd_think,
            "/usage": self.cmd_usage,
            "/tools": self.cmd_tools,
            "/approve": self.cmd_approve,
            "/config": self.cmd_config,
            "/clear": self.cmd_clear,
        }
        handler = handlers.get(cmd)
        if handler is None:
            self.console.print(f"[red]未知命令 {escape(cmd)}[/]，输入 /help 查看可用命令")
            return
        try:
            handler(arg)
        except Exception as e:
            logger.exception("命令执行失败：%s", line)
            self.console.print(f"[red]命令执行失败：{escape(str(e))}[/]")

    def cmd_help(self, _: str) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        for c, desc in COMMANDS:
            table.add_row(Text(c, style="cyan"), desc)
        self.console.print(Panel(table, title="命令", border_style="cyan", expand=False))

    def cmd_new(self, _: str) -> None:
        sid = self.rt.new_session()
        self.console.print(f"[green]已开始新会话[/] {sid}")

    def cmd_sessions(self, _: str) -> None:
        self.listing = self.rt.store.list(20)
        if not self.listing:
            self.console.print("[dim]暂无历史会话[/]")
            return
        table = Table(title="历史会话（最近 20 个）", title_justify="left")
        for col in ("#", "ID", "标题", "模型", "最后更新"):
            table.add_column(col)
        for i, s in enumerate(self.listing, 1):
            mark = " *" if s.thread_id == self.rt.thread_id else ""
            table.add_row(str(i), s.thread_id + mark, s.title, s.model, s.updated_at)
        self.console.print(table)
        self.console.print("[dim]用 /resume <序号或ID> 恢复会话（* 为当前会话）[/]")

    def _resolve_session(self, arg: str):
        if not arg:
            self.console.print("[yellow]请提供会话 ID 或序号（先用 /sessions 查看）[/]")
            return None
        if arg.isdigit() and self.listing and 1 <= int(arg) <= len(self.listing):
            return self.listing[int(arg) - 1]
        matches = self.rt.store.find(arg)
        if not matches:
            self.console.print(f"[red]找不到会话：{escape(arg)}[/]")
            return None
        if len(matches) > 1:
            self.console.print(f"[yellow]有 {len(matches)} 个会话匹配，请输入更完整的 ID[/]")
            return None
        return matches[0]

    def cmd_resume(self, arg: str) -> None:
        s = self._resolve_session(arg)
        if s is None:
            return
        self.rt.thread_id = s.thread_id
        self.console.print(f"[green]已恢复会话[/] {s.thread_id}：{escape(s.title)}")
        self.show_history(limit=10)

    def cmd_delete(self, arg: str) -> None:
        s = self._resolve_session(arg)
        if s is None:
            return
        ans = self.console.input(f"确认删除会话 {s.thread_id}（{escape(s.title)}）？[y/N] ").strip().lower()
        if ans not in ("y", "yes", "是"):
            return
        self.rt.store.delete(s.thread_id)
        try:
            self.rt.transcript.delete(s.thread_id)
        except OSError:
            logger.exception("删除对话记录文件失败")
        self.listing = [x for x in self.listing if x.thread_id != s.thread_id]
        self.console.print("[green]已删除[/]")
        if s.thread_id == self.rt.thread_id:
            self.cmd_new("")

    def cmd_history(self, arg: str) -> None:
        if not arg or arg.lower() in ("all", "全部"):
            self.show_history()
            return
        try:
            limit = int(arg)
        except ValueError:
            limit = 0
        if limit <= 0:
            self.console.print("[red]条数必须是正整数[/]，例如 /history 20；省略则显示全部")
            return
        self.show_history(limit=limit)

    def show_history(self, limit: int | None = None) -> None:
        msgs = self.rt.messages()
        if not msgs:
            self.console.print("[dim]当前会话暂无消息[/]")
            return
        if limit and len(msgs) > limit:
            self.console.print(f"[dim]……（省略前 {len(msgs) - limit} 条，/history 查看全部）[/]")
            msgs = msgs[-limit:]
        self.console.rule("[dim]会话记录[/]")
        for m in msgs:
            if isinstance(m, HumanMessage):
                self.console.print(Text.assemble(("你 › ", "bold green"), message_text(m)))
            elif isinstance(m, AIMessage):
                _, text = extract_chunk(m)
                if text.strip():
                    self.console.print(Text(f"{self.settings.agent.name} ›", style="bold cyan"))
                    self.console.print(Markdown(text.strip()))
                for tc in m.tool_calls:
                    self.console.print(Text.assemble(("  ▶ ", "magenta"), (tc["name"], "magenta"), " ",
                                                     (short(tc["args"]), "dim")))
            elif isinstance(m, ToolMessage):
                ok = m.status != "error"
                self.console.print(Text.assemble(("  ✓ " if ok else "  ✗ ", "green" if ok else "red"),
                                                 (short(message_text(m) or str(m.content), 120), "dim")))
        self.console.rule()

    def cmd_model(self, arg: str) -> None:
        if not arg:
            table = Table(title="模型", title_justify="left")
            for col in ("", "名称", "类型", "模型 ID", "推理", "说明", "状态"):
                table.add_column(col)
            for name, mc in self.settings.models.items():
                cur = "*" if name == self.rt.model_name else ""
                status = "[green]可用[/]" if mc.available else f"[red]缺少 {', '.join(mc.missing_fields())}[/]"
                table.add_row(cur, name, mc.provider, escape(mc.model or "-"), self.rt.reasoning_for(name),
                              escape(mc.description), status)
            self.console.print(table)
            self.console.print("[dim]用 /model <名称> 切换[/]")
            return
        if arg == self.rt.model_name:
            self.console.print(f"[dim]当前已是 {escape(arg)}[/]")
            return
        try:
            self.rt.switch_model(arg)
        except ConfigError as e:
            self.console.print(f"[red]{escape(str(e))}[/]")
            return
        mc = self.rt.model_config
        self.console.print(f"[green]已切换到[/] {mc.name}（{escape(mc.model)}，推理 {self.rt.reasoning_level}），"
                           "当前会话上下文保留")

    def cmd_think(self, arg: str) -> None:
        if not arg:
            self.console.print(f"模型 {self.rt.model_name} 当前推理强度：[cyan]{self.rt.reasoning_level}[/]"
                               f"（可选：{' / '.join(REASONING_LEVELS)}）")
            if self.rt.model_config.provider == "deepseek":
                self.console.print("[dim]DeepSeek 只区分开/关：off 为关闭，其他级别均为开启[/]")
            return
        try:
            self.rt.set_reasoning(arg)
        except ConfigError as e:
            self.console.print(f"[red]{escape(str(e))}[/]")
            return
        self.console.print(f"[green]{self.rt.model_name} 的推理强度已设为[/] {self.rt.reasoning_level}"
                           "[dim]（仅本次运行有效，默认值在 config.yaml 中修改）[/]")

    @staticmethod
    def _usage_cells(u: Usage) -> list[str]:
        cache = f"{fmt_tokens(u.cache_read)}（{u.cache_rate:.0%}）" if u.input else fmt_tokens(u.cache_read)
        return [str(u.calls), fmt_tokens(u.input), cache, fmt_tokens(u.output),
                fmt_tokens(u.reasoning), fmt_tokens(u.total)]

    def cmd_usage(self, _: str) -> None:
        scopes = [
            ("当前会话", self.rt.usage.by_model(thread_id=self.rt.thread_id)),
            ("今天", self.rt.usage.by_model(day=date.today().isoformat())),
            ("全部", self.rt.usage.by_model()),
        ]
        table = Table(title="Token 用量", title_justify="left")
        table.add_column("范围")
        table.add_column("模型")
        for col in ("调用", "输入", "缓存命中", "输出", "推理", "合计"):
            table.add_column(col, justify="right")
        for scope, rows in scopes:
            if not rows:
                table.add_row(scope, "[dim]暂无[/]", *[""] * 6)
            else:
                total = Usage()
                for i, r in enumerate(rows):
                    total.add(r.usage)
                    label = Text.assemble((r.profile, "cyan"), " ", (r.model, "dim"))
                    table.add_row(scope if i == 0 else "", label, *self._usage_cells(r.usage))
                if len(rows) > 1:
                    table.add_row("", Text("小计", style="bold"), *self._usage_cells(total))
            table.add_section()
        self.console.print(table)
        self.console.print("[dim]输入包含缓存命中部分；推理 token 计入输出。[/]")

    def cmd_tools(self, _: str) -> None:
        table = Table(title="可用工具", title_justify="left")
        table.add_column("名称", style="cyan")
        table.add_column("需确认")
        table.add_column("说明")
        for t in self.rt.tools:
            need = "是" if t.name in self.rt.approval.required else ""
            if t.name in self.rt.approval.session_allowed:
                need = "已始终允许"
            table.add_row(t.name, need, (t.description or "").strip().splitlines()[0])
        self.console.print(table)
        if not self.rt.approval.enabled:
            self.console.print("[yellow]注意：危险操作确认已关闭[/]")

    def cmd_approve(self, arg: str) -> None:
        a = arg.lower()
        if a in ("on", "开"):
            self.rt.approval.enabled = True
            self.rt.approval.session_allowed.clear()
            self.console.print("[green]已开启危险操作确认[/]")
        elif a in ("off", "关"):
            self.rt.approval.enabled = False
            self.console.print("[yellow]已关闭危险操作确认，写文件和执行命令将不再询问，请谨慎！[/]")
        else:
            state = "开启" if self.rt.approval.enabled else "关闭"
            self.console.print(f"危险操作确认当前为：{state}（用法：/approve on|off）")

    def cmd_config(self, _: str) -> None:
        s = self.settings
        table = Table(show_header=False, box=None, padding=(0, 2))
        rows = [
            ("配置文件", str(s.config_path)),
            ("智能体名称", s.agent.name),
            ("当前模型", f"{self.rt.model_name}（{self.rt.model_config.provider} / {self.rt.model_config.model}）"),
            ("推理强度", self.rt.reasoning_level),
            ("当前会话", self.rt.thread_id),
            ("工作目录", str(s.tools.workspace)),
            ("会话数据库", str(s.memory.db_path)),
            ("对话记录", str(self.rt.transcript.path(self.rt.thread_id)) if s.memory.transcripts_enabled else "已关闭"),
            ("日志文件", str(s.logging.file or "-")),
            ("单轮最大步数", str(s.agent.recursion_limit)),
            ("历史消息上限", str(s.agent.max_history_messages)),
            ("需确认的工具", ", ".join(sorted(self.rt.approval.required)) or "-"),
            ("禁用的工具", ", ".join(s.tools.disabled) or "-"),
            ("流式输出", "是" if s.cli.stream else "否"),
        ]
        for k, v in rows:
            table.add_row(Text(k, style="dim"), v)
        self.console.print(Panel(table, title="当前配置", border_style="cyan", expand=False))

    def cmd_clear(self, _: str) -> None:
        self.console.clear()


# ====================== 入口 ======================
def setup_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.logging.level, logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    if settings.logging.file:
        settings.logging.file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(settings.logging.file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    else:
        root.addHandler(logging.NullHandler())
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mimir", description="Mimir —— 基于 LangChain + LangGraph 的个人 CLI 智能体")
    p.add_argument("-c", "--config", help="配置文件路径（默认为项目目录下的 config.yaml）")
    p.add_argument("-m", "--model", help="使用的模型名称（config.yaml 中 models 的键名）")
    p.add_argument("-t", "--think", choices=REASONING_LEVELS, help="本次运行的推理强度（覆盖 config.yaml）")
    p.add_argument("-w", "--workspace", help="文件/命令工具的工作目录（默认为当前目录）")
    g = p.add_mutually_exclusive_group()
    g.add_argument("-r", "--resume", metavar="ID", help="恢复指定会话（ID 或 ID 前缀）")
    g.add_argument("--continue", dest="cont", action="store_true", help="继续最近一次会话")
    p.add_argument("-p", "--prompt", help="单次提问，回答后退出")
    p.add_argument("--yolo", action="store_true", help="跳过所有危险操作确认（慎用）")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def _create_runtime(settings: Settings, args: argparse.Namespace, console: Console) -> AgentRuntime:
    default = settings.agent.default_model
    candidates = [args.model] if args.model else [default, *[n for n in settings.models if n != default]]
    errors = []
    for name in candidates:
        try:
            rt = AgentRuntime(settings, name, approval_enabled=not args.yolo, reasoning=args.think)
        except ConfigError as e:
            errors.append(str(e))
            continue
        if errors:
            console.print(f"[yellow]{escape(errors[0])}，已改用模型 {name}[/]")
        return rt
    raise ConfigError("没有可用的模型：\n  " + "\n  ".join(errors) + "\n请参考 .env.example 配置环境变量")


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
    args = parse_args(argv)
    console = Console()
    try:
        settings = load_settings(args.config, workspace=args.workspace)
        setup_logging(settings)
        runtime = _create_runtime(settings, args, console)
    except ConfigError as e:
        console.print(f"[red]配置错误：[/]{escape(str(e))}")
        return 1

    cli = AgentCLI(runtime, console)
    try:
        if args.resume or args.cont:
            sessions = runtime.store.find(args.resume) if args.resume else runtime.store.list(1)
            if len(sessions) == 1:
                runtime.thread_id = sessions[0].thread_id
            else:
                console.print("[yellow]未找到唯一匹配的会话，已开始新会话[/]")
        if args.prompt:
            cli.run_turn(args.prompt)
        else:
            if args.resume or args.cont:
                cli.show_history(limit=10)
            cli.loop()
    finally:
        runtime.close()
    return 0
