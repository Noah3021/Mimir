"""执行一轮对话：驱动 LangGraph 运行，把过程转换成事件交给界面。

界面只负责显示事件、回答危险操作确认，不直接接触 LangGraph。同一套执行逻辑可以配不同的界面：
现在的逐行输出、输入框固定在底部的界面，以及以后的无头模式。

中断：界面调用 `TurnRunner.cancel()`（可以从其他线程调用），执行方在收到下一段数据、
或确认返回后停止，本轮以 `TurnEnd(interrupted=True)` 结束。在主线程运行时按 Ctrl+C
产生的 KeyboardInterrupt 也按中断处理。
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command

from .graph import extract_chunk
from .usage import Usage

logger = logging.getLogger("mimir")

THINKING = "思考中…"
RUNNING_TOOLS = "执行工具…"
RECONNECTING = "重新连接中…"


# ====================== 事件 ======================
@dataclass
class Status:
    """Agent 当前在做什么，界面显示在等待动画后面。"""
    text: str


@dataclass
class ReasoningDelta:
    """一段思考过程。"""
    text: str


@dataclass
class TextDelta:
    """一段正文。"""
    text: str


@dataclass
class ToolCall:
    """模型发起的一次工具调用（LangChain 的 tool_call 字典：name / args / id）。"""
    call: dict


@dataclass
class ToolResult:
    """一次工具调用的结果。"""
    message: ToolMessage


@dataclass
class Retry:
    """模型请求连接中断，即将重试。界面应丢弃这次请求已显示的半截输出。"""
    attempt: int
    max: int
    delay: float
    error: str


@dataclass
class Compact:
    """上下文压缩。stage：start / done / failed。"""
    stage: str
    messages: int
    before: int = 0
    after: int = 0
    error: str = ""


@dataclass
class ProjectUpdated:
    """项目说明文件（MIMIR.md 等）在会话中途被修改或删除，本轮会把最新内容告诉模型。"""
    file: str
    deleted: bool


@dataclass
class TurnEnd:
    """本轮结束。interrupted：被用户中断；error：出错时的异常。"""
    usage: Usage | None
    elapsed: float
    interrupted: bool = False
    error: Exception | None = None


Event = Status | ReasoningDelta | TextDelta | ToolCall | ToolResult | Retry | Compact | ProjectUpdated | TurnEnd
Emit = Callable[[Event], None]
# 危险操作确认：收到 interrupt 的值（tool / args / id），返回 (决定, 是否本次运行始终允许该工具)
Approver = Callable[[dict], tuple[Any, bool]]


class TurnCancelled(Exception):
    """界面请求中断本轮。"""


# ====================== 执行 ======================
class TurnRunner:
    def __init__(self, runtime: Any, emit: Emit, ask: Approver, *, streaming: bool = True):
        self.rt = runtime
        self.emit = emit
        self.ask = ask
        self.streaming = streaming
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise TurnCancelled

    def run(self, user_input: str) -> TurnEnd:
        rt = self.rt
        config = rt.run_config()
        started = time.perf_counter()
        rt.begin_turn()
        try:
            rt.repair_dangling_tool_calls()
        except Exception:
            logger.exception("修复中断的工具调用失败")

        payload: Any = {"messages": [self._user_message(user_input)]}
        always_allow: set[str] = set()
        interrupted, error = False, None
        try:
            while True:
                self._check_cancel()
                self._stream(payload, config)
                interrupts = rt.pending_interrupts()
                if not interrupts:
                    break
                value = interrupts[0].value or {}
                decision, always = self.ask(value)
                self._check_cancel()  # 确认期间被中断：不再继续执行
                if always:
                    always_allow.add(value.get("tool", ""))
                payload = Command(resume=decision)
        except (KeyboardInterrupt, TurnCancelled):
            interrupted = True
        except Exception as e:
            from langgraph.errors import GraphRecursionError
            if not isinstance(e, GraphRecursionError):
                logger.exception("对话出错")
            error = e
        finally:
            for name in always_allow:  # 一轮结束后再更新策略，避免中断恢复时重放不一致
                rt.approval.allow(name)
            try:
                rt.record_turn(user_input)
            except Exception:
                logger.exception("保存会话信息失败")
            rt.sync_transcript()  # 兜底：中断/出错时也把已产生的消息写入明文记录

        usage = None
        try:
            usage = rt.finish_turn()
        except Exception:
            logger.exception("统计用量失败")
        end = TurnEnd(usage=usage, elapsed=time.perf_counter() - started, interrupted=interrupted, error=error)
        self.emit(end)
        return end

    def _user_message(self, user_input: str) -> HumanMessage:
        # 项目说明（MIMIR.md）在会话中途变化时，把最新内容记在本轮用户消息上，发送时附带提醒
        from .project_context import PROJECT_UPDATE_KEY
        update = None
        try:
            update = self.rt.check_project_update()
        except Exception:
            logger.exception("检查项目说明更新失败")
        if not update:
            return HumanMessage(user_input)
        self.emit(ProjectUpdated(file=update["file"], deleted=not update["text"]))
        return HumanMessage(user_input, additional_kwargs={PROJECT_UPDATE_KEY: update})

    def _stream(self, payload: Any, config: dict) -> None:
        modes = ["messages", "updates", "custom"] if self.streaming else ["updates", "custom"]
        self.emit(Status(THINKING))
        stream = self.rt.graph.stream(payload, config, stream_mode=modes)
        try:
            for mode, data in stream:
                self._check_cancel()
                if mode == "messages":
                    chunk, meta = data
                    if meta.get("langgraph_node") == "agent" and isinstance(chunk, (AIMessageChunk, AIMessage)):
                        self._emit_content(chunk)
                elif mode == "custom" and isinstance(data, dict):
                    self._on_custom(data)
                elif mode == "updates" and isinstance(data, dict):
                    self._on_update(data)
                    self.rt.sync_transcript()  # 每个节点完成后立即追加，时间戳更准确
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    def _emit_content(self, m: AIMessage | AIMessageChunk) -> None:
        reasoning, text = extract_chunk(m)
        if reasoning:
            self.emit(ReasoningDelta(reasoning))
        if text:
            self.emit(TextDelta(text))

    def _on_custom(self, data: dict) -> None:
        kind = data.get("type")
        if kind == "retry":
            self.emit(Retry(attempt=data.get("attempt", 0), max=data.get("max", 0),
                            delay=data.get("delay", 0), error=str(data.get("error", ""))))
            self.emit(Status(RECONNECTING))
        elif kind == "compact":
            n = data.get("messages", 0)
            stage = data.get("stage", "")
            self.emit(Compact(stage=stage, messages=n, before=data.get("before", 0),
                              after=data.get("after", 0), error=str(data.get("error", ""))))
            self.emit(Status(f"上下文接近上限，正在摘要较早的 {n} 条消息…" if stage == "start" else THINKING))

    def _on_update(self, data: dict) -> None:
        for node, value in data.items():
            if not isinstance(value, dict):
                continue
            msgs = value.get("messages") or []
            if node == "agent":
                for m in msgs:
                    if not isinstance(m, AIMessage):
                        continue
                    if not self.streaming:  # 非流式：整条消息在这里才拿到
                        self._emit_content(m)
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            self.emit(ToolCall(tc))
                        self.emit(Status(RUNNING_TOOLS))
            elif node == "tools":
                for m in msgs:
                    if isinstance(m, ToolMessage):
                        self.emit(ToolResult(m))
                self.emit(Status(THINKING))
