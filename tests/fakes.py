"""测试辅助：按脚本回复的假模型，以及构造工具调用、运行配置的小工具。"""
from __future__ import annotations

from typing import Any
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """按脚本依次返回回复的假模型，并把每次收到的消息列表记在 calls 里。

    script 中的每一项对应一次调用：
    - str / AIMessage：作为回复（每次返回副本，同一个对象可以重复使用）；
    - 异常实例：直接抛出；
    - 函数：以收到的消息列表为参数，返回以上两种之一。
    script 用完后改用 responder（同样是函数）；都没有时报错。
    """

    script: list[Any] = Field(default_factory=list)
    responder: Callable[[list[BaseMessage]], Any] | None = None
    calls: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        messages = list(messages)
        self.calls.append(messages)
        if self.script:
            item = self.script.pop(0)
        elif self.responder is not None:
            item = self.responder
        else:
            raise AssertionError(f"假模型的脚本已用完（第 {len(self.calls)} 次调用）")
        if callable(item) and not isinstance(item, BaseMessage):
            item = item(messages)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            item = AIMessage(item)
        return ChatResult(generations=[ChatGeneration(message=item.model_copy(deep=True))])


def tool_call(name: str, args: dict | None = None, call_id: str = "call_1", text: str = "") -> AIMessage:
    """带一个工具调用的 AI 消息。"""
    return AIMessage(text, tool_calls=[{"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}])


def tool_calls(*calls: tuple[str, dict, str]) -> AIMessage:
    """带多个工具调用的 AI 消息：tool_calls(("名称", 参数, "id"), ...)。"""
    return AIMessage("", tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"} for n, a, i in calls])


def run_config(thread_id: str = "t1", **extra) -> dict:
    return {"configurable": {"thread_id": thread_id}, **extra}


def pending_interrupts(graph, config) -> list:
    """当前挂起的 interrupt（与 AgentRuntime.pending_interrupts 的取法一致）。"""
    st = graph.get_state(config)
    ints = getattr(st, "interrupts", None)
    if ints is None:
        ints = [i for t in st.tasks for i in getattr(t, "interrupts", ())]
    return list(ints)
