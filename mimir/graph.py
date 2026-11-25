"""LangGraph 工作流：START -> agent(LLM) -> [有工具调用] -> tools -> agent ... -> END

tools 节点在执行需要确认的工具前调用 interrupt() 暂停，由 CLI 询问用户后用
Command(resume=...) 恢复。审批放在执行之前统一进行，保证恢复重放节点时不会重复产生副作用。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

try:
    from langgraph.config import get_stream_writer
except ImportError:  # 旧版 langgraph
    get_stream_writer = None

from .approval import ApprovalPolicy

logger = logging.getLogger(__name__)


def message_text(msg: BaseMessage) -> str:
    t = getattr(msg, "text", "")
    if isinstance(t, str):  # 新版 .text 是属性（str 子类）
        return str(t)
    if callable(t):  # 兼容旧版 langchain-core 的 .text()
        t = t()
    return t or ""


def extract_chunk(msg: BaseMessage) -> tuple[str, str]:
    """从（流式）AI 消息中提取 (推理内容, 正文)。兼容 Chat Completions / Responses API / reasoning_content。"""
    reasoning, text = "", ""
    content = msg.content
    if isinstance(content, str):
        text = content
    else:
        for block in content or []:
            if isinstance(block, str):
                text += block
            elif isinstance(block, dict):
                kind = block.get("type")
                if kind in ("text", "output_text"):
                    text += block.get("text") or ""
                elif kind == "reasoning":
                    if isinstance(block.get("reasoning"), str):
                        reasoning += block["reasoning"]
                    for s in block.get("summary") or []:
                        if isinstance(s, dict):
                            reasoning += s.get("text") or ""
                elif kind == "thinking":
                    reasoning += block.get("thinking") or ""
    rc = msg.additional_kwargs.get("reasoning_content") if hasattr(msg, "additional_kwargs") else None
    if isinstance(rc, str):
        reasoning += rc
    return reasoning, text


# 可重试的网络类错误（按类名匹配，兼容 openai / anthropic / httpx 及 LangChain 的包装类）
_TRANSIENT_ERROR_NAMES = {
    "APIConnectionError", "APITimeoutError", "OpenAIConnectionError",
    "RemoteProtocolError", "ReadError", "ConnectError", "ReadTimeout", "ConnectTimeout",
}


def is_transient_error(exc: BaseException) -> bool:
    """沿异常链判断是否为连接中断 / 超时等可重试错误。"""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if any(c.__name__ in _TRANSIENT_ERROR_NAMES for c in type(cur).__mro__):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _stream_writer() -> Callable[[Any], None]:
    """获取 LangGraph 自定义流写入器，用于通知 CLI 重试；不可用时返回空函数。"""
    if get_stream_writer is None:
        return lambda _: None
    try:
        return get_stream_writer()
    except Exception:
        return lambda _: None


def _to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(output)


def _normalize_decision(decision: Any) -> tuple[bool, str]:
    if isinstance(decision, dict):
        return bool(decision.get("approved")), str(decision.get("feedback") or "")
    if isinstance(decision, bool):
        return decision, ""
    if isinstance(decision, str):
        return decision.strip().lower() in {"y", "yes", "是"}, ""
    return False, ""


def _prepare_history(messages: Sequence[BaseMessage], max_messages: int, profile: str) -> list[BaseMessage]:
    msgs = list(messages)
    if max_messages and len(msgs) > max_messages:
        trimmed = trim_messages(
            msgs, max_tokens=max_messages, token_counter=len, strategy="last",
            start_on="human", include_system=False, allow_partial=False,
        )
        if not trimmed:  # 单轮工具调用过多时，至少保留最近一条用户消息起的内容
            last_human = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=0)
            trimmed = msgs[last_human:]
        msgs = list(trimmed)

    result = []
    for m in msgs:
        # 切换模型后，把其他模型产生的消息简化为纯文本，避免不同接口格式（如推理块）不兼容
        if isinstance(m, AIMessage):
            origin = m.response_metadata.get("agent_profile")
            if origin and origin != profile:
                m = AIMessage(content=message_text(m), tool_calls=m.tool_calls, id=m.id)
        result.append(m)
    return result


def build_agent_graph(
    *,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: Callable[[], str],
    approval: ApprovalPolicy,
    checkpointer: BaseCheckpointSaver | None,
    max_history_messages: int = 60,
    profile: str = "default",
    max_tool_output_chars: int = 8000,
    max_stream_retries: int = 5,
):
    tools_by_name = {t.name: t for t in tools}
    model = llm.bind_tools(list(tools)) if tools else llm

    def agent_node(state: MessagesState, config: RunnableConfig):
        history = _prepare_history(state["messages"], max_history_messages, profile)
        messages = [SystemMessage(system_prompt()), *history]
        writer = _stream_writer()
        attempt = 0
        while True:
            try:
                response = model.invoke(messages, config)
                break
            except GraphBubbleUp:
                raise
            except Exception as e:
                # SDK 自带的 max_retries 只在收到响应前重试；流式传输中途断开需要在这里重发整个请求。
                # 失败的那次没有产生任何消息，重发不会污染历史，也不影响提示词缓存。
                if attempt >= max_stream_retries or not is_transient_error(e):
                    raise
                attempt += 1
                delay = min(2 ** (attempt - 1), 8)  # 1, 2, 4, 8, 8 秒
                logger.warning("模型请求连接中断（%s: %s），%ss 后第 %d/%d 次重试",
                               type(e).__name__, e, delay, attempt, max_stream_retries)
                writer({"type": "retry", "attempt": attempt, "max": max_stream_retries,
                        "delay": delay, "error": f"{type(e).__name__}: {e}"})
                time.sleep(delay)
        response.response_metadata["agent_profile"] = profile
        return {"messages": [response]}

    def tools_node(state: MessagesState):
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []

        # 阶段 1：逐个审批（只有 interrupt，没有副作用，可安全重放）
        decisions: dict[str, tuple[bool, str]] = {}
        for tc in calls:
            if approval.needs(tc["name"]):
                decision = interrupt({"type": "approval", "tool": tc["name"], "args": tc["args"], "id": tc["id"]})
                decisions[tc["id"]] = _normalize_decision(decision)

        # 阶段 2：执行
        results: list[ToolMessage] = []
        for tc in calls:
            name, call_id = tc["name"], tc["id"]
            approved, feedback = decisions.get(call_id, (True, ""))
            if not approved:
                content = "用户拒绝执行此操作。"
                content += f"用户反馈：{feedback}" if feedback else "请询问用户下一步怎么做，不要重复尝试相同操作。"
                results.append(ToolMessage(content, tool_call_id=call_id, name=name, status="error"))
                continue
            tool = tools_by_name.get(name)
            if tool is None:
                results.append(ToolMessage(f"错误：不存在工具 {name}", tool_call_id=call_id, name=name, status="error"))
                continue
            try:
                output, status = _to_text(tool.invoke(tc["args"])), "success"
            except GraphBubbleUp:
                raise
            except Exception as e:  # 工具异常反馈给模型，而不是中断整个对话
                logger.exception("工具 %s 执行失败", name)
                output, status = f"工具执行出错：{type(e).__name__}: {e}", "error"
            if len(output) > max_tool_output_chars:
                output = output[:max_tool_output_chars] + "\n...（已截断）"
            results.append(ToolMessage(output, tool_call_id=call_id, name=name, status=status))
        return {"messages": results}

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
