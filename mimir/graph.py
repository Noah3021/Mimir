"""LangGraph 工作流：

    START -> [需要压缩] -> compact -> agent(LLM) -> [有工具调用] -> tools -> [需要压缩] -> compact -> agent ... -> END

tools 节点在执行需要确认的工具前调用 interrupt() 暂停，由 CLI 询问用户后用
Command(resume=...) 恢复。审批放在执行之前统一进行，保证恢复重放节点时不会重复产生副作用。

compact 节点只在历史超过高水位时进入：清理旧工具输出，仍超出时把最早的若干轮对话交给模型
生成摘要。压缩状态（summary / summary_cut / clear_before）写入图状态，随 Checkpointer 持久化；
agent 节点只按状态拼装请求，两次压缩之间发送内容的前缀保持不变，便于命中提示词缓存。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any
from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

try:
    from langgraph.config import get_stream_writer
except ImportError:  # 旧版 langgraph
    get_stream_writer = None

from .approval import ApprovalPolicy
from .config import ContextConfig
from .context import ContextPlan, estimate_message_tokens, estimate_text_tokens, history_budget, plan_context
from .project_context import ProjectContext, inject_context, inject_updates, latest_update
from .summary import build_summary_request, split_chunks, summary_block
from .tool_clearing import clear_boundary, clear_old_tool_outputs

logger = logging.getLogger(__name__)


class AgentState(MessagesState):
    summary: str        # 早期对话摘要（拼接在系统提示词后面）
    summary_cut: int    # messages[:summary_cut] 已由摘要代替，不再发送
    clear_before: int   # messages[:clear_before] 中的大段工具输出发送时替换为占位文字


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
    """获取 LangGraph 自定义流写入器，用于通知 CLI 重试 / 压缩；不可用时返回空函数。"""
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


def _tools_overhead(tools: Sequence[BaseTool]) -> int:
    """估算工具定义（JSON Schema）占用的 token。"""
    try:
        schemas = [convert_to_openai_tool(t) for t in tools]
    except Exception:  # 估算失败不影响对话
        schemas = [{"name": t.name, "description": t.description} for t in tools]
    return estimate_text_tokens(json.dumps(schemas, ensure_ascii=False, default=str))


def _adapt_for_profile(messages: Sequence[BaseMessage], profile: str) -> list[BaseMessage]:
    result = []
    for m in messages:
        # 切换模型后，把其他模型产生的消息简化为纯文本，避免不同接口格式（如推理块）不兼容
        if isinstance(m, AIMessage):
            origin = m.response_metadata.get("agent_profile")
            if origin and origin != profile:
                m = AIMessage(content=message_text(m), tool_calls=m.tool_calls, id=m.id)
        result.append(m)
    return result


def context_state(state: dict) -> tuple[str, int, int]:
    """读取压缩状态 (summary, summary_cut, clear_before)；旧会话没有这些字段时取默认值。"""
    return (state.get("summary") or "", int(state.get("summary_cut") or 0), int(state.get("clear_before") or 0))


def build_agent_graph(
    *,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: Callable[[], str],
    approval: ApprovalPolicy,
    project_context: Callable[[], ProjectContext] | None = None,
    checkpointer: BaseCheckpointSaver | None,
    context: ContextConfig | None = None,
    context_window: int | None = None,
    profile: str = "default",
    max_tool_output_chars: int = 8000,
    max_stream_retries: int = 5,
    on_usage: Callable[[AIMessage], None] | None = None,
):
    tools_by_name = {t.name: t for t in tools}
    model = llm.bind_tools(list(tools)) if tools else llm
    ctx = context or ContextConfig()
    window = context_window or ctx.window
    tools_tokens = _tools_overhead(tools)
    clearing = ctx.clear_keep_turns > 0
    # 新摘要的预估 token（摘要按 summary_chars 字控制长度，按最坏情况 1 字 1 token 计）
    summary_reserve = estimate_text_tokens(summary_block("中" * ctx.summary_chars))

    def project_text(msgs: Sequence[BaseMessage], cut: int) -> str:
        """拼在第一条用户消息前的项目上下文。已被摘要压缩掉的消息里如有项目说明更新，
        改用其中最新的版本（更新提醒随消息一起被压缩，最新内容要保留下来）；只在压缩点前移时变化。"""
        if project_context is None:
            return ""
        return project_context().with_update(latest_update(msgs[:cut])).render()

    def invoke_with_retry(runnable: Runnable, messages: list[BaseMessage], config: RunnableConfig) -> AIMessage:
        writer = _stream_writer()
        attempt = 0
        while True:
            try:
                return runnable.invoke(messages, config)
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

    def plan(state: AgentState) -> tuple[ContextPlan, int, int]:
        summary, cut, cb = context_state(state)
        msgs = inject_updates(state["messages"])  # 项目说明更新提醒计入对应用户消息的 token
        overhead = (estimate_text_tokens(system_prompt()) + tools_tokens
                    + estimate_text_tokens(project_text(state["messages"], cut)))
        high, low = history_budget(window, ctx.high, ctx.low, reserve=ctx.output_reserve, overhead=overhead)
        p = plan_context(
            msgs, high=high, low=low, cut=cut, clear_before=cb,
            summary_tokens=estimate_text_tokens(summary_block(summary)), summary_reserve=summary_reserve,
            cleared_cost=(lambda m: estimate_message_tokens(
                clear_old_tool_outputs([m], 1, min_chars=ctx.clear_min_chars)[0])) if clearing else None,
            clear_boundary=(lambda end: clear_boundary(msgs[:end], ctx.clear_keep_turns)) if clearing else None,
        )
        return p, high, low

    def route_in(state: AgentState) -> str:
        """进入 agent 前判断是否需要压缩（只在越过高水位时为真）。"""
        p, _, _ = plan(state)
        _, cut, cb = context_state(state)
        return "compact" if (p.cut, p.clear_before) != (cut, cb) else "agent"

    def summarize(previous: str, msgs: Sequence[BaseMessage], chunk_budget: int, config: RunnableConfig) -> str:
        summary = previous
        for chunk in split_chunks(msgs, chunk_budget):
            resp = invoke_with_retry(llm, build_summary_request(summary, chunk, ctx.summary_chars), config)
            resp.response_metadata["agent_profile"] = profile
            if on_usage is not None:
                on_usage(resp)
            text = message_text(resp).strip()
            if not text:
                raise RuntimeError("模型返回了空摘要")
            summary = text
        return summary

    def compact_node(state: AgentState, config: RunnableConfig):
        p, high, low = plan(state)
        summary, cut, cb = context_state(state)
        update: dict[str, Any] = {"clear_before": p.clear_before}
        if p.cut <= cut:
            logger.info("上下文压缩：清理第 %d 条之前的旧工具输出，约 %d → %d tokens（高水位 %d / 低水位 %d，窗口 %d）",
                        p.clear_before, p.before, p.tokens, high, low, window)
            return update

        writer = _stream_writer()
        n = p.cut - cut
        writer({"type": "compact", "stage": "start", "messages": n})
        chunk_budget = max(2000, low - summary_reserve)
        try:
            new_summary, ok, error = summarize(summary, state["messages"][cut:p.cut], chunk_budget, config), True, ""
        except GraphBubbleUp:
            raise
        except Exception as e:  # 摘要失败时退化为直接省略，避免请求超出上下文窗口
            logger.exception("生成对话摘要失败")
            note = f"（另有 {n} 条更早的消息因摘要生成失败而被省略。）"
            new_summary, ok, error = f"{summary}\n\n{note}".strip(), False, f"{type(e).__name__}: {e}"
        after = estimate_text_tokens(summary_block(new_summary)) + p.tokens - summary_reserve
        logger.info("上下文压缩：%s前 %d 条消息（第 %d~%d 条），清理第 %d 条之前的旧工具输出，约 %d → %d tokens"
                    "（高水位 %d / 低水位 %d，窗口 %d）", "摘要" if ok else "省略", p.cut, cut, p.cut - 1,
                    p.clear_before, p.before, after, high, low, window)
        writer({"type": "compact", "stage": "done" if ok else "failed", "messages": n,
                "before": p.before, "after": after, "error": error})
        update.update(summary=new_summary, summary_cut=p.cut)
        return update

    def agent_node(state: AgentState, config: RunnableConfig):
        summary, cut, cb = context_state(state)
        msgs = state["messages"]
        if cb:
            msgs = clear_old_tool_outputs(msgs, cb, min_chars=ctx.clear_min_chars)
        history = _adapt_for_profile(list(msgs)[cut:], profile)
        # 项目说明更新提醒拼在记录它的那条用户消息前面；项目上下文拼在（压缩后剩余的）第一条用户消息前面。
        # 两者都只影响发送内容，不改动状态里的消息
        history = inject_updates(history)
        history = inject_context(history, project_text(state["messages"], cut))
        messages = [SystemMessage(system_prompt() + summary_block(summary)), *history]
        response = invoke_with_retry(model, messages, config)
        response.response_metadata["agent_profile"] = profile
        return {"messages": [response]}

    def tools_node(state: AgentState):
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

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("compact", compact_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_conditional_edges(START, route_in, ["compact", "agent"])
    graph.add_edge("compact", "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_conditional_edges("tools", route_in, ["compact", "agent"])
    return graph.compile(checkpointer=checkpointer)
