"""上下文窗口管理：按估算 token 计量历史消息，超过高水位时压缩到低水位。

压缩状态（摘要覆盖到第几条消息 cut、旧工具输出清理边界 clear_before、摘要本身）保存在
图状态里，随 Checkpointer 持久化。每次调用只在已有状态的基础上判断是否需要再压缩一次：
未超过高水位时状态不变，发给模型的前缀逐字不变，可以持续命中提示词缓存。

压缩分两级：先清理旧工具输出；仍超过低水位，再把最早的若干轮对话（以用户消息为边界，
不会拆散工具调用与工具结果）交给模型生成摘要，替代原始消息。当前这一轮不压缩。
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

MESSAGE_OVERHEAD = 4  # 每条消息的角色、分隔符等固定开销（估算）


def estimate_text_tokens(text: str) -> int:
    """粗略估算 token 数：非 ASCII 字符（中文等）按 1 个，ASCII 约 3 个字符 1 个。偏保守。"""
    if not text:
        return 0
    ascii_n = len(text.encode("ascii", "ignore"))
    return (len(text) - ascii_n) + (ascii_n + 2) // 3


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def estimate_message_tokens(msg: BaseMessage) -> int:
    n = MESSAGE_OVERHEAD + estimate_text_tokens(_content_text(msg.content))
    if isinstance(msg, AIMessage) and msg.tool_calls:
        calls = [{"name": tc.get("name"), "args": tc.get("args")} for tc in msg.tool_calls]
        n += estimate_text_tokens(_content_text(calls))
    return n


@dataclass(frozen=True)
class ContextPlan:
    cut: int           # 发送 messages[cut:]；messages[:cut] 由摘要代替
    clear_before: int  # messages[:clear_before] 中的旧工具输出需要清理（0 = 不清理）
    tokens: int        # 发送部分的估算 token（含摘要；cut 前移时按 summary_reserve 预估）
    total_tokens: int  # 完整历史的估算 token（未清理、未压缩）
    before: int = 0    # 按传入的压缩状态，本次压缩前的估算 token（含现有摘要）


def plan_context(
    messages: Sequence[BaseMessage],
    *,
    high: int,
    low: int,
    cut: int = 0,
    clear_before: int = 0,
    summary_tokens: int = 0,
    summary_reserve: int = 0,
    cost: Callable[[BaseMessage], int] = estimate_message_tokens,
    cleared_cost: Callable[[BaseMessage], int] | None = None,
    clear_boundary: Callable[[int], int] | None = None,
) -> ContextPlan:
    """在已有压缩状态（cut / clear_before / 现有摘要 summary_tokens）基础上计算本次压缩。

    high / low 为历史可用的 token 预算（已扣除系统提示词、工具定义和输出预留）。
    未超过 high：原样返回。超过时：先把清理边界前移到 clear_boundary(len(messages))；
    仍超过 low，再按用户消息边界前移 cut，直到“新摘要（预估 summary_reserve）+ 剩余消息”
    不超过 low，或只剩当前这一轮。跨过的 messages[旧 cut:新 cut] 由调用方生成摘要。
    cleared_cost 为消息被清理后的 token 数；与 clear_boundary 都提供时才启用工具输出清理。
    """
    n = len(messages)
    cut = max(0, min(cut, n))
    full = [cost(m) for m in messages]
    clearing = cleared_cost is not None and clear_boundary is not None
    cleared = [cleared_cost(m) for m in messages] if clearing else full

    pf, pc = [0] * (n + 1), [0] * (n + 1)  # 前缀和：原始 / 清理后
    for i in range(n):
        pf[i + 1] = pf[i] + full[i]
        pc[i + 1] = pc[i] + cleared[i]

    def total(c: int, cb: int) -> int:
        """messages[c:] 的 token 数，其中下标 < cb 的按清理后计算。"""
        a = min(max(cb, c), n)
        return (pc[a] - pc[c]) + (pf[n] - pf[a])

    cb = clear_before
    tokens = before = summary_tokens + total(cut, cb)
    if tokens <= high:
        return ContextPlan(cut=cut, clear_before=cb, tokens=tokens, total_tokens=pf[n], before=before)

    if clearing:
        cb = max(cb, clear_boundary(n))
        tokens = summary_tokens + total(cut, cb)

    if tokens > low:
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        last_human = humans[-1] if humans else 0  # 当前这一轮（最后一条用户消息起）不能压缩
        new_cut = cut
        while True:
            j = bisect.bisect_right(humans, new_cut)
            if j >= len(humans) or humans[j] > last_human:
                break
            new_cut = humans[j]
            if summary_reserve + total(new_cut, cb) <= low:
                break
        if new_cut > cut:
            cut = new_cut
            tokens = summary_reserve + total(cut, cb)
    return ContextPlan(cut=cut, clear_before=cb, tokens=tokens, total_tokens=pf[n], before=before)


def history_budget(
    context_window: int, high_ratio: float, low_ratio: float, *, reserve: int, overhead: int
) -> tuple[int, int]:
    """把上下文窗口换算成历史消息的 (高水位, 低水位) token 预算。

    reserve 为模型输出预留，overhead 为系统提示词与工具定义占用。
    """
    high = min(int(context_window * high_ratio), context_window - reserve) - overhead
    low = int(context_window * low_ratio) - overhead
    high = max(high, 1000)
    low = max(min(low, high * 3 // 4), 500)
    return high, low
