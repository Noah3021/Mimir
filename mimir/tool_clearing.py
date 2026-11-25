"""上下文压缩：清理旧的工具输出。

把清理边界之前的大段 ToolMessage 替换为简短占位，保留 tool_call_id / name / status / id，
使 AIMessage.tool_calls 与 ToolMessage 仍一一配对。全部为纯函数，给定相同输入与边界输出恒定，
便于持久化边界后命中提示词缓存。
"""
from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

PLACEHOLDER_PREFIX = "[旧工具输出已清理："


def _content_len(content) -> int:
    """内容的文本长度：str 直接取长度；list 累加其中文本块的长度。"""
    if isinstance(content, str):
        return len(content)
    total = 0
    for block in content or []:
        if isinstance(block, str):
            total += len(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                total += len(text)
    return total


def is_cleared(msg: BaseMessage) -> bool:
    """是否已经是占位内容。"""
    return isinstance(msg.content, str) and msg.content.startswith(PLACEHOLDER_PREFIX)


def _placeholder(name: str | None, n: int) -> str:
    return f"{PLACEHOLDER_PREFIX}{name or '未知工具'}，原 {n} 字符。如仍需要，请重新调用工具获取]"


def clear_boundary(messages: Sequence[BaseMessage], keep_turns: int) -> int:
    """返回清理边界下标：倒数第 keep_turns 个 HumanMessage 的下标（这一轮及之后不清理）。
    HumanMessage 不足 keep_turns 个时返回 0（什么都不清理）。keep_turns <= 0 时返回 len(messages)。"""
    if keep_turns <= 0:
        return len(messages)
    humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(humans) < keep_turns:
        return 0
    return humans[-keep_turns]


def clear_old_tool_outputs(messages: Sequence[BaseMessage], boundary: int, *, min_chars: int = 500) -> list[BaseMessage]:
    """返回新列表：messages[:boundary] 中内容长度 > min_chars 的 ToolMessage 换成占位副本，其余消息原样保留（同一对象）。
    不得修改传入的消息对象。"""
    boundary = max(0, min(boundary, len(messages)))
    result: list[BaseMessage] = []
    for i, m in enumerate(messages):
        if i < boundary and isinstance(m, ToolMessage) and not is_cleared(m):
            n = _content_len(m.content)
            if n > min_chars:
                # model_copy 为浅拷贝且不改动原对象；tool_call_id / name / status / id 随之保留
                m = m.model_copy(update={"content": _placeholder(m.name, n)})
        result.append(m)
    return result


def cleared_chars(before: Sequence[BaseMessage], after: Sequence[BaseMessage]) -> int:
    """返回清理净减少的字符数（原内容长度 - 占位长度，按位置逐条比较），用于日志记录。"""
    return sum(
        _content_len(b.content) - _content_len(a.content)
        for b, a in zip(before, after)
        if b is not a
    )
