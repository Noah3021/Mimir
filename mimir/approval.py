"""危险操作的人工确认策略。"""
from __future__ import annotations

from typing import Iterable


class ApprovalPolicy:
    """决定某个工具调用是否需要用户确认。

    注意：同一轮对话内（中断 -> 恢复之间）策略不能变化，否则 LangGraph 重放节点时
    interrupt 的顺序会错位。因此“本次运行总是允许”由 CLI 在一轮结束后再调用 allow()。
    """

    def __init__(self, required: Iterable[str], enabled: bool = True):
        self.required = set(required)
        self.enabled = enabled
        self.session_allowed: set[str] = set()

    def needs(self, tool_name: str) -> bool:
        return self.enabled and tool_name in self.required and tool_name not in self.session_allowed

    def allow(self, tool_name: str) -> None:
        self.session_allowed.add(tool_name)
