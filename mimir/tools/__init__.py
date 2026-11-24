"""工具注册。新增工具：在对应模块的 build_* 函数中添加，或新建模块并在这里汇总。"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from ..config import ToolsConfig
from .basic import build_basic_tools
from .filesystem import build_filesystem_tools
from .shell import build_shell_tools


def build_tools(cfg: ToolsConfig) -> list[BaseTool]:
    tools: list[BaseTool] = [
        *build_basic_tools(),
        *build_filesystem_tools(cfg),
        *build_shell_tools(cfg),
    ]
    disabled = set(cfg.disabled)
    return [t for t in tools if t.name not in disabled]
