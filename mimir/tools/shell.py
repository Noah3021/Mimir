"""Shell 命令工具：在工作目录中执行命令。"""
from __future__ import annotations

import locale
import os
import platform
import subprocess
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from ..config import ToolsConfig


def _decode(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # Windows 的 cmd 默认输出 GBK 等本地编码
        return data.decode(locale.getpreferredencoding(False) or "utf-8", errors="replace")


def build_shell_tools(cfg: ToolsConfig) -> list[BaseTool]:
    if not cfg.shell_enabled:
        return []

    root = cfg.workspace
    executable = cfg.shell_executable
    limit = cfg.max_output_chars
    if executable:
        shell_desc = executable
    else:
        shell_desc = "cmd.exe" if os.name == "nt" else os.environ.get("SHELL", "/bin/sh")

    @tool
    def run_shell(
        command: Annotated[str, Field(description="要执行的命令")],
        timeout: Annotated[int, Field(description="超时时间（秒）")] = cfg.shell_timeout,
    ) -> str:
        """在工作目录中执行 shell 命令，返回退出码、标准输出和标准错误。不要执行交互式或长时间常驻的命令。"""
        lowered = command.lower()
        for pat in cfg.blocked_patterns:
            if pat.lower() in lowered:
                return f"命令包含被禁止的片段 “{pat}”，已拒绝执行。"
        timeout = max(1, min(int(timeout), 600))
        try:
            if executable:
                proc = subprocess.run([executable, "-c", command], cwd=root, capture_output=True,
                                      timeout=timeout, stdin=subprocess.DEVNULL)
            else:
                proc = subprocess.run(command, shell=True, cwd=root, capture_output=True,
                                      timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return f"命令执行超时（{timeout} 秒），已终止。"
        out, err = _decode(proc.stdout).strip(), _decode(proc.stderr).strip()
        parts = [f"退出码：{proc.returncode}"]
        if out:
            parts.append(f"[stdout]\n{out}")
        if err:
            parts.append(f"[stderr]\n{err}")
        if not out and not err:
            parts.append("（无输出）")
        text = "\n".join(parts)
        if len(text) > limit:
            text = text[:limit] + f"\n...（输出过长，已截断，共 {len(text)} 字符）"
        return text

    run_shell.description += f"\n当前系统：{platform.system()}，使用的 shell：{shell_desc}，请使用该 shell 的语法。"
    return [run_shell]
