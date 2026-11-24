"""系统提示词渲染。"""
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Iterable

WEEKDAYS = "一二三四五六日"


def render_system_prompt(
    template: str,
    *,
    agent_name: str,
    workspace: Path,
    model: str,
    tools: Iterable[str],
) -> str:
    """用简单替换填充占位符（不用 str.format，避免提示词里的花括号报错）。"""
    now = datetime.now().astimezone()
    values = {
        "agent_name": agent_name,
        "now": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": f"星期{WEEKDAYS[now.weekday()]}",
        "os": f"{platform.system()} {platform.release()}",
        "workspace": str(workspace),
        "model": model,
        "tools": ", ".join(tools) or "无",
    }
    text = template
    for key, val in values.items():
        text = text.replace("{" + key + "}", val)
    return text.strip()
