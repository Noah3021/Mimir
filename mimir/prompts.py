"""系统提示词渲染。"""
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable

WEEKDAYS = "一二三四五六日"


def render_system_prompt(
    template: str,
    *,
    agent_name: str,
    workspace: Path,
    model: str,
    tools: Iterable[str],
) -> str:
    """用简单替换填充占位符（不用 str.format，避免提示词里的花括号报错）。

    时间只精确到日期：系统提示词位于请求最前面，其中任何字符变化都会让其后的
    全部历史消息无法命中提示词缓存。需要精确时间时由模型调用 get_current_time。
    """
    now = datetime.now().astimezone()
    date = now.strftime("%Y-%m-%d")
    values = {
        "agent_name": agent_name,
        "date": date,
        "now": date,  # 兼容旧配置：{now} 同样只到日期
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
