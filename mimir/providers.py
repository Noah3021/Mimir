"""供应商预设：`mimir init` 让用户选择的几家模型服务。

每个预设决定底层用哪种接口（provider）、默认的 Base URL 和模型名，以及推理参数怎么传。
用户通常只需要填 API Key；选“自定义服务”时再自己填 Base URL 和接口类型。

注意：默认模型名会随各家发布新模型而过时，发版前请对照官网确认。只需要改这一个文件。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Preset:
    key: str                  # 写入配置时的默认模型名称（/model 切换时用），也用于环境变量名
    label: str                # 菜单里显示的名字
    provider: str             # 底层接口：openai | deepseek | anthropic
    base_url: str | None      # None = 使用 SDK 的官方地址；自定义服务由用户填写
    model: str                # 默认模型名，用户可在向导里改
    reasoning: str = "medium"
    key_url: str = ""         # 在哪里申请 API Key（提示用户）
    extra: dict[str, Any] = field(default_factory=dict)  # 额外写入模型配置的字段


PRESETS: list[Preset] = [
    Preset(
        key="openai", label="OpenAI（GPT）", provider="openai", base_url=None, model="gpt-5",
        key_url="https://platform.openai.com/api-keys",
        extra={"use_responses_api": True, "reasoning_summary": "auto"},
    ),
    Preset(
        key="claude", label="Claude（Anthropic）", provider="anthropic", base_url=None, model="claude-sonnet-5-5",
        key_url="https://console.anthropic.com/settings/keys",
        extra={"max_tokens": 8192},
    ),
    Preset(
        key="deepseek", label="DeepSeek", provider="deepseek", base_url="https://api.deepseek.com",
        model="deepseek-chat", reasoning="off", key_url="https://platform.deepseek.com/api_keys",
        extra={"thinking_param": "thinking"},
    ),
    Preset(
        key="gemini", label="Gemini（Google）", provider="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", model="gemini-2.5-flash",
        key_url="https://aistudio.google.com/apikey",
    ),
]

CUSTOM_LABEL = "自定义服务（阿里云百炼、OpenRouter、硅基流动、Ollama 等）"

# 自定义服务可选的接口类型：(provider, 菜单说明, 写入配置的额外字段)
CUSTOM_PROTOCOLS: list[tuple[str, str, dict[str, Any]]] = [
    ("openai", "OpenAI 兼容（绝大多数服务）", {}),
    ("deepseek", "DeepSeek 兼容（能显示 reasoning_content 思考过程，如百炼上的 DeepSeek / Qwen）",
     {"thinking_param": "enable_thinking"}),
    ("anthropic", "Anthropic 兼容（Claude 中转）", {"max_tokens": 8192}),
]


def env_key(name: str) -> str:
    """模型名称 -> 保存 API Key 的环境变量名，例如 claude -> MIMIR_CLAUDE_API_KEY。"""
    safe = "".join(c if c.isalnum() else "_" for c in name.upper()).strip("_") or "MODEL"
    return f"MIMIR_{safe}_API_KEY"
