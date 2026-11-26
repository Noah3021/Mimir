"""根据模型配置创建 LangChain 聊天模型，并统一三类接口的推理（思考）参数。

provider：
- openai    : ChatOpenAI。use_responses_api=true 时传 reasoning={effort, summary}，否则传 reasoning_effort
- deepseek  : ChatDeepSeek（OpenAI 兼容接口，能解析 reasoning_content）。推理只区分开/关，由 thinking_param 控制
- anthropic : ChatAnthropic。thinking={type: enabled, budget_tokens}，按推理强度映射思考预算
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from .config import ConfigError, ModelConfig, normalize_reasoning

logger = logging.getLogger(__name__)

ANTHROPIC_THINKING_BUDGETS = {"low": 2048, "medium": 8192, "high": 16384, "xhigh": 32000}
DEFAULT_MAX_OUTPUT = 8192


def create_llm(mc: ModelConfig, *, streaming: bool = True, reasoning: str | None = None) -> BaseChatModel:
    if not mc.available:
        raise ConfigError(
            f"模型 '{mc.name}' 配置不完整，缺少：{', '.join(mc.missing_fields())}（请检查 .env 或 config.yaml）"
        )
    level = normalize_reasoning(reasoning) if reasoning else mc.reasoning
    builders: dict[str, Callable[[ModelConfig, str, bool], BaseChatModel]] = {
        "openai": _build_openai,
        "deepseek": _build_deepseek,
        "anthropic": _build_anthropic,
    }
    try:
        return builders[mc.provider](mc, level, streaming)
    except ImportError as e:
        raise ConfigError(f"模型 '{mc.name}' 缺少依赖 {e.name or e}，请执行 pip install -r requirements.txt") from e


# ---------------- OpenAI（GPT） ----------------
def _build_openai(mc: ModelConfig, level: str, streaming: bool) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    extra = dict(mc.extra)
    summary = extra.pop("reasoning_summary", "auto")
    kw: dict[str, Any] = dict(
        model=mc.model, api_key=mc.api_key, base_url=mc.base_url, timeout=mc.timeout,
        max_retries=mc.max_retries, streaming=streaming, stream_usage=True,
    )
    if level != "off":
        if extra.get("use_responses_api"):
            kw["reasoning"] = {"effort": level, "summary": summary}
        else:
            kw["reasoning_effort"] = level
    elif mc.temperature is not None:  # 推理模型不支持自定义 temperature
        kw["temperature"] = mc.temperature
    if mc.max_tokens:
        kw["max_tokens"] = mc.max_tokens
    kw.update(extra)
    return ChatOpenAI(**kw)


# ---------------- DeepSeek ----------------
@lru_cache(maxsize=1)
def _deepseek_class():
    from langchain_deepseek import ChatDeepSeek

    class ChatDeepSeekWithReasoning(ChatDeepSeek):
        """在本轮工具调用循环中，把 assistant 消息的 reasoning_content 回传给接口。

        DeepSeek 思考模式下调用工具时，同一轮内后续请求需要带上之前的 reasoning_content；
        之前轮次的不回传（旧版接口带上会报错，新版会忽略）。
        """

        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            try:
                src = self._convert_input(input_).to_messages()
                dst = payload.get("messages") or []
                if len(src) == len(dst):
                    last_human = max((i for i, m in enumerate(src) if isinstance(m, HumanMessage)), default=-1)
                    for m, d in zip(src[last_human + 1:], dst[last_human + 1:], strict=True):
                        if isinstance(m, AIMessage) and isinstance(d, dict) and d.get("role") == "assistant":
                            rc = m.additional_kwargs.get("reasoning_content")
                            if rc:
                                d.setdefault("reasoning_content", rc)
            except Exception:  # 回传失败不影响请求本身
                logger.debug("回传 reasoning_content 失败", exc_info=True)
            return payload

    return ChatDeepSeekWithReasoning


def _build_deepseek(mc: ModelConfig, level: str, streaming: bool) -> BaseChatModel:
    cls = _deepseek_class()
    extra = dict(mc.extra)
    param = extra.pop("thinking_param", "enable_thinking")
    body = dict(extra.pop("extra_body", None) or {})
    on = level != "off"
    if param == "thinking":  # DeepSeek 官方接口：thinking={"type": "enabled"}
        body["thinking"] = {"type": "enabled" if on else "disabled"}
    elif param:  # 阿里云百炼等：enable_thinking=true
        body[param] = on
    kw: dict[str, Any] = dict(
        model=mc.model, api_key=mc.api_key, timeout=mc.timeout, max_retries=mc.max_retries,
        streaming=streaming, stream_usage=True,
    )
    if mc.base_url:
        kw["api_base"] = mc.base_url
    if body:
        kw["extra_body"] = body
    if mc.temperature is not None and not on:
        kw["temperature"] = mc.temperature
    if mc.max_tokens:
        kw["max_tokens"] = mc.max_tokens
    kw.update(extra)
    return cls(**kw)


# ---------------- Anthropic（Claude） ----------------
def _build_anthropic(mc: ModelConfig, level: str, streaming: bool) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    extra = dict(mc.extra)
    budgets = {**ANTHROPIC_THINKING_BUDGETS, **(extra.pop("thinking_budgets", None) or {})}
    max_output = mc.max_tokens or DEFAULT_MAX_OUTPUT
    kw: dict[str, Any] = dict(
        model=mc.model, api_key=mc.api_key, timeout=mc.timeout, max_retries=mc.max_retries,
        streaming=streaming, stream_usage=True,
    )
    if mc.base_url:
        kw["base_url"] = mc.base_url
    if level != "off":
        budget = int(budgets[level])
        kw["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kw["max_tokens"] = max_output + budget  # max_tokens 必须大于思考预算
    else:
        kw["max_tokens"] = max_output
        if mc.temperature is not None:  # 开启思考时不允许设置 temperature
            kw["temperature"] = mc.temperature
    kw.update(extra)
    return ChatAnthropic(**kw)
