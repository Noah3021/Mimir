"""create_llm：按 provider 和推理强度生成的模型对象和参数。只构造对象，不发请求。"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mimir import llm
from mimir.config import ConfigError, ModelConfig
from mimir.llm import ANTHROPIC_THINKING_BUDGETS, DEFAULT_MAX_OUTPUT, create_llm


def mc(provider: str, **kw) -> ModelConfig:
    kw.setdefault("name", "t")
    kw.setdefault("model", "test-model")
    kw.setdefault("api_key", "sk-test")
    return ModelConfig(provider=provider, **kw)


# ---------------- 通用 ----------------
def test_incomplete_config_raises():
    with pytest.raises(ConfigError, match="配置不完整"):
        create_llm(mc("openai", api_key=None))


def test_missing_dependency_becomes_config_error(monkeypatch):
    def boom(*a, **k):
        raise ImportError("no module", name="langchain_openai")

    monkeypatch.setattr(llm, "_build_openai", boom)
    with pytest.raises(ConfigError, match="langchain_openai"):
        create_llm(mc("openai"))


def test_reasoning_argument_overrides_config():
    m = create_llm(mc("openai", reasoning="low"), reasoning="high")
    assert m.reasoning_effort == "high"


# ---------------- OpenAI ----------------
def test_openai_basic():
    from langchain_openai import ChatOpenAI

    m = create_llm(mc("openai", base_url="https://example.invalid/v1", max_tokens=1000), streaming=False)
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "test-model"
    assert m.streaming is False
    assert m.max_tokens == 1000
    assert m.reasoning_effort is None


def test_openai_reasoning_effort_drops_temperature():
    m = create_llm(mc("openai", reasoning="medium", temperature=0.5))
    assert m.reasoning_effort == "medium"
    assert m.temperature is None  # 推理模型不支持自定义 temperature


def test_openai_off_keeps_temperature():
    assert create_llm(mc("openai", temperature=0.5)).temperature == 0.5


def test_openai_responses_api():
    m = create_llm(mc("openai", reasoning="high", extra={"use_responses_api": True, "reasoning_summary": "concise"}))
    assert m.reasoning == {"effort": "high", "summary": "concise"}
    assert m.reasoning_effort is None
    assert m.use_responses_api is True


# ---------------- DeepSeek ----------------
def test_deepseek_default_param_off():
    m = create_llm(mc("deepseek", base_url="https://example.invalid/v1", temperature=0.3))
    assert isinstance(m, llm._deepseek_class())
    assert m.extra_body == {"enable_thinking": False}
    assert m.temperature == 0.3
    assert m.api_base == "https://example.invalid/v1"


def test_deepseek_official_thinking_param_on():
    m = create_llm(mc("deepseek", reasoning="low", temperature=0.3, extra={"thinking_param": "thinking"}))
    assert m.extra_body == {"thinking": {"type": "enabled"}}
    assert m.temperature != 0.3  # 开启思考时不传 temperature


def test_deepseek_thinking_param_disabled():
    m = create_llm(mc("deepseek", reasoning="high", extra={"thinking_param": ""}))
    assert not m.extra_body


def test_deepseek_passes_back_reasoning_content_only_for_current_turn():
    m = create_llm(mc("deepseek", reasoning="high"))
    messages = [
        HumanMessage("q1"),
        AIMessage("a1", additional_kwargs={"reasoning_content": "旧一轮的推理"}),
        HumanMessage("q2"),
        AIMessage("", additional_kwargs={"reasoning_content": "本轮的推理"},
                  tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}]),
        ToolMessage("结果", tool_call_id="c1"),
    ]
    payload = m._get_request_payload(messages)
    assistants = [d for d in payload["messages"] if d.get("role") == "assistant"]
    assert len(assistants) == 2
    assert assistants[0].get("reasoning_content") != "旧一轮的推理"
    assert assistants[1].get("reasoning_content") == "本轮的推理"


# ---------------- Anthropic ----------------
def test_anthropic_off():
    from langchain_anthropic import ChatAnthropic

    m = create_llm(mc("anthropic", temperature=0.2))
    assert isinstance(m, ChatAnthropic)
    assert m.max_tokens == DEFAULT_MAX_OUTPUT
    assert m.temperature == 0.2
    assert not m.thinking


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh"])
def test_anthropic_thinking_budget(level):
    m = create_llm(mc("anthropic", reasoning=level, max_tokens=4000, temperature=0.2))
    budget = ANTHROPIC_THINKING_BUDGETS[level]
    assert m.thinking == {"type": "enabled", "budget_tokens": budget}
    assert m.max_tokens == 4000 + budget  # max_tokens 必须大于思考预算
    assert m.temperature != 0.2


def test_anthropic_custom_budgets():
    m = create_llm(mc("anthropic", reasoning="low", extra={"thinking_budgets": {"low": 1024}}))
    assert m.thinking["budget_tokens"] == 1024
    assert m.max_tokens == DEFAULT_MAX_OUTPUT + 1024
