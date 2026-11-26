"""公共 fixture。所有测试离线运行：不访问网络，不读取真实的 .env。"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from mimir.approval import ApprovalPolicy
from mimir.config import ToolsConfig
from mimir.graph import build_agent_graph

SYSTEM_PROMPT = "你是测试助手。"


@pytest.fixture(autouse=True)
def no_env_files(monkeypatch):
    """不加载任何 .env，避免读到本机的真实密钥。"""
    monkeypatch.setattr("mimir.config.load_env_files", lambda: [])


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def tools_config(workspace):
    return ToolsConfig(workspace=workspace, max_output_chars=8000, shell_timeout=10)


@pytest.fixture
def make_graph():
    """用假模型构造完整的图：make_graph(model, tools=[...], approval=..., **build_agent_graph 的其他参数)。"""

    def factory(model, *, tools=(), approval=None, checkpointer=None, system_prompt=SYSTEM_PROMPT, **kwargs):
        kwargs.setdefault("max_stream_retries", 5)
        return build_agent_graph(
            llm=model,
            tools=list(tools),
            system_prompt=lambda: system_prompt,
            approval=approval or ApprovalPolicy([]),
            checkpointer=checkpointer or InMemorySaver(),
            profile="test",
            **kwargs,
        )

    return factory
