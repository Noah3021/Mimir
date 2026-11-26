"""配置加载：环境变量插值、校验、路径解析。"""
from __future__ import annotations

import pytest

from mimir.config import ConfigError, _interpolate, load_settings, normalize_reasoning

BASE = """
agent:
  default_model: m1
models:
  m1:
    provider: openai
    model: ${T_MODEL:gpt-x}
    api_key: ${T_KEY}
    timeout: ${T_TIMEOUT:30}
"""


@pytest.fixture
def cfg_dir(tmp_path):
    d = tmp_path / "cfg"
    d.mkdir()
    return d


def load(cfg_dir, text, workspace):
    p = cfg_dir / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return load_settings(p, workspace=workspace)


def test_env_interpolation_and_defaults(cfg_dir, workspace, monkeypatch):
    for k in ("T_MODEL", "T_KEY", "T_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)
    m = load(cfg_dir, BASE, workspace).models["m1"]
    assert m.model == "gpt-x"
    assert m.api_key is None
    assert m.timeout == 30.0
    assert m.missing_fields() == ["api_key"]
    assert not m.available

    monkeypatch.setenv("T_KEY", "sk-test")
    m = load(cfg_dir, BASE, workspace).models["m1"]
    assert m.api_key == "sk-test"
    assert m.available


def test_interpolate_restores_scalars_only_for_whole_value(monkeypatch):
    monkeypatch.delenv("T_X", raising=False)
    assert _interpolate("${T_X:5}") == 5
    assert _interpolate("${T_X:true}") is True
    assert _interpolate("${T_X:0.5}") == 0.5
    assert _interpolate("a-${T_X:1}") == "a-1"
    assert _interpolate({"k": ["${T_X:-x}"]}) == {"k": ["x"]}


@pytest.mark.parametrize("ctx", [
    "{low: 0.8, high: 0.5}",
    "{low: 0.5, high: 1.5}",
    "{low: 0, high: 0.8}",
    "{window: 0}",
])
def test_invalid_context_rejected(cfg_dir, workspace, ctx):
    text = BASE.replace("agent:\n", f"agent:\n  context: {ctx}\n")
    with pytest.raises(ConfigError):
        load(cfg_dir, text, workspace)


@pytest.mark.parametrize("value, expected", [
    (None, "off"), ("", "off"), (False, "off"), (True, "high"), ("max", "xhigh"),
    ("开", "high"), ("关", "off"), ("Medium", "medium"), ({"effort": "low"}, "low"),
])
def test_normalize_reasoning(value, expected):
    assert normalize_reasoning(value) == expected


def test_normalize_reasoning_invalid():
    with pytest.raises(ConfigError):
        normalize_reasoning("bogus")


def test_unknown_provider(cfg_dir, workspace):
    with pytest.raises(ConfigError, match="provider"):
        load(cfg_dir, BASE.replace("provider: openai", "provider: nope"), workspace)


def test_default_model_must_exist(cfg_dir, workspace):
    with pytest.raises(ConfigError, match="default_model"):
        load(cfg_dir, BASE.replace("default_model: m1", "default_model: m9"), workspace)


def test_relative_paths_resolve_against_config_dir(cfg_dir, workspace):
    s = load(cfg_dir, BASE + "memory:\n  db_path: data/x.db\n", workspace)
    assert s.memory.db_path == (cfg_dir / "data" / "x.db").resolve()
    assert s.memory.transcripts_dir == (cfg_dir / "data" / "transcripts").resolve()
    assert s.tools.workspace == workspace.resolve()


def test_missing_workspace_rejected(cfg_dir, tmp_path):
    with pytest.raises(ConfigError, match="工作目录"):
        load(cfg_dir, BASE, tmp_path / "nope")


def test_model_context_window_and_extra_passthrough(cfg_dir, workspace):
    text = BASE + "    context_window: 32000\n    use_responses_api: true\n"
    m = load(cfg_dir, text, workspace).models["m1"]
    assert m.context_window == 32000
    assert m.extra == {"use_responses_api": True}


def test_legacy_max_history_messages_ignored(cfg_dir, workspace):
    s = load(cfg_dir, BASE.replace("agent:\n", "agent:\n  max_history_messages: 10\n"), workspace)
    assert s.agent.context.window == 128_000
