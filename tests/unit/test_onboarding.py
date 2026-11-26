"""首次配置：默认配置 + 用户配置合并、mimir init 向导、--import 导入旧目录、mimir doctor。

conftest 已把 MIMIR_HOME 指向临时目录（fixture: mimir_home），不会碰到本机的 ~/.mimir。
"""
from __future__ import annotations

import io

import pytest
import yaml
from rich.console import Console

from mimir.cli import main
from mimir.config import SetupRequired, deep_merge, load_settings
from mimir.onboarding import Prompter, deep_diff, import_legacy, run_doctor, run_init, write_env
from mimir.providers import env_key


def make_console() -> Console:
    return Console(file=io.StringIO(), width=200, color_system=None)


def scripted(console: Console, *answers: str, secret: str = "sk-test") -> Prompter:
    """按顺序回答 ask；secret 固定返回同一个 Key。答案用完时报错，便于发现多问了问题。"""
    it = iter(answers)

    def ask(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"没有准备答案的提问：{prompt}") from None

    return Prompter(console, ask_fn=ask, secret_fn=lambda prompt: secret)


def ok_tester(mc) -> None:
    return None


def write_user_config(home, data: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def restore_env(monkeypatch):
    """write_env 会直接写 os.environ；先用 monkeypatch 占位，测试结束后自动还原。"""

    def _register(*names: str) -> None:
        for n in names:
            monkeypatch.setenv(n, "placeholder")
            monkeypatch.delenv(n)

    return _register


# ---------------- 配置合并 ----------------
def test_deep_merge_overrides_nested_and_replaces_lists():
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2], "s": "base"}
    merged = deep_merge(base, {"a": {"y": 3, "z": 4}, "l": [9]})
    assert merged == {"a": {"x": 1, "y": 3, "z": 4}, "l": [9], "s": "base"}
    assert base == {"a": {"x": 1, "y": 2}, "l": [1, 2], "s": "base"}  # 不修改原对象


def test_deep_diff_keeps_only_changes():
    assert deep_diff({"a": {"x": 1, "y": 5}, "b": 2}, {"a": {"x": 1, "y": 2}, "b": 2}) == {"a": {"y": 5}}
    assert deep_diff({"a": 1}, {"a": 1}) is None


def test_missing_user_config_requires_setup(mimir_home):
    with pytest.raises(SetupRequired, match="mimir init"):
        load_settings()


def test_user_config_only_overrides_what_it_sets(mimir_home, workspace):
    write_user_config(mimir_home, {
        "agent": {"default_model": "m1", "context": {"window": 64000}},
        "models": {"m1": {"provider": "openai", "model": "x", "api_key": "k"}},
    })
    s = load_settings(workspace=workspace)
    assert s.agent.context.window == 64000
    assert s.agent.context.high == 0.8                      # 未覆盖的项来自默认配置
    assert "{agent_name}" in s.agent.system_prompt
    assert "run_shell" in s.tools.require_approval
    assert s.memory.db_path == (mimir_home / "data" / "sessions.db").resolve()  # 相对用户配置目录


def test_env_key():
    assert env_key("claude") == "MIMIR_CLAUDE_API_KEY"
    assert env_key("qwen-max.2") == "MIMIR_QWEN_MAX_2_API_KEY"


def test_write_env_updates_in_place_and_keeps_other_lines(tmp_path, restore_env):
    restore_env("A", "B")
    p = tmp_path / ".env"
    p.write_text("# 注释\nA=old\nOTHER=1\n", encoding="utf-8")
    write_env(p, {"A": "new", "B": 'has "quote"'})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines == ["# 注释", 'A="new"', "OTHER=1", 'B="has \\"quote\\""']


# ---------------- mimir init ----------------
def test_init_first_model(mimir_home, workspace, restore_env):
    restore_env("MIMIR_DEEPSEEK_API_KEY")
    console = make_console()
    # 选 3) DeepSeek；模型名和配置名都用默认值
    assert run_init(console, prompter=scripted(console, "3", "", ""), tester=ok_tester) == 0

    raw = yaml.safe_load((mimir_home / "config.yaml").read_text(encoding="utf-8"))
    assert raw["agent"] == {"default_model": "deepseek"}
    m = raw["models"]["deepseek"]
    assert m["api_key"] == "${MIMIR_DEEPSEEK_API_KEY}"       # 配置里只有引用，没有真实的 Key
    assert m["model"] == "deepseek-chat" and m["provider"] == "deepseek"
    assert "sk-test" not in (mimir_home / "config.yaml").read_text(encoding="utf-8")
    assert 'MIMIR_DEEPSEEK_API_KEY="sk-test"' in (mimir_home / ".env").read_text(encoding="utf-8")

    s = load_settings(workspace=workspace)                   # 生成的配置能直接加载
    assert s.agent.default_model == "deepseek"
    assert s.models["deepseek"].api_key == "sk-test"


def test_init_second_model_keeps_first_and_asks_default(mimir_home, restore_env):
    restore_env("MIMIR_DEEPSEEK_API_KEY", "MIMIR_GEMINI_API_KEY")
    console = make_console()
    run_init(console, prompter=scripted(console, "3", "", ""), tester=ok_tester)
    # 选 4) Gemini，不设为默认
    assert run_init(console, prompter=scripted(console, "4", "", "", "n"), tester=ok_tester) == 0
    raw = yaml.safe_load((mimir_home / "config.yaml").read_text(encoding="utf-8"))
    assert list(raw["models"]) == ["deepseek", "gemini"]
    assert raw["agent"]["default_model"] == "deepseek"


def test_init_custom_service(mimir_home, restore_env):
    restore_env("MIMIR_BAILIAN_API_KEY")
    console = make_console()
    answers = ("5", "https://example.com/v1", "2", "qwen-max", "bailian")  # 自定义 → DeepSeek 兼容
    assert run_init(console, prompter=scripted(console, *answers), tester=ok_tester) == 0
    m = yaml.safe_load((mimir_home / "config.yaml").read_text(encoding="utf-8"))["models"]["bailian"]
    assert m["provider"] == "deepseek" and m["base_url"] == "https://example.com/v1"
    assert m["model"] == "qwen-max" and m["thinking_param"] == "enable_thinking"


def test_init_connection_failure_then_quit_saves_nothing(mimir_home):
    def failing(mc):
        raise RuntimeError("401 Unauthorized")

    console = make_console()
    assert run_init(console, prompter=scripted(console, "3", "", "", "q"), tester=failing) == 1
    assert not (mimir_home / "config.yaml").exists()
    assert not (mimir_home / ".env").exists()
    assert "401 Unauthorized" in console.file.getvalue()


def test_init_connection_failure_save_anyway(mimir_home, restore_env):
    restore_env("MIMIR_DEEPSEEK_API_KEY")

    def failing(mc):
        raise RuntimeError("timeout")

    console = make_console()
    assert run_init(console, prompter=scripted(console, "3", "", "", "s"), tester=failing) == 0
    assert (mimir_home / "config.yaml").is_file()


def test_init_cancel_with_ctrl_c(mimir_home):
    console = make_console()

    def interrupt(prompt):
        raise KeyboardInterrupt

    assert run_init(console, prompter=Prompter(console, ask_fn=interrupt), tester=ok_tester) == 1
    assert not (mimir_home / "config.yaml").exists()


# ---------------- mimir init --import ----------------
def test_import_legacy_directory(tmp_path, mimir_home, restore_env):
    restore_env("MIMIR_DS_API_KEY")
    src = tmp_path / "old_mimir"
    (src / "data").mkdir(parents=True)
    (src / "data" / "sessions.db").write_bytes(b"db")
    (src / ".env").write_text('API_KEY="sk-old"\nMODEL_ID="ds-model"\n', encoding="utf-8")
    old_config = (
        "agent:\n  default_model: ds\n  recursion_limit: 80\n"
        "models:\n"
        "  ds: {provider: deepseek, model: '${MODEL_ID}', api_key: '${API_KEY}', base_url: '${BASE_URL}'}\n"
        "  nokey: {provider: openai, model: x, api_key: '${MISSING_KEY_FOR_TEST}'}\n"
    )
    (src / "config.yaml").write_text(old_config, encoding="utf-8")

    r = import_legacy(src)
    assert r.models == ["ds", "nokey"] and r.no_key == ["nokey"]
    assert r.overrides == ["agent.default_model", "agent.recursion_limit"]
    assert r.data == "copied"

    raw = yaml.safe_load((mimir_home / "config.yaml").read_text(encoding="utf-8"))
    ds = raw["models"]["ds"]
    assert ds == {"provider": "deepseek", "model": "ds-model", "api_key": "${MIMIR_DS_API_KEY}"}  # 空的 base_url 去掉
    assert raw["agent"] == {"default_model": "ds", "recursion_limit": 80}
    assert 'MIMIR_DS_API_KEY="sk-old"' in (mimir_home / ".env").read_text(encoding="utf-8")
    assert (mimir_home / "data" / "sessions.db").read_bytes() == b"db"
    assert (src / "config.yaml").read_text(encoding="utf-8") == old_config  # 原目录不改动

    again = import_legacy(src)                               # 再导入一次：同名模型跳过，数据不覆盖
    assert again.skipped == ["ds", "nokey"] and again.data == "exists"


# ---------------- mimir doctor / 入口 ----------------
def test_doctor_without_config_reports_problem(mimir_home):
    console = make_console()
    assert run_doctor(console, offline=True) == 1
    assert "mimir init" in console.file.getvalue()


def test_doctor_tests_each_model(mimir_home, restore_env):
    write_user_config(mimir_home, {
        "agent": {"default_model": "good"},
        "models": {"good": {"provider": "openai", "model": "a", "api_key": "k"},
                   "bad": {"provider": "openai", "model": "b", "api_key": "k"}},
    })

    def tester(mc):
        if mc.name == "bad":
            raise RuntimeError("连接被拒绝")

    console = make_console()
    assert run_doctor(console, tester=tester) == 0          # 至少有一个模型可用
    text = console.file.getvalue()
    assert "模型 good（默认）" in text and "连接被拒绝" in text


def test_main_without_config_non_interactive(mimir_home, capsys):
    assert main(["-p", "你好"]) == 1
    assert "mimir init" in capsys.readouterr().out


def test_main_dispatches_doctor_subcommand(mimir_home, capsys):
    assert main(["doctor", "--offline"]) == 1
    assert "环境检查" in capsys.readouterr().out
