"""Shell 工具。命令只用 cmd 和 sh 都支持的写法。"""
from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

from mimir.tools.shell import build_shell_tools

PY = f'"{sys.executable}"'


def shell(cfg, **overrides):
    return build_shell_tools(dataclasses.replace(cfg, **overrides))[0]


@pytest.fixture
def run(tools_config):
    return shell(tools_config)


@pytest.fixture
def fake_run(monkeypatch):
    """替换 subprocess.run，只记录命令、不真正执行（用于测试危险命令）。"""
    seen = []

    def fake(cmd, *args, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("mimir.tools.shell.subprocess.run", fake)
    return seen


def test_echo(run):
    out = run.invoke({"command": "echo hello"})
    assert out.startswith("退出码：0")
    assert "[stdout]\nhello" in out


def test_exit_code_and_no_output(run):
    out = run.invoke({"command": "exit 3"})
    assert out.startswith("退出码：3")
    assert "（无输出）" in out


def test_stderr(run):
    out = run.invoke({"command": "echo oops 1>&2"})
    assert "[stderr]\noops" in out


def test_runs_in_workspace(run, workspace):
    (workspace / "marker.txt").write_text("x", encoding="utf-8")
    out = run.invoke({"command": f'{PY} -c "import os; print(os.listdir())"'})
    assert "marker.txt" in out


def test_timeout(run):
    out = run.invoke({"command": f'{PY} -c "import time; time.sleep(3)"', "timeout": 1})
    assert "超时" in out


def test_output_truncated(tools_config):
    run = shell(tools_config, max_output_chars=50)
    out = run.invoke({"command": f"{PY} -c \"print('x' * 200)\""})
    assert "已截断" in out
    assert len(out) < 150


def test_disabled(tools_config):
    assert build_shell_tools(dataclasses.replace(tools_config, shell_enabled=False)) == []


@pytest.mark.parametrize("command", ["rm -rf /", "RM -RF /", "sudo rm -rf / --no-preserve-root"])
def test_blocked_patterns(tools_config, fake_run, command):
    run = shell(tools_config, blocked_patterns=["rm -rf /"])
    assert "已拒绝" in run.invoke({"command": command})
    assert fake_run == []


@pytest.mark.xfail(reason="已知问题：黑名单按子串匹配，多一个空格即可绕过；改为规则式权限后应通过", strict=True)
def test_blocked_patterns_whitespace_bypass(tools_config, fake_run):
    run = shell(tools_config, blocked_patterns=["rm -rf /"])
    out = run.invoke({"command": "rm  -rf /"})
    assert "已拒绝" in out and fake_run == []
