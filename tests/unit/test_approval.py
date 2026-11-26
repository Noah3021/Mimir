"""确认策略。"""
from __future__ import annotations

from mimir.approval import ApprovalPolicy


def test_needs_and_allow():
    p = ApprovalPolicy(["run_shell", "write_file"])
    assert p.needs("run_shell")
    assert not p.needs("read_file")
    p.allow("run_shell")
    assert not p.needs("run_shell")
    assert p.needs("write_file")


def test_disabled():
    p = ApprovalPolicy(["run_shell"], enabled=False)
    assert not p.needs("run_shell")
