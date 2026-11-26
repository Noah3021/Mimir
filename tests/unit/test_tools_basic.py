"""基础工具：计算器、当前时间。"""
from __future__ import annotations

import pytest

from mimir.tools.basic import build_basic_tools


@pytest.fixture
def tools():
    return {t.name: t for t in build_basic_tools()}


@pytest.mark.parametrize("expr, expected", [
    ("(3+5)*2", "= 16"),
    ("sqrt(16)", "= 4"),
    ("2^10", "= 1024"),
    ("7 // 2", "= 3"),
    ("-pi", "= -3.14159"),
    ("max(1, 5, 3)", "= 5"),
])
def test_calculator(tools, expr, expected):
    assert expected in tools["calculator"].invoke({"expression": expr})


@pytest.mark.parametrize("expr, error", [
    ("__import__('os').system('echo hi')", "计算错误"),
    ("open('x')", "计算错误"),
    ("1/0", "除数为零"),
    ("2**100000", "指数过大"),
    ("factorial(5000)", "阶乘参数过大"),
    ("1 +", "计算错误"),
])
def test_calculator_rejects(tools, expr, error):
    assert error in tools["calculator"].invoke({"expression": expr})


def test_current_time(tools):
    assert "星期" in tools["get_current_time"].invoke({})
    out = tools["get_current_time"].invoke({"timezone": "Asia/Shanghai"})
    assert "星期" in out and "CST" in out
    assert "未知时区" in tools["get_current_time"].invoke({"timezone": "Mars/Base"})
