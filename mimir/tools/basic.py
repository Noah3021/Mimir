"""基础工具：时间、计算器。"""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool
from pydantic import Field

_WEEKDAYS = "一二三四五六日"

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _factorial(n):
    if n > 1000:
        raise ValueError("阶乘参数过大（上限 1000）")
    return math.factorial(int(n))


_FUNCS = {
    name: getattr(math, name)
    for name in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "log", "log10", "log2",
                 "exp", "floor", "ceil", "radians", "degrees", "gcd", "hypot")
}
_FUNCS.update(abs=abs, round=round, min=min, max=max, factorial=_factorial)


def _safe_eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("指数过大（上限 1000）")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError(f"不支持的表达式成分：{type(node).__name__}")


def build_basic_tools() -> list[BaseTool]:
    @tool
    def get_current_time(
        timezone: Annotated[
            str, Field(description="IANA 时区名，如 Asia/Shanghai、America/New_York；留空表示本机时区")
        ] = "",
    ) -> str:
        """获取当前日期、时间和星期。涉及“今天/现在/几号/星期几”等问题时使用。"""
        if timezone:
            try:
                now = datetime.now(ZoneInfo(timezone))
            except (ZoneInfoNotFoundError, ValueError):
                return f"未知时区：{timezone}（Windows 上需安装 tzdata 包）"
        else:
            now = datetime.now().astimezone()
        return f"{now:%Y-%m-%d %H:%M:%S} {now.tzname()} 星期{_WEEKDAYS[now.weekday()]}"

    @tool
    def calculator(
        expression: Annotated[str, Field(description="数学表达式，如 (3+5)*2、sqrt(16)、2**10、log(100, 10)")],
    ) -> str:
        """精确计算数学表达式。支持 + - * / // % **、括号、常量 pi/e，以及 sqrt/sin/cos/log/exp/abs/round/min/max/factorial 等函数。"""  # noqa: E501 —— 工具描述会原样发给模型，不折行
        try:
            tree = ast.parse(expression.replace("^", "**"), mode="eval")
            result = _safe_eval(tree)
        except ZeroDivisionError:
            return "计算错误：除数为零"
        except (SyntaxError, ValueError, TypeError, OverflowError) as e:
            return f"计算错误：{e}"
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            result = int(result)
        return f"{expression} = {result}"

    return [get_current_time, calculator]
