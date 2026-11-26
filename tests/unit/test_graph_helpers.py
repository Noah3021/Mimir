"""graph.py 中的纯函数。"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from mimir.graph import (
    _adapt_for_profile,
    _normalize_decision,
    context_state,
    extract_chunk,
    is_transient_error,
    message_text,
)


class APIConnectionError(Exception):
    """与 openai SDK 同名，is_transient_error 按类名识别。"""


class SubConnError(APIConnectionError):
    pass


def chained(inner: BaseException, outer: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException as e:
            raise outer from e
    except BaseException as e:
        return e


def test_transient_error_detection():
    assert is_transient_error(APIConnectionError())
    assert is_transient_error(SubConnError())
    assert is_transient_error(chained(APIConnectionError(), RuntimeError("wrapped")))
    assert not is_transient_error(ValueError("x"))


def test_transient_error_implicit_context():
    try:
        try:
            raise APIConnectionError()
        except APIConnectionError:
            raise RuntimeError("during handling")  # noqa: B904  测试 __context__
    except RuntimeError as e:
        assert is_transient_error(e)


def test_transient_error_cycle_terminates():
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert not is_transient_error(a)


def test_transient_error_real_httpx():
    httpx = pytest.importorskip("httpx")
    assert is_transient_error(httpx.ReadError("x"))
    assert is_transient_error(httpx.ConnectTimeout("x"))


def test_extract_chunk_variants():
    assert extract_chunk(AIMessage("hi")) == ("", "hi")
    responses_api = AIMessage([{"type": "reasoning", "summary": [{"type": "summary_text", "text": "想"}]},
                               {"type": "output_text", "text": "答"}])
    assert extract_chunk(responses_api) == ("想", "答")
    claude = AIMessage([{"type": "thinking", "thinking": "T"}, {"type": "text", "text": "x"}, "y"])
    assert extract_chunk(claude) == ("T", "xy")
    deepseek = AIMessage("答", additional_kwargs={"reasoning_content": "R"})
    assert extract_chunk(deepseek) == ("R", "答")


def test_message_text():
    assert message_text(AIMessage([{"type": "thinking", "thinking": "T"}, {"type": "text", "text": "x"}])) == "x"
    assert message_text(AIMessage("plain")) == "plain"


@pytest.mark.parametrize("decision, expected", [
    ({"approved": True, "feedback": "ok"}, (True, "ok")),
    ({"approved": False}, (False, "")),
    (True, (True, "")),
    (False, (False, "")),
    ("y", (True, "")),
    (" YES ", (True, "")),
    ("是", (True, "")),
    ("n", (False, "")),
    (None, (False, "")),
])
def test_normalize_decision(decision, expected):
    assert _normalize_decision(decision) == expected


def test_adapt_for_profile():
    calls = [{"name": "read_file", "args": {"path": "a"}, "id": "c1", "type": "tool_call"}]
    other = AIMessage([{"type": "thinking", "thinking": "T"}, {"type": "text", "text": "x"}],
                      tool_calls=calls, id="m1", response_metadata={"agent_profile": "other"})
    same = AIMessage("y", response_metadata={"agent_profile": "test"})
    legacy = AIMessage("z")
    out = _adapt_for_profile([other, same, legacy], "test")
    assert out[0].content == "x"
    assert out[0].id == "m1"
    assert out[0].tool_calls[0]["id"] == "c1"
    assert out[1] is same and out[2] is legacy


def test_context_state_defaults():
    assert context_state({}) == ("", 0, 0)
    assert context_state({"summary": "s", "summary_cut": 3, "clear_before": 2}) == ("s", 3, 2)
