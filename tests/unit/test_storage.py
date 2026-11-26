"""会话元信息（SessionStore）与明文对话记录（TranscriptWriter）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mimir.memory import SessionStore
from mimir.transcript import TranscriptWriter, workspace_dirname


@pytest.fixture
def store(tmp_path):
    s = SessionStore(tmp_path / "db" / "sessions.db")
    yield s
    s.close()


def test_session_ids_unique():
    ids = {SessionStore.new_session_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(i) == 12 for i in ids)


def test_touch_list_find_delete(store):
    store.touch("abc111", "第一个", "m1")
    store.touch("abc222", "第二个", "m1")
    store.touch("xyz333", "第三个", "m2")
    assert {s.thread_id for s in store.list()} == {"abc111", "abc222", "xyz333"}
    assert {s.thread_id for s in store.find("abc")} == {"abc111", "abc222"}
    assert [s.thread_id for s in store.find("xyz")] == ["xyz333"]

    store.touch("abc111", "新标题", "m2")  # 已存在：更新模型和时间，标题不变
    s = store.find("abc111")[0]
    assert (s.title, s.model) == ("第一个", "m2")

    store.delete("abc111")
    assert store.find("abc111") == []
    assert len(store.list(limit=1)) == 1


def test_workspace_dirname():
    assert workspace_dirname(Path("D:\\a\\b c")) == "D-a-b-c"
    assert workspace_dirname(Path("/")) == "default"


def conversation():
    return [
        HumanMessage("读一下 a.py", id="h1"),
        AIMessage("", id="a1", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}],
                  usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                  response_metadata={"agent_profile": "deepseek", "finish_reason": "tool_calls"}),
        ToolMessage("内容", tool_call_id="c1", name="read_file", id="t1"),
        AIMessage([{"type": "thinking", "thinking": "想"}, {"type": "text", "text": "读完了"}], id="a2"),
    ]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_transcript_sync(tmp_path, workspace):
    tw = TranscriptWriter(tmp_path / "tr", workspace)
    msgs = conversation()
    assert tw.sync("s1", msgs, profile="p", model_id="m", user_time="T0") == 4
    recs = read_jsonl(tw.path("s1"))
    assert [r["type"] for r in recs] == ["user", "assistant", "tool", "assistant"]
    assert recs[0]["timestamp"] == "T0" and recs[0]["content"] == "读一下 a.py"
    assert recs[1]["profile"] == "deepseek" and recs[1]["model"] == "m"
    assert recs[1]["tool_calls"][0]["name"] == "read_file"
    assert recs[1]["usage"]["input_tokens"] == 10
    assert recs[1]["finish_reason"] == "tool_calls"
    assert recs[2]["tool_call_id"] == "c1" and recs[2]["status"] == "success"
    assert recs[3]["content"] == "读完了" and recs[3]["reasoning"] == "想"
    assert all(r["session_id"] == "s1" and r["cwd"] == str(workspace) for r in recs)

    assert tw.sync("s1", msgs, profile="p", model_id="m") == 0  # 已写入的不重复
    assert tw.sync("s1", [*msgs, HumanMessage("下一轮", id="h2")], profile="p", model_id="m") == 1

    fresh = TranscriptWriter(tmp_path / "tr", workspace)  # 新实例从文件恢复已写入的 ID
    assert fresh.sync("s1", msgs, profile="p", model_id="m") == 0


def test_transcript_backfill(tmp_path, workspace):
    tw = TranscriptWriter(tmp_path / "tr", workspace)
    tw.sync("s2", conversation()[:3], profile="p", model_id="m", backfill_before=2, user_time="T0")
    recs = read_jsonl(tw.path("s2"))
    assert [r.get("backfilled", False) for r in recs] == [True, True, False]
    assert recs[0]["timestamp"] != "T0"  # 补写的用户消息不用本轮时间


def test_transcript_disabled_and_delete(tmp_path, workspace):
    off = TranscriptWriter(tmp_path / "tr", workspace, enabled=False)
    assert off.sync("s3", conversation(), profile="p", model_id="m") == 0
    assert not off.path("s3").exists()

    tw = TranscriptWriter(tmp_path / "tr", workspace)
    tw.sync("s3", conversation(), profile="p", model_id="m")
    tw.delete("s3")
    assert not tw.path("s3").exists()
    assert tw.sync("s3", conversation(), profile="p", model_id="m") == 4
