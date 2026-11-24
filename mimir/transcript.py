"""明文对话记录（JSONL）：参照 Claude Code，每个会话一个文件，每行一条 JSON 记录，可直接用编辑器查看。

- SQLite checkpointer 负责“恢复对话”；这里只做“只追加”的可读记录，写入失败不影响对话本身。
- 目录结构：<transcripts_dir>/<工作目录名>/<会话ID>.jsonl
- 每条记录的 type：user（你的输入）/ assistant（模型回复，含推理、工具调用、token 用量）/ tool（工具结果）
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .graph import extract_chunk, message_text

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r'[\\/:*?"<>|\s]+')


def workspace_dirname(workspace: Path) -> str:
    """把工作目录路径转成合法的文件夹名，例如 D:\\a\\b -> D-a-b。"""
    return _UNSAFE.sub("-", str(workspace)).strip("-") or "default"


def now_iso() -> str:
    """本地时间（带时区偏移），精确到毫秒。"""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _plain(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


class TranscriptWriter:
    def __init__(self, root: Path, workspace: Path, *, enabled: bool = True):
        self.enabled = enabled
        self.workspace = workspace
        self.dir = root / workspace_dirname(workspace)
        self._seen: dict[str, set[str]] = {}  # 会话 ID -> 已写入的消息 ID

    def path(self, thread_id: str) -> Path:
        return self.dir / f"{thread_id}.jsonl"

    def _load_seen(self, thread_id: str) -> set[str]:
        if thread_id in self._seen:
            return self._seen[thread_id]
        seen: set[str] = set()
        p = self.path(thread_id)
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        mid = json.loads(line).get("message_id")
                    except (json.JSONDecodeError, AttributeError):
                        continue
                    if mid:
                        seen.add(mid)
        self._seen[thread_id] = seen
        return seen

    def sync(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage],
        *,
        profile: str,
        model_id: str,
        backfill_before: int = 0,
        user_time: str | None = None,
    ) -> int:
        """把会话中尚未写入的消息追加到 JSONL 文件，返回新写入的条数。

        backfill_before：本轮开始前已有的消息数。下标小于它且尚未写入的消息，说明是启用记录前的旧消息，
        会标记 backfilled=true（其时间戳是补写时间，不是原始时间）。
        user_time：本轮用户输入的时间，用作本轮 user 记录的时间戳。
        """
        if not self.enabled or not messages:
            return 0
        seen = self._load_seen(thread_id)
        now = now_iso()
        lines, keys = [], []
        for i, m in enumerate(messages):
            key = m.id or f"{m.type}-{i}"
            if key in seen:
                continue
            backfilled = i < backfill_before
            ts = user_time if (isinstance(m, HumanMessage) and user_time and not backfilled) else now
            rec = self._record(m, key, thread_id, profile, model_id, ts)
            if backfilled:
                rec["backfilled"] = True
            lines.append(json.dumps(rec, ensure_ascii=False, default=str))
            keys.append(key)
        if not lines:
            return 0
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path(thread_id).open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        seen.update(keys)
        return len(lines)

    def _record(self, m: BaseMessage, key: str, thread_id: str, profile: str, model_id: str, ts: str) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "type": m.type,
            "timestamp": ts,
            "session_id": thread_id,
            "message_id": key,
            "cwd": str(self.workspace),
        }
        if isinstance(m, HumanMessage):
            rec["type"] = "user"
            rec["content"] = message_text(m) or _plain(m.content)
        elif isinstance(m, AIMessage):
            reasoning, text = extract_chunk(m)
            meta = m.response_metadata or {}
            rec["type"] = "assistant"
            rec["profile"] = meta.get("agent_profile") or profile
            rec["model"] = meta.get("model_name") or meta.get("model") or model_id
            rec["content"] = text
            if reasoning:
                rec["reasoning"] = reasoning
            if m.tool_calls:
                rec["tool_calls"] = [{"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
                                     for tc in m.tool_calls]
            if m.usage_metadata:
                rec["usage"] = dict(m.usage_metadata)
            if meta.get("finish_reason") or meta.get("status"):
                rec["finish_reason"] = meta.get("finish_reason") or meta.get("status")
        elif isinstance(m, ToolMessage):
            rec["type"] = "tool"
            rec["name"] = m.name
            rec["tool_call_id"] = m.tool_call_id
            rec["status"] = m.status
            rec["content"] = message_text(m) or _plain(m.content)
        else:
            rec["content"] = _plain(m.content)
        return rec

    def delete(self, thread_id: str) -> None:
        self._seen.pop(thread_id, None)
        self.path(thread_id).unlink(missing_ok=True)
