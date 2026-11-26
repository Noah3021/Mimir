"""项目说明（MIMIR.md）与 git 状态。"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mimir.project_context import (
    PROJECT_UPDATE_KEY,
    ProjectContext,
    collect_git_status,
    inject_context,
    inject_updates,
    latest_update,
    read_project_file,
    render_update,
)


def test_read_project_file(workspace):
    assert read_project_file(workspace, "MIMIR.md", 100) == (None, "", False)
    assert read_project_file(workspace, "", 100) == (None, "", False)

    (workspace / "MIMIR.md").write_text("   \n", encoding="utf-8")
    assert read_project_file(workspace, "MIMIR.md", 100)[0] is None

    (workspace / "MIMIR.md").write_text("规则一\n", encoding="utf-8")
    path, text, truncated = read_project_file(workspace, "MIMIR.md", 100)
    assert path == workspace / "MIMIR.md" and text == "规则一" and not truncated

    (workspace / "MIMIR.md").write_text("0123456789", encoding="utf-8")
    _, text, truncated = read_project_file(workspace, "MIMIR.md", 5)
    assert text == "01234" and truncated


def test_render():
    assert ProjectContext().render() == ""
    text = ProjectContext(project_file=Path("MIMIR.md"), project_text="规则", project_truncated=True,
                          git="当前分支：main").render()
    assert text.startswith("<project-context>") and text.endswith("</project-context>")
    assert "MIMIR.md" in text and "规则" in text and "已截断" in text
    assert "## Git 状态" in text and "main" in text


def test_with_update():
    base = ProjectContext(project_file=Path("MIMIR.md"), project_text="旧")
    assert base.with_update(None) is base
    new = base.with_update({"file": "MIMIR.md", "text": "新", "truncated": False})
    assert new.project_text == "新"
    deleted = base.with_update({"file": "MIMIR.md", "text": ""})
    assert deleted.project_file is None and "项目说明" not in deleted.render()


def test_render_update():
    assert "已被删除" in render_update({"file": "MIMIR.md", "text": ""})
    text = render_update({"file": "MIMIR.md", "text": "新规则"})
    assert text.startswith("<project-update>") and "新规则" in text


def test_inject_context_first_human_only():
    history = [HumanMessage("a"), AIMessage("b"), HumanMessage("c")]
    out = inject_context(history, "CTX")
    assert out[0].content == "CTX\n\na"
    assert out[1] is history[1] and out[2] is history[2]
    assert history[0].content == "a"  # 原消息不变
    assert inject_context(history, "") == history
    assert inject_context([AIMessage("x")], "CTX")[0].content == "CTX"


def test_inject_context_list_content():
    out = inject_context([HumanMessage([{"type": "text", "text": "hi"}])], "CTX")
    assert out[0].content[0] == {"type": "text", "text": "CTX"}
    assert out[0].content[1] == {"type": "text", "text": "hi"}


def test_inject_updates_strips_record_field():
    upd = {"file": "MIMIR.md", "text": "新规则"}
    m = HumanMessage("问题", additional_kwargs={PROJECT_UPDATE_KEY: upd, "other": 1})
    out = inject_updates([m, AIMessage("x")])
    assert out[0].content.startswith("<project-update>") and out[0].content.endswith("问题")
    assert out[0].additional_kwargs == {"other": 1}
    assert PROJECT_UPDATE_KEY in m.additional_kwargs  # 原消息不变
    assert m.content == "问题"


def test_latest_update():
    msgs = [HumanMessage("1", additional_kwargs={PROJECT_UPDATE_KEY: {"text": "v1"}}),
            AIMessage("x", additional_kwargs={PROJECT_UPDATE_KEY: {"text": "ignored"}}),
            HumanMessage("2", additional_kwargs={PROJECT_UPDATE_KEY: {"text": "v2"}}),
            HumanMessage("3")]
    assert latest_update(msgs) == {"text": "v2"}
    assert latest_update([HumanMessage("x")]) is None


def test_git_status_not_a_repo(workspace, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(workspace.parent))
    assert collect_git_status(workspace) == ""


@pytest.mark.skipif(shutil.which("git") is None, reason="未安装 git")
def test_git_status_in_repo(workspace, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(workspace.parent))

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                        "-c", "init.defaultBranch=main", *args], cwd=workspace, check=True, capture_output=True)

    git("init")
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-m", "第一次提交")

    text = collect_git_status(workspace)
    assert "当前分支：main" in text
    assert "（干净）" in text
    assert "第一次提交" in text

    (workspace / "新文件.txt").write_text("b", encoding="utf-8")
    assert "?? 新文件.txt" in collect_git_status(workspace)
