"""文件工具：路径限制、读写、编辑、搜索。"""
from __future__ import annotations

import pytest

from mimir.tools.filesystem import build_filesystem_tools


@pytest.fixture
def fs(tools_config):
    return {t.name: t for t in build_filesystem_tools(tools_config)}


def call(fs, name, **args):
    return fs[name].invoke(args)


# ---------- 路径限制 ----------

@pytest.mark.parametrize("name, extra", [
    ("read_file", {}),
    ("write_file", {"content": "x"}),
    ("edit_file", {"old_text": "a", "new_text": "b"}),
])
def test_relative_escape_rejected(fs, name, extra):
    with pytest.raises(PermissionError):
        call(fs, name, path="../outside.txt", **extra)


def test_absolute_outside_rejected(fs, tmp_path):
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(PermissionError):
        call(fs, "read_file", path=str(tmp_path / "outside.txt"))


def test_list_directory_escape_rejected(fs):
    with pytest.raises(PermissionError):
        call(fs, "list_directory", path="..")


# ---------- read_file ----------

def test_read_with_line_numbers(fs, workspace):
    (workspace / "a.txt").write_text("a\nb\nc\n", encoding="utf-8")
    out = call(fs, "read_file", path="a.txt", start_line=2, max_lines=1)
    assert "    2 | b" in out
    assert "a" not in out.split("\n", 1)[1]
    assert "后面还有 1 行" in out


def test_read_gbk(fs, workspace):
    (workspace / "g.txt").write_bytes("中文内容".encode("gbk"))
    out = call(fs, "read_file", path="g.txt")
    assert "中文内容" in out and "gbk" in out


def test_read_missing(fs):
    assert "文件不存在" in call(fs, "read_file", path="nope.txt")


def test_read_binary_rejected(fs, workspace):
    (workspace / "b.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="二进制"):
        call(fs, "read_file", path="b.bin")


def test_read_too_large_rejected(fs, workspace, monkeypatch):
    monkeypatch.setattr("mimir.tools.filesystem.MAX_FILE_BYTES", 10)
    (workspace / "big.txt").write_text("x" * 20, encoding="utf-8")
    with pytest.raises(ValueError, match="过大"):
        call(fs, "read_file", path="big.txt")


# ---------- write_file ----------

def test_write_create_overwrite_append(fs, workspace):
    assert "已创建" in call(fs, "write_file", path="sub/dir/f.txt", content="one")
    f = workspace / "sub" / "dir" / "f.txt"
    assert f.read_text(encoding="utf-8") == "one"
    assert "已覆盖" in call(fs, "write_file", path="sub/dir/f.txt", content="two")
    assert "已追加" in call(fs, "write_file", path="sub/dir/f.txt", content="+3", append=True)
    assert f.read_text(encoding="utf-8") == "two+3"


def test_write_to_directory_rejected(fs, workspace):
    (workspace / "d").mkdir()
    assert "目标是目录" in call(fs, "write_file", path="d", content="x")


# ---------- edit_file ----------

def test_edit_unique(fs, workspace):
    (workspace / "e.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    assert "替换 1 处" in call(fs, "edit_file", path="e.py", old_text="y = 2", new_text="y = 3")
    assert (workspace / "e.py").read_text(encoding="utf-8") == "x = 1\ny = 3\n"


def test_edit_ambiguous_and_replace_all(fs, workspace):
    (workspace / "e.py").write_text("a\na\n", encoding="utf-8")
    assert "出现了 2 次" in call(fs, "edit_file", path="e.py", old_text="a", new_text="b")
    assert (workspace / "e.py").read_text(encoding="utf-8") == "a\na\n"
    assert "替换 2 处" in call(fs, "edit_file", path="e.py", old_text="a", new_text="b", replace_all=True)
    assert (workspace / "e.py").read_text(encoding="utf-8") == "b\nb\n"


def test_edit_not_found_and_empty(fs, workspace):
    (workspace / "e.py").write_text("a\n", encoding="utf-8")
    assert "未找到" in call(fs, "edit_file", path="e.py", old_text="zzz", new_text="b")
    assert "不能为空" in call(fs, "edit_file", path="e.py", old_text="", new_text="b")
    assert "文件不存在" in call(fs, "edit_file", path="nope.py", old_text="a", new_text="b")


def test_edit_crlf_compat(fs, workspace):
    (workspace / "w.txt").write_bytes(b"a\r\nb\r\nc\r\n")
    assert "已修改" in call(fs, "edit_file", path="w.txt", old_text="a\nb", new_text="x\ny")
    assert (workspace / "w.txt").read_bytes() == b"x\r\ny\r\nc\r\n"


def test_edit_does_not_add_bom(fs, workspace):
    (workspace / "u.txt").write_bytes("中文 a\n".encode())
    assert "utf-8，" in call(fs, "read_file", path="u.txt")
    call(fs, "edit_file", path="u.txt", old_text="a", new_text="b")
    assert (workspace / "u.txt").read_bytes() == "中文 b\n".encode()


def test_edit_keeps_existing_bom(fs, workspace):
    (workspace / "bom.txt").write_bytes(b"\xef\xbb\xbfa\n")
    call(fs, "edit_file", path="bom.txt", old_text="a", new_text="b")
    assert (workspace / "bom.txt").read_bytes() == b"\xef\xbb\xbfb\n"


def test_edit_keeps_gbk(fs, workspace):
    (workspace / "g.txt").write_bytes("你好世界".encode("gbk"))
    call(fs, "edit_file", path="g.txt", old_text="世界", new_text="朋友")
    assert (workspace / "g.txt").read_bytes().decode("gbk") == "你好朋友"


# ---------- search_files / list_directory ----------

@pytest.fixture
def tree(workspace):
    files = {"a.py": "hello world\n", "b.md": "hello md\n", "sub/c.py": "nothing\nHELLO again\n",
             ".git/x.py": "hello\n", "node_modules/y.py": "hello\n"}
    for rel, text in files.items():
        p = workspace / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return workspace


def test_search_by_pattern(fs, tree):
    out = call(fs, "search_files", pattern="*.py")
    assert "a.py" in out and "sub/c.py" in out
    assert ".git" not in out and "node_modules" not in out
    assert "b.md" not in out


def test_search_by_keyword(fs, tree):
    out = call(fs, "search_files", keyword="hello")
    assert "a.py:1: hello world" in out
    assert "sub/c.py:2: HELLO again" in out  # 不区分大小写
    assert ".git" not in out


def test_search_limit_and_empty(fs, tree):
    assert "已达上限 1 条" in call(fs, "search_files", keyword="hello", max_results=1)
    assert call(fs, "search_files", keyword="不存在的词") == "没有找到匹配的结果"


def test_list_directory(fs, tree):
    out = call(fs, "list_directory")
    assert "[D] sub/" in out and "[F] a.py" in out
    assert ".git" not in out
    assert ".git" in call(fs, "list_directory", show_hidden=True)
    assert "不是目录" in call(fs, "list_directory", path="a.py")
