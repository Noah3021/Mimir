"""CLI 的纯函数、输出渲染和命令行参数。用写入 StringIO 的 Console 断言输出。"""
from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.text import Text

from mimir.cli import StreamPrinter, _Tail, clip_lines, parse_args, short


def make_console(**kw) -> Console:
    kw.setdefault("width", 100)
    return Console(file=io.StringIO(), force_terminal=False, color_system=None, **kw)


def output(console: Console) -> str:
    return console.file.getvalue()


# ---------------- short / clip_lines ----------------
def test_short_collapses_whitespace():
    assert short("a\n  b\tc") == "a b c"


def test_short_truncates():
    assert short("x" * 200, limit=10) == "x" * 10 + "…"


def test_short_serializes_non_str():
    assert short({"路径": "a.txt", "n": 1}) == '{"路径": "a.txt", "n": 1}'


def test_clip_lines_keeps_short_text():
    assert clip_lines("a\nb", 5) == "a\nb"


def test_clip_lines_truncates():
    out = clip_lines("\n".join(str(i) for i in range(100)), 3)
    assert out.splitlines()[:3] == ["0", "1", "2"]
    assert "共 100 行，仅显示前 3 行" in out


# ---------------- _Tail ----------------
def test_tail_keeps_last_lines_within_terminal_height():
    console = make_console(height=10)
    console.print(_Tail(Text("\n".join(f"line{i}" for i in range(50)))))
    lines = output(console).splitlines()
    assert len(lines) == 8  # 终端高度 - 2
    assert lines[-1].strip() == "line49"
    assert lines[0].strip() == "line42"


def test_tail_short_content_unchanged():
    console = make_console(height=10)
    console.print(_Tail(Text("a\nb")))
    assert [s.strip() for s in output(console).splitlines()] == ["a", "b"]


# ---------------- StreamPrinter ----------------
def test_stream_printer_prints_text_once():
    console = make_console()
    p = StreamPrinter(console)
    p.wait()
    for piece in ["你好", "，", "世界"]:
        p.feed("text", piece)
    p.end()
    assert output(console).count("你好，世界") == 1
    assert p.live is None and p.buf == ""


def test_stream_printer_reasoning_then_text():
    console = make_console()
    p = StreamPrinter(console)
    p.feed("reasoning", "先想一想")
    p.feed("text", "答案")
    p.end()
    out = output(console)
    assert "思考过程" in out and "先想一想" in out
    assert out.index("先想一想") < out.index("答案")


def test_stream_printer_hides_reasoning():
    console = make_console()
    p = StreamPrinter(console, show_reasoning=False)
    p.feed("reasoning", "不该显示")
    p.feed("text", "答案")
    p.end()
    assert "不该显示" not in output(console)
    assert "答案" in output(console)


def test_stream_printer_discard_prints_nothing():
    console = make_console()
    p = StreamPrinter(console)
    p.feed("text", "半截输出")
    p.discard()
    p.end()
    assert "半截输出" not in output(console)


def test_stream_printer_ignores_empty_and_whitespace():
    console = make_console()
    p = StreamPrinter(console)
    p.feed("text", "")
    p.feed("text", "   ")
    p.end()
    assert output(console).strip() == ""


# ---------------- parse_args ----------------
def test_parse_args_defaults():
    a = parse_args([])
    assert (a.config, a.model, a.think, a.workspace, a.resume, a.cont, a.prompt, a.yolo) == (
        None, None, None, None, None, False, None, False)


def test_parse_args_all_options():
    a = parse_args(["-c", "c.yaml", "-m", "gpt", "-t", "high", "-w", "ws", "-r", "abc", "-p", "hi", "--yolo"])
    assert (a.config, a.model, a.think, a.workspace, a.resume, a.prompt, a.yolo) == (
        "c.yaml", "gpt", "high", "ws", "abc", "hi", True)


@pytest.mark.parametrize("argv", [["-r", "abc", "--continue"], ["-t", "extreme"]])
def test_parse_args_rejects_invalid(argv, capsys):
    with pytest.raises(SystemExit) as e:
        parse_args(argv)
    assert e.value.code == 2
