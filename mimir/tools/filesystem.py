"""文件工具：所有路径都被限制在工作目录（workspace）内。"""
from __future__ import annotations

import os
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from ..config import ToolsConfig

MAX_FILE_BYTES = 5 * 1024 * 1024
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".mypy_cache", ".pytest_cache", ".vscode"}


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _read_text(p: Path) -> tuple[str, str]:
    """读取文本文件，返回 (内容, 编码)。依次尝试 utf-8 / gbk。

    只有文件确实以 BOM 开头时才返回 utf-8-sig：编码会原样用于写回，utf-8-sig 写入时会在开头加 BOM。
    """
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"文件过大（{_human_size(size)}），超过 {_human_size(MAX_FILE_BYTES)} 限制")
    data = p.read_bytes()
    if b"\x00" in data[:8192]:
        raise ValueError("疑似二进制文件，无法按文本读取")
    for enc in ("utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8", "gbk"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...（输出过长，已截断，共 {len(text)} 字符）"


def build_filesystem_tools(cfg: ToolsConfig) -> list[BaseTool]:
    root = cfg.workspace.resolve()
    limit = cfg.max_output_chars

    def resolve(path: str) -> Path:
        p = Path(path or ".").expanduser()
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        if not p.is_relative_to(root):
            raise PermissionError(f"路径 {path} 超出工作目录 {root}，拒绝访问")
        return p

    def rel(p: Path) -> str:
        return p.relative_to(root).as_posix() or "."

    @tool
    def list_directory(
        path: Annotated[str, Field(description="目录路径（相对工作目录），默认为工作目录本身")] = ".",
        show_hidden: Annotated[bool, Field(description="是否显示以 . 开头的隐藏文件")] = False,
    ) -> str:
        """列出目录下的文件和子目录（含大小、修改时间）。"""
        d = resolve(path)
        if not d.is_dir():
            return f"不是目录或不存在：{path}"
        entries = [e for e in d.iterdir() if show_hidden or not e.name.startswith(".")]
        entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = [f"目录 {rel(d)}（共 {len(entries)} 项）"]
        for e in entries[:300]:
            try:
                st = e.stat()
                mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                lines.append(f"[?] {e.name}")
                continue
            if e.is_dir():
                lines.append(f"[D] {e.name}/  {mtime}")
            else:
                lines.append(f"[F] {e.name}  {_human_size(st.st_size)}  {mtime}")
        if len(entries) > 300:
            lines.append("...（仅显示前 300 项）")
        return _truncate("\n".join(lines), limit)

    @tool
    def read_file(
        path: Annotated[str, Field(description="文件路径（相对工作目录）")],
        start_line: Annotated[int, Field(description="起始行号（从 1 开始）")] = 1,
        max_lines: Annotated[int, Field(description="最多读取的行数")] = 300,
    ) -> str:
        """读取文本文件内容，输出带行号。大文件请配合 start_line 分段读取。"""
        p = resolve(path)
        if not p.is_file():
            return f"文件不存在：{path}"
        text, enc = _read_text(p)
        lines = text.splitlines()
        start = max(1, start_line)
        chunk = lines[start - 1:start - 1 + max(1, max_lines)]
        body = "\n".join(f"{i:>5} | {line}" for i, line in enumerate(chunk, start))
        end = start + len(chunk) - 1
        header = f"文件 {rel(p)}（{enc}，共 {len(lines)} 行，显示 {start}-{end} 行）"
        if end < len(lines):
            header += f"，后面还有 {len(lines) - end} 行"
        return _truncate(header + "\n" + body, limit)

    @tool
    def write_file(
        path: Annotated[str, Field(description="文件路径（相对工作目录），不存在的父目录会自动创建")],
        content: Annotated[str, Field(description="要写入的完整内容")],
        append: Annotated[bool, Field(description="True 追加到末尾；False 覆盖整个文件")] = False,
    ) -> str:
        """创建文件或写入内容（默认覆盖）。修改已有文件的局部内容请优先使用 edit_file。"""
        p = resolve(path)
        if p.is_dir():
            return f"目标是目录，无法写入：{path}"
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if append else "w", encoding="utf-8", newline="") as f:
            f.write(content)
        action = "追加" if append else ("覆盖" if existed else "创建")
        return f"已{action}文件 {rel(p)}（写入 {len(content)} 字符）"

    @tool
    def edit_file(
        path: Annotated[str, Field(description="文件路径（相对工作目录）")],
        old_text: Annotated[str, Field(description="要被替换的原文本，必须与文件内容完全一致（含缩进）")],
        new_text: Annotated[str, Field(description="替换后的新文本")],
        replace_all: Annotated[bool, Field(description="是否替换所有匹配；False 时要求原文本唯一")] = False,
    ) -> str:
        """精确替换文件中的一段文本。使用前请先 read_file 确认原文。"""
        p = resolve(path)
        if not p.is_file():
            return f"文件不存在：{path}"
        if not old_text:
            return "old_text 不能为空"
        text, enc = _read_text(p)
        old, new = old_text, new_text
        count = text.count(old)
        if count == 0 and "\r\n" in text:  # 兼容 Windows 换行
            old, new = old.replace("\r\n", "\n").replace("\n", "\r\n"), new.replace("\r\n", "\n").replace("\n", "\r\n")
            count = text.count(old)
        if count == 0:
            return "未找到 old_text，请先 read_file 确认原文（注意空格与缩进）"
        if count > 1 and not replace_all:
            return f"old_text 出现了 {count} 次，请提供更多上下文使其唯一，或设置 replace_all=true"
        text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        with p.open("w", encoding=enc, newline="") as f:
            f.write(text)
        return f"已修改 {rel(p)}（替换 {count if replace_all else 1} 处）"

    @tool
    def search_files(
        pattern: Annotated[str, Field(description="文件名通配符，如 *.py、*.md、src/*.txt；默认匹配所有文件")] = "*",
        keyword: Annotated[str, Field(description="可选：在文件内容中搜索的关键字（不区分大小写）")] = "",
        max_results: Annotated[int, Field(description="最多返回的结果数")] = 50,
    ) -> str:
        """在工作目录中按文件名查找文件，或按关键字搜索文件内容（返回 路径:行号: 内容）。"""
        results: list[str] = []
        kw = keyword.lower()
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                fp = Path(dirpath) / name
                r = rel(fp)
                if not (fnmatch(name, pattern) or fnmatch(r, pattern)
                        or (pattern.startswith("**/") and fnmatch(r, pattern[3:]))):
                    continue
                if not kw:
                    results.append(r)
                else:
                    try:
                        text, _ = _read_text(fp)
                    except (ValueError, OSError):
                        continue
                    for i, line in enumerate(text.splitlines(), 1):
                        if kw in line.lower():
                            results.append(f"{r}:{i}: {line.strip()[:200]}")
                            if len(results) >= max_results:
                                break
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
        if not results:
            return "没有找到匹配的结果"
        suffix = f"\n...（已达上限 {max_results} 条）" if len(results) >= max_results else ""
        return _truncate("\n".join(results) + suffix, limit)

    return [list_directory, read_file, write_file, edit_file, search_files]
