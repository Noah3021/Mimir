"""首次配置与环境检查：mimir init（交互向导 / --import 导入旧目录）、mimir doctor。

用户配置只写和默认配置（mimir/default_config.yaml）不同的部分，通常只有 models 和 agent.default_model；
API Key 写在同目录的 .env 里，配置中用 ${MIMIR_<名称>_API_KEY} 引用，配置文件可以放心给别人看。
"""
from __future__ import annotations

import dataclasses
import getpass
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from rich.console import Console
from rich.markup import escape

from . import __version__
from .config import (
    DEFAULT_CONFIG_PATH,
    HOME_ENV,
    ConfigError,
    ModelConfig,
    _interpolate,
    _parse_model,
    deep_merge,
    load_settings,
    mimir_home,
    read_yaml,
    user_config_path,
    user_env_path,
)
from .providers import CUSTOM_LABEL, CUSTOM_PROTOCOLS, PRESETS, env_key

CONFIG_HEADER = """\
# Mimir 用户配置（由 mimir init 生成，可以手动修改）
# 这里只写和默认配置不同的部分，其余设置见安装包中的 mimir/default_config.yaml，想改哪一项就在这里写哪一项。
# API Key 保存在同目录的 .env 中，这里用 ${环境变量名} 引用。
"""

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")

# 测试连接：成功时正常返回，失败时抛出异常
Tester = Callable[[ModelConfig], None]


# ---------------- 文件读写 ----------------

def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_env(path: Path, updates: dict[str, str]) -> None:
    """更新 .env：已有的同名变量就地替换，其余行（含注释）保持不变，新变量追加到末尾。

    同时写入当前进程的环境变量，使本次运行立即生效。非 Windows 系统上把文件权限设为仅本人可读写。
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    remaining = dict(updates)
    out = []
    for line in lines:
        stripped = line.strip()
        key = None
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip().removeprefix("export ").strip()
        if key in remaining:
            out.append(f"{key}={_quote(remaining.pop(key))}")
        else:
            out.append(line)
    out += [f"{k}={_quote(v)}" for k, v in remaining.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    if os.name != "nt":
        os.chmod(path, 0o600)
    os.environ.update(updates)


def save_user_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    path.write_text(f"{CONFIG_HEADER}\n{body}", encoding="utf-8")


def _ordered(raw: dict[str, Any]) -> dict[str, Any]:
    """agent、models 放在最前面，便于阅读。"""
    head = {k: raw[k] for k in ("agent", "models") if raw.get(k)}
    return {**head, **{k: v for k, v in raw.items() if k not in head}}


def deep_diff(value: Any, base: Any) -> Any:
    """value 中与 base 不同的部分（字典逐键比较）；完全相同时返回 None。"""
    if isinstance(value, dict) and isinstance(base, dict):
        out = {}
        for k, v in value.items():
            if k not in base:
                out[k] = v
                continue
            d = deep_diff(v, base[k])
            if d is not None:
                out[k] = d
        return out or None
    return None if value == base else value


def _flat_keys(d: dict[str, Any], prefix: str = "") -> list[str]:
    keys = []
    for k, v in d.items():
        path = f"{prefix}{k}"
        keys += _flat_keys(v, f"{path}.") if isinstance(v, dict) and v else [path]
    return keys


def _short_error(e: BaseException) -> str:
    text = str(e).strip().splitlines()[0] if str(e).strip() else ""
    text = f"{type(e).__name__}: {text}" if text else type(e).__name__
    return text if len(text) <= 300 else text[:300] + "…"


def ping_model(mc: ModelConfig) -> None:
    """发一个最小请求测试模型能否使用：关闭推理、不重试、超时最多 30 秒。"""
    from langchain_core.messages import HumanMessage

    from .llm import create_llm

    quick = dataclasses.replace(mc, timeout=min(mc.timeout, 30.0), max_retries=0)
    create_llm(quick, streaming=False, reasoning="off").invoke([HumanMessage("ping，回复 OK 即可")])


# ---------------- 交互 ----------------

@dataclass
class Prompter:
    """终端问答。测试时可传入 ask_fn / secret_fn 代替真实输入。"""

    console: Console
    ask_fn: Callable[[str], str] | None = None
    secret_fn: Callable[[str], str] | None = None

    def ask(self, prompt: str, default: str = "") -> str:
        text = f"{prompt} [{default}]：" if default else f"{prompt}："
        raw = self.ask_fn(text) if self.ask_fn else self.console.input(escape(text))
        return raw.strip() or default

    def secret(self, prompt: str) -> str:
        text = f"{prompt}："
        raw = self.secret_fn(text) if self.secret_fn else getpass.getpass(text)
        return raw.strip()

    def choose(self, prompt: str, n: int) -> int:
        while True:
            v = self.ask(f"{prompt} [1-{n}]") if n > 1 else "1"
            if v.isdigit() and 1 <= int(v) <= n:
                return int(v)
            self.console.print(f"[yellow]请输入 1-{n} 之间的数字[/]")

    def confirm(self, prompt: str, default: bool = False) -> bool:
        v = self.ask(f"{prompt}（{'Y/n' if default else 'y/N'}）").lower()
        return default if not v else v in ("y", "yes", "是")


@dataclass
class ModelSpec:
    name: str                 # 配置名称（models 下的键）
    entry: dict[str, Any]     # 写入配置的内容，api_key 为 ${环境变量} 引用
    api_key: str              # 真实的 Key，写入 .env


def _ask_name(p: Prompter, models: dict[str, Any], default: str) -> str:
    suggestion, n = default, 2
    while suggestion in models:
        suggestion, n = f"{default}-{n}", n + 1
    while True:
        name = p.ask("配置名称（用 /model 切换时输入的名字）", suggestion)
        if not _NAME_RE.fullmatch(name):
            p.console.print("[yellow]只能包含英文字母、数字、- 和 _[/]")
            continue
        if name in models and not p.confirm(f"{name} 已存在，覆盖它吗？"):
            continue
        return name


def _ask_model(p: Prompter, models: dict[str, Any]) -> ModelSpec:
    console = p.console
    console.print("\n选择模型供应商：")
    options = [pr.label for pr in PRESETS] + [CUSTOM_LABEL]
    for i, label in enumerate(options, 1):
        console.print(f"  {i}) {escape(label)}")
    idx = p.choose("请输入序号", len(options))

    if idx <= len(PRESETS):
        pr = PRESETS[idx - 1]
        provider, base_url, default_model = pr.provider, pr.base_url or "", pr.model
        reasoning, extra, label, default_name = pr.reasoning, dict(pr.extra), pr.label, pr.key
        if pr.key_url:
            console.print(f"[dim]API Key 在这里获取：{pr.key_url}[/]")
    else:
        label, default_model, reasoning, default_name = "自定义服务", "", "off", "custom"
        base_url = ""
        while not base_url:
            base_url = p.ask("Base URL（例如 https://dashscope.aliyuncs.com/compatible-mode/v1）")
        console.print("接口类型：")
        for i, (_, desc, _) in enumerate(CUSTOM_PROTOCOLS, 1):
            console.print(f"  {i}) {escape(desc)}")
        provider, _, proto_extra = CUSTOM_PROTOCOLS[p.choose("请输入序号", len(CUSTOM_PROTOCOLS)) - 1]
        extra = dict(proto_extra)

    api_key = ""
    while not api_key:
        api_key = p.secret("API Key（输入时不显示）")
    model = ""
    while not model:
        model = p.ask("模型名", default_model)
    name = _ask_name(p, models, default_name)

    entry: dict[str, Any] = {"provider": provider, "description": label}
    if base_url:
        entry["base_url"] = base_url
    entry.update({"api_key": "${" + env_key(name) + "}", "model": model, "reasoning": reasoning, **extra})
    return ModelSpec(name, entry, api_key)


def run_init(console: Console, *, prompter: Prompter | None = None, tester: Tester = ping_model) -> int:
    """交互式添加一个模型，写入 MIMIR_HOME 下的 config.yaml 和 .env。可重复运行，已有的模型不受影响。"""
    p = prompter or Prompter(console)
    cfg_path, env_path = user_config_path(), user_env_path()
    try:
        raw = read_yaml(cfg_path) if cfg_path.is_file() else {}
        models = dict(raw.get("models") or {})
        console.print(f"[bold]配置 Mimir[/]  配置保存在 {escape(str(cfg_path.parent))}")
        if models:
            console.print(f"已有模型：{escape(', '.join(models))}。这次会新增一个，原有的不受影响。")

        while True:
            spec = _ask_model(p, models)
            console.print("正在测试连接…")
            start = time.perf_counter()
            try:
                tester(_parse_model(spec.name, {**spec.entry, "api_key": spec.api_key}))
            except Exception as e:
                console.print(f"[red]✗ 连接失败：[/]{escape(_short_error(e))}")
                action = p.ask("r = 重新填写，s = 仍然保存，q = 放弃", "r").lower()
                if action == "s":
                    break
                if action == "q":
                    console.print("已放弃，没有保存任何内容。")
                    return 1
                continue
            console.print(f"[green]✓ 可用[/]（{time.perf_counter() - start:.1f}s）")
            break

        models[spec.name] = spec.entry
        agent = dict(raw.get("agent") or {})
        current = agent.get("default_model")
        if current not in models or current == spec.name or p.confirm(f"把 {spec.name} 设为默认模型？", True):
            agent["default_model"] = spec.name
        raw.update(agent=agent, models=models)
        write_env(env_path, {env_key(spec.name): spec.api_key})
        save_user_config(cfg_path, _ordered(raw))
    except (KeyboardInterrupt, EOFError):
        console.print("\n已取消，没有保存任何内容。")
        return 1
    console.print(f"[green]已保存[/]：模型 {escape(spec.name)}（{escape(spec.entry['model'])}）"
                  f"，默认模型 {escape(str(agent['default_model']))}")
    console.print(f"[dim]配置 {escape(str(cfg_path))}；API Key {escape(str(env_path))}。"
                  "再运行 mimir init 可以添加更多模型。[/]")
    return 0


# ---------------- 导入旧版本的配置 ----------------

@dataclass
class ImportResult:
    models: list[str] = field(default_factory=list)     # 导入的模型
    skipped: list[str] = field(default_factory=list)    # 已存在同名模型，未导入
    no_key: list[str] = field(default_factory=list)     # 旧配置里没有找到 API Key
    overrides: list[str] = field(default_factory=list)  # 和默认配置不同、一并导入的其他设置
    data: str = "none"                                   # copied / exists / none


def import_legacy(src: Path) -> ImportResult:
    """从旧版本的目录（源码目录，含 config.yaml、.env、data/）导入。只复制，不修改或删除原文件。

    - 模型：按旧的 .env（目录下的 .env，其次上级目录的 .env）解析 ${...}，Key 写入新的 .env，其余写入配置；
    - 其他设置：只导入和默认配置不同的部分；已有的用户配置优先；
    - data/：新目录下还没有数据时整体复制。
    """
    src = src.expanduser().resolve()
    old_cfg = src / "config.yaml"
    if not old_cfg.is_file():
        raise ConfigError(f"{src} 下没有 config.yaml")
    old = read_yaml(old_cfg)

    values: dict[str, str] = {}
    for env_file in (src / ".env", src.parent / ".env"):  # 旧版本的查找顺序：先找到的优先
        if env_file.is_file():
            for k, v in dotenv_values(env_file).items():
                if v is not None:
                    values.setdefault(k, v)

    def lookup(name: str) -> str | None:
        return values.get(name) or os.getenv(name)

    cfg_path = user_config_path()
    raw = read_yaml(cfg_path) if cfg_path.is_file() else {}
    models = dict(raw.get("models") or {})
    result, env_updates = ImportResult(), {}
    for name, m in (old.get("models") or {}).items():
        name = str(name)
        if not isinstance(m, dict):
            continue
        if name in models:
            result.skipped.append(name)
            continue
        resolved = _interpolate(m, lookup)
        key = resolved.pop("api_key", None)
        entry = {k: v for k, v in resolved.items() if v not in (None, "")}
        if key:
            env_updates[env_key(name)] = str(key)
            entry["api_key"] = "${" + env_key(name) + "}"
        else:
            result.no_key.append(name)
        models[name] = entry
        result.models.append(name)

    overrides = deep_diff({k: v for k, v in old.items() if k != "models"}, read_yaml(DEFAULT_CONFIG_PATH)) or {}
    result.overrides = _flat_keys(overrides)
    raw = deep_merge(overrides, raw)  # 已有的用户配置优先
    agent = dict(raw.get("agent") or {})
    if agent.get("default_model") not in models and models:
        agent["default_model"] = next(iter(models))
    raw.update(agent=agent, models=models)

    if env_updates:
        write_env(user_env_path(), env_updates)
    save_user_config(cfg_path, _ordered(raw))

    src_data, dst_data = src / "data", mimir_home() / "data"
    if not src_data.is_dir():
        result.data = "none"
    elif dst_data.is_dir() and any(dst_data.iterdir()):
        result.data = "exists"
    else:
        shutil.copytree(src_data, dst_data, dirs_exist_ok=True)
        result.data = "copied"
    return result


def run_import(console: Console, src: Path) -> int:
    try:
        r = import_legacy(src)
    except ConfigError as e:
        console.print(f"[red]导入失败：[/]{escape(str(e))}")
        return 1
    console.print(f"[green]已导入到 {escape(str(mimir_home()))}[/]")
    console.print(f"  模型：{escape(', '.join(r.models)) or '无'}")
    if r.skipped:
        console.print(f"  [yellow]已存在同名模型，未导入：{escape(', '.join(r.skipped))}[/]")
    if r.no_key:
        console.print("  [yellow]没有找到 API Key（导入后不可用，可运行 mimir init 重新添加）："
                      f"{escape(', '.join(r.no_key))}[/]")
    if r.overrides:
        console.print(f"  其他和默认配置不同的设置：{escape(', '.join(r.overrides))}")
    data = {"copied": "已复制会话、对话记录和日志", "exists": "新目录下已有数据，未复制",
            "none": "原目录没有 data/"}[r.data]
    console.print(f"  数据：{data}")
    console.print("[dim]原目录中的文件没有改动，确认无误后可以自行删除。运行 mimir doctor 检查配置。[/]")
    return 0


# ---------------- 环境检查 ----------------

def run_doctor(console: Console, config_path: str | None = None, *, offline: bool = False,
               tester: Tester = ping_model) -> int:
    """检查 Python 版本、配置、各模型能否使用、可选工具。有可用模型时返回 0。"""
    problems = 0

    def line(status: str, label: str, detail: str) -> None:
        style = {"✓": "green", "✗": "red", "!": "yellow"}.get(status, "dim")
        console.print(f"  [{style}]{status}[/] {label}  {escape(detail)}")

    console.print(f"[bold]Mimir {__version__}[/]  环境检查")
    v = sys.version_info
    if v >= (3, 11):
        line("✓", "Python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        line("✗", "Python", f"{v.major}.{v.minor}.{v.micro}，需要 3.11 及以上")
        problems += 1

    home = mimir_home()
    line("·", "配置目录", f"{home}" + (f"（来自环境变量 {HOME_ENV}）" if os.environ.get(HOME_ENV) else ""))
    cfg = Path(config_path).expanduser() if config_path else user_config_path()
    if cfg.is_file():
        line("✓", "配置文件", str(cfg))
    else:
        line("✗", "配置文件", f"{cfg} 不存在，运行 mimir init 生成")
        problems += 1
    env = user_env_path()
    env_detail = str(env) if env.is_file() else f"{env} 不存在（也可以直接用环境变量提供 Key）"
    line("✓" if env.is_file() else "·", ".env", env_detail)

    settings = None
    if cfg.is_file():
        try:
            settings = load_settings(config_path)
        except ConfigError as e:
            line("✗", "配置内容", str(e))
            problems += 1

    if settings is not None:
        usable = 0
        for name, mc in settings.models.items():
            label = f"模型 {name}" + ("（默认）" if name == settings.agent.default_model else "")
            if mc.missing_fields():
                line("✗", label, f"缺少 {', '.join(mc.missing_fields())}")
                continue
            desc = f"{mc.provider} · {mc.model}" + (f" · {mc.base_url}" if mc.base_url else "")
            if offline:
                line("✓", label, f"{desc}（未测试连接）")
                usable += 1
                continue
            start = time.perf_counter()
            try:
                tester(mc)
            except Exception as e:
                line("✗", label, f"{desc}：{_short_error(e)}")
                continue
            line("✓", label, f"{desc}（{time.perf_counter() - start:.1f}s）")
            usable += 1
        if not usable:
            problems += 1
        line("·", "数据目录", str(settings.memory.db_path.parent))

    for tool, why in (("git", "注入 git 状态"), ("rg", "以后的代码搜索")):
        found = shutil.which(tool)
        line("✓" if found else "!", tool, found or f"未安装（可选，用于{why}）")

    console.print("[green]没有发现问题[/]" if not problems else f"[red]发现 {problems} 个问题，按上面的提示处理[/]")
    return 0 if not problems else 1
