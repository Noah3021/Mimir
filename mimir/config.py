"""配置加载：YAML 配置文件 + .env 环境变量插值。"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 默认配置随包发布（不含模型和密钥）；用户配置在 MIMIR_HOME（默认 ~/.mimir）下，启动时覆盖在默认配置上
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "default_config.yaml"
HOME_ENV = "MIMIR_HOME"

# ${VAR} 或 ${VAR:默认值}（也兼容 ${VAR:-默认值}）
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-?([^}]*))?\}")
_SCALAR_PATTERN = re.compile(r"^(?:-?\d+(?:\.\d+)?|true|false|True|False)$")

_MODEL_KNOWN_KEYS = {"base_url", "api_key", "model", "temperature", "timeout", "max_retries", "description",
                     "provider", "reasoning", "max_tokens", "context_window"}

PROVIDERS = ("openai", "deepseek", "anthropic")
REASONING_LEVELS = ("off", "low", "medium", "high", "xhigh")
_REASONING_ALIASES = {
    "none": "off", "false": "off", "0": "off", "disabled": "off", "disable": "off", "关": "off",
    "minimal": "low", "true": "high", "on": "high", "enabled": "high", "开": "high", "max": "xhigh",
}


class ConfigError(Exception):
    """配置错误。"""


class SetupRequired(ConfigError):
    """还没有用户配置（未指定 -c 且 MIMIR_HOME 下没有 config.yaml），需要先运行 mimir init。"""


def mimir_home() -> Path:
    """用户配置与数据目录：环境变量 MIMIR_HOME，未设置时为 ~/.mimir。"""
    value = os.environ.get(HOME_ENV, "").strip()
    return (Path(value).expanduser() if value else Path.home() / ".mimir").resolve()


def user_config_path() -> Path:
    return mimir_home() / "config.yaml"


def user_env_path() -> Path:
    return mimir_home() / ".env"


def normalize_reasoning(value: Any, field_name: str = "reasoning") -> str:
    """把各种写法统一为 off / low / medium / high / xhigh。"""
    if value is None or value == "":
        return "off"
    if isinstance(value, bool):
        return "high" if value else "off"
    if isinstance(value, dict):  # 兼容旧写法 reasoning: {effort: high, summary: auto}
        value = value.get("effort") or ("high" if value.get("enabled") else "off")
    v = str(value).strip().lower()
    v = _REASONING_ALIASES.get(v, v)
    if v not in REASONING_LEVELS:
        raise ConfigError(f"{field_name} 的值无效：{value!r}（可选：{' / '.join(REASONING_LEVELS)}）")
    return v


@dataclass
class ModelConfig:
    name: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    timeout: float = 120.0
    max_retries: int = 2
    description: str = ""
    provider: str = "openai"         # openai | deepseek | anthropic
    reasoning: str = "off"           # off | low | medium | high | xhigh
    max_tokens: int | None = None    # 最大输出 token（anthropic 不含思考预算）
    context_window: int | None = None  # 上下文窗口 token 数；None = 使用 agent.context.window
    extra: dict[str, Any] = field(default_factory=dict)  # 透传给模型类的其他参数

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.model:
            missing.append("model")
        if not self.api_key:
            missing.append("api_key")
        return missing

    @property
    def available(self) -> bool:
        return not self.missing_fields()


@dataclass
class ContextConfig:
    window: int = 128_000            # 默认上下文窗口（token），模型可用 context_window 覆盖
    high: float = 0.8                # 高水位：历史超过窗口的该比例时触发压缩
    low: float = 0.5                 # 低水位：压缩后的目标比例
    output_reserve: int = 16_000     # 为模型输出（含思考）预留的 token
    clear_keep_turns: int = 3        # 清理旧工具输出时，最近几轮不动；0 = 不清理
    clear_min_chars: int = 500       # 只清理超过该字符数的工具输出
    summary_chars: int = 2000        # 早期对话摘要的目标长度（字）


@dataclass
class ProjectConfig:
    file: str = "MIMIR.md"           # 工作目录下的项目说明文件；空字符串 = 不读取
    git_status: bool = True          # 会话开始时注入 git 分支、工作区状态和最近提交
    max_chars: int = 20_000          # 项目说明最多读取的字符数，超出截断


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    default_model: str
    recursion_limit: int = 50
    stream_retries: int = 5          # 连接中断时自动重发请求的最大次数
    context: ContextConfig = field(default_factory=ContextConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)


@dataclass
class ToolsConfig:
    workspace: Path
    disabled: list[str] = field(default_factory=list)
    require_approval: list[str] = field(default_factory=list)
    max_output_chars: int = 8000
    shell_enabled: bool = True
    shell_executable: str | None = None
    shell_timeout: int = 60
    blocked_patterns: list[str] = field(default_factory=list)


@dataclass
class MemoryConfig:
    db_path: Path
    transcripts_enabled: bool = True
    transcripts_dir: Path = Path("data/transcripts")


@dataclass
class CLIConfig:
    stream: bool = True
    show_tool_calls: bool = True
    show_reasoning: bool = True
    show_usage: bool = True
    history_file: Path | None = None


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: Path | None = None


@dataclass
class Settings:
    config_path: Path
    agent: AgentConfig
    models: dict[str, ModelConfig]
    tools: ToolsConfig
    memory: MemoryConfig
    cli: CLIConfig
    logging: LoggingConfig


def load_env_files() -> list[Path]:
    """依次加载 .env：当前目录 -> MIMIR_HOME/.env。先加载的优先（不覆盖已有的环境变量）。"""
    loaded = []
    seen = set()
    for p in (Path.cwd() / ".env", user_env_path()):
        rp = p.resolve()
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        load_dotenv(rp, override=False)
        loaded.append(rp)
    return loaded


def _interpolate(value: Any, lookup: Callable[[str], str | None] = os.getenv) -> Any:
    """替换 ${VAR} / ${VAR:默认值}。lookup 默认读环境变量；导入旧配置时传入旧 .env 的取值。"""
    if isinstance(value, dict):
        return {k: _interpolate(v, lookup) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, lookup) for v in value]
    if not isinstance(value, str):
        return value

    whole = _ENV_PATTERN.fullmatch(value.strip()) is not None
    result = _ENV_PATTERN.sub(lambda m: lookup(m.group(1)) or (m.group(2) or ""), value)
    # 整个值就是一个占位符时，尝试还原数字/布尔类型
    if whole and _SCALAR_PATTERN.match(result.strip()):
        return yaml.safe_load(result.strip())
    return result


def _to_number(value: Any, cast, default, field_name: str):
    if value is None or value == "":
        return default
    try:
        return cast(value)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"配置项 {field_name} 的值无效：{value!r}") from e


def _parse_model(name: str, raw: Any) -> ModelConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"模型 '{name}' 的配置必须是键值对")
    provider = str(raw.get("provider") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"模型 '{name}' 的 provider 无效：{provider}（可选：{', '.join(PROVIDERS)}）")
    return ModelConfig(
        name=name,
        provider=provider,
        reasoning=normalize_reasoning(raw.get("reasoning"), f"models.{name}.reasoning"),
        max_tokens=_to_number(raw.get("max_tokens"), int, None, f"models.{name}.max_tokens"),
        context_window=_to_number(raw.get("context_window"), int, None, f"models.{name}.context_window"),
        model=str(raw.get("model") or "").strip(),
        base_url=(str(raw.get("base_url")).strip() or None) if raw.get("base_url") else None,
        api_key=(str(raw.get("api_key")).strip() or None) if raw.get("api_key") else None,
        temperature=_to_number(raw.get("temperature"), float, None, f"models.{name}.temperature"),
        timeout=_to_number(raw.get("timeout"), float, 120.0, f"models.{name}.timeout"),
        max_retries=_to_number(raw.get("max_retries"), int, 2, f"models.{name}.max_retries"),
        description=str(raw.get("description") or ""),
        extra={k: v for k, v in raw.items() if k not in _MODEL_KNOWN_KEYS},
    )


def _parse_context(raw: Any) -> ContextConfig:
    if not isinstance(raw, dict):
        raise ConfigError("agent.context 必须是键值对")
    d = ContextConfig()
    ctx = ContextConfig(
        window=_to_number(raw.get("window"), int, d.window, "agent.context.window"),
        high=_to_number(raw.get("high"), float, d.high, "agent.context.high"),
        low=_to_number(raw.get("low"), float, d.low, "agent.context.low"),
        output_reserve=_to_number(raw.get("output_reserve"), int, d.output_reserve, "agent.context.output_reserve"),
        clear_keep_turns=max(0, _to_number(raw.get("clear_keep_turns"), int, d.clear_keep_turns,
                                           "agent.context.clear_keep_turns")),
        clear_min_chars=max(0, _to_number(raw.get("clear_min_chars"), int, d.clear_min_chars,
                                          "agent.context.clear_min_chars")),
        summary_chars=max(200, _to_number(raw.get("summary_chars"), int, d.summary_chars,
                                          "agent.context.summary_chars")),
    )
    if ctx.window <= 0:
        raise ConfigError("agent.context.window 必须大于 0")
    if not 0 < ctx.low < ctx.high <= 1:
        raise ConfigError(f"agent.context 需满足 0 < low < high <= 1（当前 low={ctx.low}, high={ctx.high}）")
    return ctx


def _parse_project(raw: Any) -> ProjectConfig:
    if not isinstance(raw, dict):
        raise ConfigError("agent.project 必须是键值对")
    d = ProjectConfig()
    git = raw.get("git_status", d.git_status)
    if isinstance(git, str):
        git = git.strip().lower() in ("true", "yes", "on", "1")
    return ProjectConfig(
        file=str(raw.get("file", d.file) or "").strip(),
        git_status=bool(git),
        max_chars=max(1000, _to_number(raw.get("max_chars"), int, d.max_chars, "agent.project.max_chars")),
    )


def read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件；空文件返回 {}。"""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"配置文件 {path} 解析失败：{e}") from e
    except OSError as e:
        raise ConfigError(f"无法读取配置文件 {path}：{e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件 {path} 的顶层必须是键值对")
    return raw


def deep_merge(base: Any, override: Any) -> Any:
    """用户配置覆盖默认配置：两边都是字典时逐键合并，否则（标量、列表）整体替换。不修改传入的对象。"""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for k, v in override.items():
            merged[k] = deep_merge(base[k], v) if k in base else v
        return merged
    return override


def load_settings(config_path: str | Path | None = None, *, workspace: str | Path | None = None) -> Settings:
    """读取默认配置，再用用户配置覆盖。

    用户配置：-c 指定的文件；未指定时为 MIMIR_HOME/config.yaml，不存在则抛出 SetupRequired。
    相对路径（数据库、对话记录、日志等）相对用户配置所在目录。
    """
    load_env_files()

    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"找不到配置文件：{path}")
    else:
        path = user_config_path()
        if not path.is_file():
            raise SetupRequired(f"还没有配置模型（{path} 不存在），请先运行 mimir init")
    data = _interpolate(deep_merge(read_yaml(DEFAULT_CONFIG_PATH), read_yaml(path)))
    base_dir = path.resolve().parent

    def resolve_path(value: Any, default: str) -> Path:
        p = Path(str(value or default)).expanduser()
        return p if p.is_absolute() else (base_dir / p).resolve()

    # ---- models ----
    models_raw = data.get("models") or {}
    if not models_raw:
        raise ConfigError(f"{path} 中还没有配置模型（models），请运行 mimir init 添加")
    models = {str(name): _parse_model(str(name), m) for name, m in models_raw.items()}

    # ---- agent ----
    a = data.get("agent") or {}
    default_model = str(a.get("default_model") or next(iter(models)))
    if default_model not in models:
        raise ConfigError(f"agent.default_model '{default_model}' 不在 models 中：{', '.join(models)}")
    agent = AgentConfig(
        name=str(a.get("name") or "Mimir"),
        system_prompt=str(a.get("system_prompt") or "你是一个乐于助人的 AI 助手。"),
        default_model=default_model,
        recursion_limit=_to_number(a.get("recursion_limit"), int, 50, "agent.recursion_limit"),
        stream_retries=max(0, _to_number(a.get("stream_retries"), int, 5, "agent.stream_retries")),
        context=_parse_context(a.get("context") or {}),
        project=_parse_project(a.get("project") or {}),
    )

    # ---- tools ----
    t = data.get("tools") or {}
    sh = t.get("shell") or {}
    if workspace:  # 命令行参数：相对当前目录
        ws = Path(workspace).expanduser()
        ws = ws if ws.is_absolute() else Path.cwd() / ws
    elif t.get("workspace"):  # 配置文件：相对配置文件目录
        ws = resolve_path(t["workspace"], ".")
    else:
        ws = Path.cwd()
    ws = ws.resolve()
    if not ws.is_dir():
        raise ConfigError(f"工作目录不存在：{ws}")
    tools = ToolsConfig(
        workspace=ws,
        disabled=[str(x) for x in (t.get("disabled") or [])],
        require_approval=[str(x) for x in (t.get("require_approval") or [])],
        max_output_chars=_to_number(t.get("max_output_chars"), int, 8000, "tools.max_output_chars"),
        shell_enabled=bool(sh.get("enabled", True)),
        shell_executable=(str(sh["executable"]) if sh.get("executable") else None),
        shell_timeout=_to_number(sh.get("timeout"), int, 60, "tools.shell.timeout"),
        blocked_patterns=[str(x) for x in (sh.get("blocked_patterns") or [])],
    )

    # ---- memory / cli / logging ----
    m = data.get("memory") or {}
    tr = m.get("transcripts") or {}
    c = data.get("cli") or {}
    lg = data.get("logging") or {}
    return Settings(
        config_path=path.resolve(),
        agent=agent,
        models=models,
        tools=tools,
        memory=MemoryConfig(
            db_path=resolve_path(m.get("db_path"), "data/sessions.db"),
            transcripts_enabled=bool(tr.get("enabled", True)),
            transcripts_dir=resolve_path(tr.get("dir"), "data/transcripts"),
        ),
        cli=CLIConfig(
            stream=bool(c.get("stream", True)),
            show_tool_calls=bool(c.get("show_tool_calls", True)),
            show_reasoning=bool(c.get("show_reasoning", True)),
            show_usage=bool(c.get("show_usage", True)),
            history_file=resolve_path(c.get("history_file"), "data/input_history.txt"),
        ),
        logging=LoggingConfig(
            level=str(lg.get("level") or "INFO").upper(),
            file=resolve_path(lg.get("file"), "data/agent.log") if lg.get("file", "data/agent.log") else None,
        ),
    )
