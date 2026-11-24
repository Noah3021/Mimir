"""配置加载：YAML 配置文件 + .env 环境变量插值。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ${VAR} 或 ${VAR:默认值}（也兼容 ${VAR:-默认值}）
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-?([^}]*))?\}")
_SCALAR_PATTERN = re.compile(r"^(?:-?\d+(?:\.\d+)?|true|false|True|False)$")

_MODEL_KNOWN_KEYS = {"base_url", "api_key", "model", "temperature", "timeout", "max_retries", "description",
                     "provider", "reasoning", "max_tokens"}

PROVIDERS = ("openai", "deepseek", "anthropic")
REASONING_LEVELS = ("off", "low", "medium", "high", "xhigh")
_REASONING_ALIASES = {
    "none": "off", "false": "off", "0": "off", "disabled": "off", "disable": "off", "关": "off",
    "minimal": "low", "true": "high", "on": "high", "enabled": "high", "开": "high", "max": "xhigh",
}


class ConfigError(Exception):
    """配置错误。"""


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
class AgentConfig:
    name: str
    system_prompt: str
    default_model: str
    recursion_limit: int = 50
    max_history_messages: int = 60


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
    """依次加载 .env：当前目录 -> 项目目录 -> 项目上级目录。先加载的优先（不覆盖）。"""
    loaded = []
    seen = set()
    for p in (Path.cwd() / ".env", PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"):
        rp = p.resolve()
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        load_dotenv(rp, override=False)
        loaded.append(rp)
    return loaded


def _interpolate(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if not isinstance(value, str):
        return value

    whole = _ENV_PATTERN.fullmatch(value.strip()) is not None
    result = _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1)) or (m.group(2) or ""), value)
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
        model=str(raw.get("model") or "").strip(),
        base_url=(str(raw.get("base_url")).strip() or None) if raw.get("base_url") else None,
        api_key=(str(raw.get("api_key")).strip() or None) if raw.get("api_key") else None,
        temperature=_to_number(raw.get("temperature"), float, None, f"models.{name}.temperature"),
        timeout=_to_number(raw.get("timeout"), float, 120.0, f"models.{name}.timeout"),
        max_retries=_to_number(raw.get("max_retries"), int, 2, f"models.{name}.max_retries"),
        description=str(raw.get("description") or ""),
        extra={k: v for k, v in raw.items() if k not in _MODEL_KNOWN_KEYS},
    )


def load_settings(config_path: str | Path | None = None, *, workspace: str | Path | None = None) -> Settings:
    load_env_files()

    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"找不到配置文件：{path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"配置文件解析失败：{e}") from e
    data = _interpolate(raw)
    base_dir = path.resolve().parent

    def resolve_path(value: Any, default: str) -> Path:
        p = Path(str(value or default)).expanduser()
        return p if p.is_absolute() else (base_dir / p).resolve()

    # ---- models ----
    models_raw = data.get("models") or {}
    if not models_raw:
        raise ConfigError("配置中至少需要定义一个模型（models）")
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
        max_history_messages=_to_number(a.get("max_history_messages"), int, 60, "agent.max_history_messages"),
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
