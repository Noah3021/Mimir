# Mimir — 个人 CLI 智能体（LangChain + LangGraph）

一个运行在终端中的个人 AI 助手：支持多模型切换（OpenAI / DeepSeek / Claude）、本地文件读写、Shell 命令、基础工具（时间/计算器），会话通过 SQLite 持久化，可随时恢复；长对话按 token 自动压缩，并能读取项目说明文件 `MIMIR.md` 和 git 状态作为上下文。

## 快速开始

```bash
cd Mimir
conda activate agent   # 或任意 Python 3.11+ 环境（LangChain 1.8 起不再支持 3.10）
pip install -r requirements.txt
# 配置环境变量：复制 .env.example 为 .env 后填写（也可复用上级目录的 .env）
python main.py        # 或：python -m mimir
```

常用启动参数：

| 参数 | 说明 |
| --- | --- |
| `-m gpt` | 指定模型（config.yaml 中 models 的键名） |
| `-t high` | 本次运行的推理强度（off / low / medium / high / xhigh） |
| `-w <目录>` | 文件/命令工具的工作目录（默认当前目录） |
| `--continue` | 继续最近一次会话 |
| `-r <ID>` | 恢复指定会话 |
| `-p "问题"` | 单次提问后退出 |
| `--yolo` | 跳过危险操作确认（慎用） |
| `-c <文件>` | 使用其他配置文件 |

## CLI 命令

| 命令 | 说明 |
| --- | --- |
| `/help` | 帮助 |
| `/new` | 新会话 |
| `/sessions` | 历史会话列表 |
| `/resume <序号或ID>` | 恢复会话 |
| `/delete <序号或ID>` | 删除会话 |
| `/history [条数]` | 当前会话记录（如 `/history 20` 只看最近 20 条，省略则显示全部） |
| `/model [名称]` | 查看/切换模型（保留上下文） |
| `/think [级别]` | 查看/设置推理强度（按模型分别记忆） |
| `/usage` | Token 用量与缓存命中率 |
| `/tools` | 可用工具 |
| `/approve on\|off` | 开关危险操作确认 |
| `/config` | 当前配置 |
| `/clear` | 清屏 |
| `/exit` 或 `退出` | 退出 |

输入时 `Esc+Enter` 换行，`Ctrl+C` 中断当前回答。

## 配置说明（config.yaml）

- `agent`：名称、默认模型、系统提示词（支持 `{date}` `{workspace}` 等占位符；每个会话只渲染一次，以便命中提示词缓存）、单轮最大步数、连接中断自动重试次数（`stream_retries`，默认 5）。
  - `agent.context`：上下文窗口管理，见下文“上下文管理”。
  - `agent.project`：项目说明文件和 git 状态，见下文“项目说明（MIMIR.md）”。
- `models`：任意多个模型，`provider` 可选 `openai` / `deepseek` / `anthropic`（含任意 OpenAI 兼容接口）；值可用 `${ENV}` / `${ENV:默认值}` 引用 .env；`context_window` 可单独设置窗口大小；额外字段（如 `use_responses_api`、`extra_body`）原样传给对应的模型类。
- `tools`：工作目录、禁用工具、需确认的工具、输出长度上限、Shell 超时与黑名单。
- `memory`：SQLite 数据库路径；`transcripts` 明文对话记录的开关与目录。
- `cli` / `logging`：流式输出、显示工具调用/推理过程、输入历史、日志文件。

## 上下文管理

按估算 token 管理发给模型的历史（`agent.context`）：历史超过窗口的 80%（`high`）时，先把最近 3 轮之前的大段工具输出换成占位文字，仍超过 50%（`low`）再把最早的几轮对话交给模型生成摘要，替代原始消息。压缩点只在越过高水位时前移，其余时间发送内容的开头不变，便于命中提示词缓存。完整历史仍保存在会话数据库和对话记录中，`/history` 可以看到全部消息；`/config` 显示当前的窗口大小和摘要情况。

## 项目说明（MIMIR.md）

在工作目录下放一个 `MIMIR.md`，写上项目约定（代码风格、常用命令、注意事项等），Mimir 会在每个会话开始时读取它，连同 git 状态（当前分支、改动文件、最近 5 次提交）一起告诉模型。

- 这些内容拼在发给模型的第一条用户消息前面，不写进系统提示词，也不保存到会话历史。
- 会话中途修改或删除 `MIMIR.md`，下一轮会自动附带最新内容，终端会提示“检测到 MIMIR.md 已修改”。
- git 状态只在会话开始时收集一次，之后不自动更新。
- 可在 `agent.project` 中改文件名（`file`，留空不读取）、关闭 git 状态（`git_status: false`）、限制读取长度（`max_chars`）。

## 对话记录

对话有两份存储：

- `data/sessions.db`：LangGraph SQLite checkpointer，用于 `/resume`、`--continue` 恢复对话（二进制，不便直接查看）。
- `data/transcripts/<工作目录名>/<会话ID>.jsonl`：明文记录，每行一条 JSON，可用任意编辑器打开。`/config` 中可看到当前会话的记录文件路径。

记录类型：`user`（输入）、`assistant`（回复正文、推理、工具调用、token 用量 `usage`、模型）、`tool`（工具结果）。每条都带 `timestamp`、`session_id`、`message_id`、`cwd`。恢复旧会话时，启用记录前的历史消息会在下一轮补写，并标记 `"backfilled": true`。

## 工具

| 工具 | 说明 | 默认需确认 |
| --- | --- | --- |
| `get_current_time` | 当前时间（可指定时区） | |
| `calculator` | 安全的数学表达式计算 | |
| `list_directory` | 列目录 | |
| `read_file` | 读文件（带行号、分段） | |
| `search_files` | 按文件名/内容搜索 | |
| `write_file` | 写文件 | ✓ |
| `edit_file` | 精确替换文本 | ✓ |
| `run_shell` | 执行命令 | ✓ |

文件工具只能访问工作目录内的路径。确认时可输入 `y` 允许、`n` 拒绝、`a` 本次运行始终允许，或直接输入意见（拒绝并反馈给助手）。

## 项目结构

```
Mimir/
├── main.py                  # 启动入口
├── config.yaml              # 智能体配置
├── .env.example
├── requirements.txt
└── mimir/
    ├── __init__.py          # 包信息（版本号）
    ├── __main__.py          # 支持 python -m mimir
    ├── config.py            # 配置加载 + 环境变量插值
    ├── llm.py               # 模型创建
    ├── usage.py             # token 用量统计
    ├── prompts.py           # 系统提示词渲染
    ├── project_context.py   # 项目说明（MIMIR.md）与 git 状态
    ├── context.py           # token 估算与上下文压缩点计算
    ├── tool_clearing.py     # 旧工具输出清理
    ├── summary.py           # 早期对话摘要
    ├── graph.py             # LangGraph 图：compact → agent ⇄ tools，interrupt 人工确认
    ├── approval.py          # 确认策略
    ├── memory.py            # SQLite 会话持久化
    ├── transcript.py        # 明文 JSONL 对话记录
    ├── agent.py             # 运行时组装
    ├── cli.py               # 富文本 CLI
    └── tools/               # 工具（basic / filesystem / shell）
```

## 扩展

- **新增工具**：在 `mimir/tools/` 下用 `@tool` 编写，并在 `tools/__init__.py` 的 `build_tools` 中注册。
- **新增模型**：在 `config.yaml` 的 `models` 下加一项即可，无需改代码。
