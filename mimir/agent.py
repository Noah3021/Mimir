"""智能体运行时：组装 模型 + 工具 + 记忆 + 图，管理当前会话、推理强度与用量统计。"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from .approval import ApprovalPolicy
from .config import ConfigError, ModelConfig, Settings, normalize_reasoning
from .graph import build_agent_graph
from .llm import create_llm
from .memory import SessionStore
from .prompts import render_system_prompt
from .tools import build_tools
from .transcript import TranscriptWriter, now_iso
from .usage import Usage, UsageStore, summarize

logger = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(
        self,
        settings: Settings,
        model_name: str | None = None,
        *,
        approval_enabled: bool = True,
        reasoning: str | None = None,
    ):
        self.settings = settings
        self.tools = build_tools(settings.tools)
        self.approval = ApprovalPolicy(settings.tools.require_approval, enabled=approval_enabled)
        self.store = SessionStore(settings.memory.db_path)
        self.usage = UsageStore(self.store.conn)
        self.transcript = TranscriptWriter(
            settings.memory.transcripts_dir, settings.tools.workspace, enabled=settings.memory.transcripts_enabled
        )
        # 推理强度优先级：/think 设置（按模型）> 启动参数 --think > config.yaml
        self._reasoning_cli = normalize_reasoning(reasoning, "--think") if reasoning else None
        self._reasoning_override: dict[str, str] = {}
        self._turn_start = 0
        self._turn_started_at: str | None = None
        self._extra_calls: list[AIMessage] = []  # 本轮不在消息历史里的模型调用（如上下文摘要）
        # 系统提示词缓存：键为 (会话 ID, 模型名)。同一会话内保持逐字不变，以命中提示词缓存
        self._prompt_cache: tuple[tuple[str, str], str] | None = None
        self.thread_id = SessionStore.new_session_id()
        self.model_name = ""
        self.graph = None
        try:
            self.switch_model(model_name or settings.agent.default_model)
        except Exception:
            self.store.close()
            raise

    # ---------- 模型 / 推理 ----------
    @property
    def model_config(self) -> ModelConfig:
        return self.settings.models[self.model_name]

    def reasoning_for(self, name: str) -> str:
        return self._reasoning_override.get(name) or self._reasoning_cli or self.settings.models[name].reasoning

    @property
    def reasoning_level(self) -> str:
        return self.reasoning_for(self.model_name)

    def switch_model(self, name: str) -> None:
        mc = self.settings.models.get(name)
        if mc is None:
            raise ConfigError(f"未知模型：{name}（可选：{', '.join(self.settings.models)}）")
        llm = create_llm(mc, streaming=self.settings.cli.stream, reasoning=self.reasoning_for(name))
        self.graph = build_agent_graph(
            llm=llm,
            tools=self.tools,
            system_prompt=lambda: self._system_prompt(mc),
            approval=self.approval,
            checkpointer=self.store.checkpointer,
            context=self.settings.agent.context,
            context_window=mc.context_window,
            profile=name,
            max_tool_output_chars=self.settings.tools.max_output_chars,
            max_stream_retries=self.settings.agent.stream_retries,
            on_usage=self._extra_calls.append,  # 摘要请求不进入消息历史，单独记录用量
        )
        self.model_name = name
        logger.info("使用模型 %s (%s)，推理强度 %s", name, mc.model, self.reasoning_for(name))

    def set_reasoning(self, level: str) -> None:
        level = normalize_reasoning(level, "推理强度")
        name = self.model_name
        old = self._reasoning_override.get(name)
        self._reasoning_override[name] = level
        try:
            self.switch_model(name)
        except Exception:
            if old is None:
                self._reasoning_override.pop(name, None)
            else:
                self._reasoning_override[name] = old
            raise

    def _system_prompt(self, mc: ModelConfig) -> str:
        """每个会话（及模型）只渲染一次。

        图中每一步 LLM 调用都会取系统提示词；若每次重新渲染，任何变化都会让后续历史
        全部无法命中缓存。新建 / 恢复会话（thread_id 变化）或切换模型时才重新渲染。
        """
        key = (self.thread_id, mc.name)
        if self._prompt_cache and self._prompt_cache[0] == key:
            return self._prompt_cache[1]
        text = render_system_prompt(
            self.settings.agent.system_prompt,
            agent_name=self.settings.agent.name,
            workspace=self.settings.tools.workspace,
            model=f"{mc.name}（{mc.model}）",
            tools=[t.name for t in self.tools],
        )
        self._prompt_cache = (key, text)
        return text

    # ---------- 会话 ----------
    def run_config(self) -> dict:
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.settings.agent.recursion_limit,
        }

    def state(self):
        return self.graph.get_state(self.run_config())

    def messages(self) -> list[BaseMessage]:
        values = self.state().values or {}
        return list(values.get("messages", []))

    def pending_interrupts(self) -> list:
        st = self.state()
        ints = getattr(st, "interrupts", None)
        if ints is None:
            ints = [i for t in st.tasks for i in getattr(t, "interrupts", ())]
        return list(ints)

    def repair_dangling_tool_calls(self) -> int:
        """上一轮被中断时，可能留下没有结果的工具调用，补上“已取消”的结果，否则接口会报错。"""
        msgs = self.messages()
        if not msgs or not isinstance(msgs[-1], AIMessage) or not msgs[-1].tool_calls:
            return 0
        tool_msgs = [
            ToolMessage("操作已被用户中断，未执行。", tool_call_id=tc["id"], name=tc["name"], status="error")
            for tc in msgs[-1].tool_calls
        ]
        self.graph.update_state(self.run_config(), {"messages": tool_msgs}, as_node="tools")
        return len(tool_msgs)

    def new_session(self) -> str:
        self.thread_id = SessionStore.new_session_id()
        return self.thread_id

    def record_turn(self, user_input: str) -> None:
        title = " ".join(user_input.split())[:40] or "(空)"
        self.store.touch(self.thread_id, title, self.model_name)

    # ---------- 每轮开始 / 结束 ----------
    def begin_turn(self) -> None:
        """记录本轮开始时的消息数与时间，供明文记录和用量统计区分本轮消息。"""
        self._turn_started_at = now_iso()
        self._extra_calls.clear()
        try:
            self._turn_start = len(self.messages())
        except Exception:
            logger.exception("读取会话状态失败")
            self._turn_start = 0

    def finish_turn(self) -> Usage:
        """把本轮新产生的模型调用用量写入数据库，并返回本轮用量。"""
        msgs = self.messages()[self._turn_start:] + self._extra_calls
        self._extra_calls.clear()  # 不能重新赋值：图里绑定的是这个列表的 append
        self.usage.record(self.thread_id, msgs, profile=self.model_name, model_id=self.model_config.model)
        return summarize(msgs)

    def context_info(self) -> tuple[str, int]:
        """当前会话的 (早期对话摘要, 已被摘要代替的消息数)。"""
        values = self.state().values or {}
        return values.get("summary") or "", int(values.get("summary_cut") or 0)

    def session_usage(self) -> Usage:
        return summarize(self.messages())

    # ---------- 明文对话记录 ----------
    def sync_transcript(self) -> None:
        if not self.transcript.enabled:
            return
        try:
            self.transcript.sync(
                self.thread_id,
                self.messages(),
                profile=self.model_name,
                model_id=self.model_config.model,
                backfill_before=self._turn_start,
                user_time=self._turn_started_at,
            )
        except Exception:  # 记录失败不影响对话
            logger.exception("写入对话记录失败")

    def close(self) -> None:
        self.store.close()
