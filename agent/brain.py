"""
大脑决策层
"""
import re
import uuid
import json
import random
import time
import os
import asyncio
import hashlib

from agent.conversation_tracker import ConversationTracker
from agent.delegation import DelegationManager, TaskTicket, TicketState, TicketError
from agent.orchestration import OrchestrationManager, PlanState, SubTaskStatus
from agent.degradation import DegradationManager, DegradationLevel
from agent.utils import call_big_model_chat
import dateparser
from datetime import datetime, timezone, timedelta
from langgraph.types import Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain.agents.middleware import SummarizationMiddleware
from agent.model_config import model_config  # 导入配置
from agent.memory import get_memory
from agent.message_buffer import MessageBuffer
from agent.tools_factory import (
    create_get_current_time_tool,
    create_load_user_profile_tool,
    create_list_online_agents_tool,
    create_send_to_agent_tool,
    create_room_tools,
    create_log_memory_tool,
    create_delegation_tools,
    create_orchestration_tools,
    create_board_tools,
)
from deepagents import create_deep_agent, SubAgent
# from deepagents.backends.filesystem import FilesystemBackend
# from deepagents.backends import LocalShellBackend
from agent.scheduler import get_scheduler
from agent.tasks import send_reminder
from agent.db import get_pool
from agent.intent import IntentType, INTENT_DESCRIPTIONS
from agent.prompts import (
    REFLECTION_SUBAGENT_PROMPT,
    CLARIFICATION_PROMPT,
    REVIEWER_PROMPT_TEMPLATE,
    ESCALATION_PROMPT,
    build_brain_system_prompt,
    build_reminder_detection_prompt,
    build_intent_classification_prompt,
    build_termination_judge_prompt,
)
from config import config
from langchain.tools import tool
from langchain_tavily import TavilySearch
from agent.sandboxed_backend import DockerSandboxBackend
from agent.tools import (launch_agent, stop_agent, stop_all_agents_impl)
from pathlib import Path
from agent.reflection import init_chroma, submit_task_for_reflection, get_skill_collection, SKILLS_DIR
from agent.browser_tools import browser, close_browser_session
from agent.computer_tools import COMPUTER_TOOLS
from agent.task_buffer import TaskBuffer
from agent.context_manager import ContextManager, ToolOutputCompactionMiddleware
from agent.skill_version_manager import get_skill_latest_version, get_skill_file_path
from agent.skill_tools import list_skills, load_skill, search_skills, skill_stats, upgrade_skill, report_skill_result
from agent.conversation_memory_extractor import trigger_memory_extraction
from langchain_core.runnables import RunnableConfig
import chromadb

import logging
logging.getLogger('langgraph').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def _stable_hash(s: str) -> str:
    """返回跨进程稳定的内容哈希，用于去重 ID。"""
    return hashlib.sha256(s.encode()).hexdigest()[:16]

class Brain:
    
    def __init__(
            self,
            comm = None,
            db_pool=None,
            use_long_term_memory=True,
            agent_id=None,
            custom_system_prompt=None,
    ):
        
        self.agent_id = agent_id
        self.online_agents = None  # 由 core.Agent 注入，Set[str]
        self.user_id = None
        # 不应出现在可委派列表中的非智能体 ID
        self._NON_PEER_IDS = {"super_user", "reminder_bot"}
        # 获取模型
        self.model = model_config.get_model(config.model.default_provider)  # model_config 仍需按需
        self.thread_id = None
        
        self.comm = comm
        self.is_busy = False  # 标记是否正在处理用户请求
        self._current_task = None  # 当前正在运行的 asyncio Task（用于外部取消）
        self.last_run_time = datetime.now()
        self.memory = get_memory() if use_long_term_memory else None
        # # 初始化反思模块的向量库
        init_chroma(self.memory.client)
        # 检查点
        if db_pool is None:
            from agent.db import get_pool
            db_pool = get_pool()
        self.checkpointer = AsyncPostgresSaver(db_pool)
        self.db_pool = db_pool  # 供 managers 持久化使用

        # 智能体对话追踪器（轮次计数 + 逐级降级 + 硬上限）— 旧兜底机制
        self.conversation_tracker = ConversationTracker()
        # TDP 任务委托管理器（主控制机制）
        self.delegation_manager = DelegationManager()
        # 多智能体任务编排管理器（TDP 之上）
        self.orchestration_manager = OrchestrationManager()
        # 注入 DB 连接池以启用持久化
        self.delegation_manager.set_pool(db_pool)
        self.orchestration_manager.set_pool(db_pool)
        self._tdp_notification = None  # 当前待处理的 TDP 通知 payload
        self._timeout_loop_started = False  # 工单超时循环惰性启动
        self._in_flight_tools = []  # 当前正在执行的工具名称列表（用于中断时清理）
        self._orchestration_dispatch_hint: str = ""  # 编排子任务完成后，提示派发下一批
        self._skip_clarify = False  # 被打断后跳过下一条消息的澄清
        self.degradation_manager = DegradationManager()  # 多层级降级策略
        self.group_context = None  # 当前消息的群聊上下文
        # 和其他Agent交互工具
        self.send_to_agent_tool = create_send_to_agent_tool(self)
        # 创建群组相关的工具
        room_tools = create_room_tools(self)
        
        self.msg_buffer = MessageBuffer(delay_seconds=5)
        self._process_lock = asyncio.Lock()

        self._pending_approvals = {}
        # 后台任务生命周期跟踪
        self._bg_tasks: set[asyncio.Task] = set()
        # 用于去重
        self.sent_msg_ids_by_thread = {}  # thread_id -> set
        # 跨 stream 去重：已发送过 tool_call_end 的 ToolMessage ID（进程生命周期内）
        self._sent_tool_msgs: set[str] = set()

        self._termination_cache = {}
        # 任务缓冲模块
        self.task_buffer = TaskBuffer()
        # 上下文管理（token 预算 + 工具输出压缩）
        self.context_manager = ContextManager()
        # 加载 Agent 专属上下文文件（类似 Claude Code 的 CLAUDE.md）
        self.agent_context = self._load_agent_context()
        
        # 1. 配置后端 (FilesystemBackend 允许技能脚本访问本地文件)
        #    这里需要根据你的项目结构调整根目录
        # root_dir = os.path.expanduser("~")  # 这会得到当前用户的家目录
        root_dir = os.getcwd()
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        # backend = LocalShellBackend(
        #
        #     virtual_mode=config.backend.backend_virtual_mode,
        #     timeout=config.backend.backend_timeout,
        #     max_output_bytes=config.backend.backend_max_output_bytes,
        #     env={
        #         "PATH": f"{os.path.dirname(sys.executable)};{os.environ.get('PATH', '')}",
        #         "PYTHONPATH": root_dir,
        #         "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
        #         "POSTGRES_URI": os.environ.get("POSTGRES_URI", ""),
        #     }
        # )
        
        self.docker_backend = DockerSandboxBackend(
            # image="python:3.12-slim",
            image="my-agent-base:latest",  # 可自定义镜像
            mem_limit="1g",
            cpu_limit=1.0,
            # network_disabled=True,  # 根据需要允许或禁用网络
            network_disabled=False,  # 浏览器需要网络
            user="pwuser",  # 浏览器需要网络
            desktop_path=config.backend.docker_volumes,      # 如果需要控制电脑桌面文件夹，需要配置
            skills_host_path=os.path.join(os.getcwd(), "agent", "skills"),
            env={
                "API_KEY": config.model.api_key,
            },
            agent_id=self.agent_id,  # 持久化工作区隔离
        )
        
        # ========== 新增：反思子代理定义 ==========
        reflection_subagent = SubAgent(
            name="reflector",
            description="用于反思主代理的上一步执行结果，检查信息完整性、逻辑一致性，并给出修正建议。",
            system_prompt=REFLECTION_SUBAGENT_PROMPT,
            tools=[],  # 反思子代理不需要额外工具，只用自身推理
        )
        # ========== 反思子代理定义结束 ==========

        # 自定义工具
        delegation_tools = create_delegation_tools(self)
        orchestration_tools = create_orchestration_tools(self)
        board_tools = create_board_tools(self)
        tools = [self.send_to_agent_tool, create_list_online_agents_tool(self), create_get_current_time_tool(self), TavilySearch(max_results=5), create_log_memory_tool(self), create_load_user_profile_tool(self), launch_agent, stop_agent, stop_all_agents_impl, browser] + room_tools + COMPUTER_TOOLS + delegation_tools + orchestration_tools + board_tools
        tools = tools + [list_skills, load_skill, search_skills, skill_stats, upgrade_skill, report_skill_result]
        self.agent = create_deep_agent(
            model=self.model,
            tools=tools,
            system_prompt=self._build_system_prompt() if custom_system_prompt is None else custom_system_prompt,
            # backend=backend,
            backend=self.docker_backend,
            checkpointer=self.checkpointer,
            subagents=[reflection_subagent],  # 在线反思子代理
            interrupt_on={
                "launch_agent": {"allowed_decisions": ["approve", "reject"]},
                "computer_execute": {"allowed_decisions": ["approve", "reject"]},       # 慎重
                # "browser": {"allowed_decisions": ["approve", "reject"]},
            },
            middleware=[
                    SummarizationMiddleware(
                        model=self.model,
                        trigger=("tokens", 20000),  # 当历史超过 20000 token 时触发
                        keep=("messages", 30),  # 保留最近 30 条消息，其余用摘要代替
                    ),
                    ToolOutputCompactionMiddleware(self.context_manager),
            ]
        )

    @property
    def peer_agents(self) -> set:
        """返回可委派/协作的在线智能体（排除用户、提醒机器人等非智能体角色和自身）。"""
        if not self.online_agents:
            return set()
        return {a for a in self.online_agents
                if a not in self._NON_PEER_IDS and a != self.agent_id}

    @property
    def _is_orchestrating(self) -> bool:
        """当存在由本 agent 发起且未完成的编排计划时返回 True。
        此时 agent-to-agent 消息应纯程序化处理，LLM 不参与。"""
        return any(
            p.state in (PlanState.READY, PlanState.EXECUTING)
            and p.issuer == self.agent_id
            for p in self.orchestration_manager._plans.values()
        )

    def _resolve_plan_id(self, tdp: dict, ticket, ticket_id: str) -> str:
        """三级回退解析编排计划 ID：
        1. 消息 payload 的 _orchestration 字段
        2. ticket 的 orchestration_plan_id
        3. _ticket_index 反查
        """
        plan_id = tdp.get("_orchestration", "") if tdp else ""
        if not plan_id and ticket:
            plan_id = getattr(ticket, 'orchestration_plan_id', None) or ""
        if not plan_id:
            entry = self.orchestration_manager._ticket_index.get(ticket_id)
            if entry:
                plan_id = entry[0]
        return plan_id

    async def _handle_tdp_notification(self, user_input: str):
        """处理 TDP 协议通知（委托/交付/接受/拒绝/取消/超时）。
        返回 (should_skip: bool, user_input: str)——should_skip 为 True 表示调用方应跳过 LLM 直接返回，
        user_input 可能被修改（非编排交付时重写为格式化摘要）。
        """
        tdp = self._tdp_notification
        if not tdp:
            return False, user_input

        tdp_type = tdp.get("_tdp", "")
        tdp_ticket_id = tdp.get("ticket_id", "")
        dm = self.delegation_manager
        _skip_llm_for_issuer = False

        # ── 终态守卫：已终止的工单不接受任何 TDP 操作（幂等丢弃）──
        if tdp_ticket_id and tdp_type in ("acceptance", "delivery", "decline"):
            existing = dm.get_ticket(tdp_ticket_id)
            if existing and existing.is_terminal:
                logger.info(
                    f"TDP: 忽略对已终止工单 {tdp_ticket_id} "
                    f"({existing.state.value}) 的 {tdp_type} 操作"
                )
                self._tdp_notification = None

        if tdp_type == "delegation" and tdp_ticket_id:
            # 收到委托通知 — 在 assignee 侧创建镜像工单
            existing = dm.get_ticket(tdp_ticket_id)
            if not existing:
                desc = tdp.get("text", user_input)[:500]
                try:
                    import uuid as _uuid
                    ticket = TaskTicket(
                        ticket_id=tdp_ticket_id,
                        issuer=self.user_id,
                        assignee=self.agent_id,
                        description=desc,
                        expected_output=tdp.get("expected_output", "未指定"),
                        max_rounds=tdp.get("max_rounds", 8),
                    )
                    orch_plan = tdp.get("_orchestration", "")
                    if orch_plan:
                        ticket.orchestration_plan_id = orch_plan
                    dm._by_id[tdp_ticket_id] = ticket
                    dm._by_pair[dm._pair_key(self.agent_id, self.user_id)] = tdp_ticket_id
                    dm._save_ticket(ticket)
                    logger.info(f"TDP 镜像工单已创建: {tdp_ticket_id} (assignee={self.agent_id})")
                except Exception as e:
                    logger.warning(f"创建 TDP 镜像工单失败: {e}")

        elif tdp_type == "delivery" and tdp_ticket_id:
            ticket = dm.get_ticket(tdp_ticket_id)
            if ticket and ticket.is_active:
                try:
                    dm.transition(tdp_ticket_id, TicketState.CLOSED)
                    ticket.result_summary = user_input[:500]
                except TicketError:
                    pass
            # ── 编排回调：子任务交付 ──
            orch_plan_id = self._resolve_plan_id(tdp, ticket, tdp_ticket_id)
            if orch_plan_id:
                plan = self.orchestration_manager.get_plan(orch_plan_id)
                result_pair = self.orchestration_manager.get_subtask_by_ticket(tdp_ticket_id)
                subtask = result_pair[1] if result_pair else None

                # ── 检查是否有审核 Agent ──
                if subtask and subtask.reviewer_agent:
                    # 进入审核流程
                    subtask.status = SubTaskStatus.REVIEWING
                    subtask.result = user_input[:500]
                    await self.orchestration_manager._save_plan(plan)

                    await self.comm.send_to_agent(subtask.reviewer_agent, {
                        "_tdp": "review_request",
                        "_orchestration": orch_plan_id,
                        "subtask_id": subtask.id,
                        "delivery_summary": user_input[:500],
                        "original_description": subtask.description,
                        "expected_output": getattr(ticket, 'expected_output', '') if ticket else '',
                    })
                    await self._safe_send(
                        f"🔍 子任务 {subtask.id} 已交付，已发送给审核 Agent "
                        f"{subtask.reviewer_agent} 审核..."
                    )
                    # 不继续处理，等待审核结果
                    return

                # ── 无审核 Agent，直接标记完成 ──
                await self.orchestration_manager.mark_completed(tdp_ticket_id, user_input[:500])
                if plan and plan.is_complete():
                    summary = self.orchestration_manager.get_progress_summary(orch_plan_id)
                    await self.comm.send_to_agent("super_user", {
                        "text": f"🎯 编排计划全部完成！\n\n{summary}"
                    })
                    self.schedule_background_task(
                        self._synthesize_plan_results(orch_plan_id)
                    )
                elif plan:
                    ready = plan.get_ready_subtasks()
                    if ready:
                        dispatched = await self._dispatch_ready_subtasks(plan, ready)
                        await self.comm.send_to_agent("super_user", {
                            "text": (
                                f"📤 子任务 {tdp_ticket_id} 已交付。"
                                f"自动派发下一批: {dispatched}"
                            )
                        })
                    else:
                        pending = [s for s in plan.subtasks if s.status == SubTaskStatus.PENDING]
                        if pending:
                            blocked = [f"{s.id} (依赖: {', '.join(s.depends_on)})" for s in pending]
                            logger.info(
                                f"编排计划 {orch_plan_id}: 子任务 {tdp_ticket_id} 完成，"
                                f"但以下子任务仍被阻塞: {blocked}"
                            )
            # ── 委托方：区分编排与非编排交付 ──
            if ticket and ticket.issuer == self.agent_id:
                if orch_plan_id:
                    _skip_llm_for_issuer = True
                    logger.info(
                        f"TDP 编排交付已程序化处理，跳过 LLM "
                        f"(issuer={self.agent_id}, ticket={tdp_ticket_id})"
                    )
                    if self.thread_id:
                        self.task_buffer.add_step(
                            self.thread_id,
                            f"来自 {self.user_id} 的编排交付（已自动处理）",
                            user_input[:300],
                        )
                else:
                    # 非编排交付：截断内容，LLM 可审查但禁止与 worker 对话
                    result_preview = user_input[:300]
                    if len(user_input) > 300:
                        result_preview += f"...（完整结果已保存，共 {len(user_input)} 字符）"
                    user_input = (
                        f"[子任务交付] 工单 {tdp_ticket_id} 已完成。"
                        f"\n交付摘要: {result_preview}"
                        f"\n\n⚠️ 工单已关闭，禁止回复 {self.user_id}。"
                        f"如需汇报，向 super_user 汇报结果。"
                    )

        elif tdp_type == "acceptance" and tdp_ticket_id:
            ticket = dm.get_ticket(tdp_ticket_id)
            if ticket and ticket.state == TicketState.PENDING:
                try:
                    dm.transition(tdp_ticket_id, TicketState.ACCEPTED)
                except TicketError:
                    pass
            orch_plan_id = self._resolve_plan_id(tdp, ticket, tdp_ticket_id)
            if orch_plan_id:
                await self.orchestration_manager.mark_accepted(tdp_ticket_id)
            is_orch_ticket = bool(orch_plan_id or getattr(ticket, 'orchestration_plan_id', None) if ticket else False)
            if ticket and ticket.is_active and not is_orch_ticket:
                try:
                    dm.cancel_ticket(tdp_ticket_id, f"TDP {tdp_type} 通知", self.user_id)
                except TicketError:
                    pass
            if ticket and ticket.issuer == self.agent_id:
                _skip_llm_for_issuer = True
                logger.info(f"TDP 接受通知已程序化处理，跳过 LLM (issuer={self.agent_id})")

        elif tdp_type == "review_request":
            # 收到审核请求——由审核 Agent 处理
            await self._handle_review_request(tdp)
            self._tdp_notification = None
            return True, user_input  # 纯程序化处理，跳过 LLM

        elif tdp_type in ("decline", "cancel", "timed_out") and tdp_ticket_id:
            decline_ticket = dm.get_ticket(tdp_ticket_id)
            orch_plan_id = self._resolve_plan_id(tdp, decline_ticket, tdp_ticket_id)
            if orch_plan_id:
                await self.orchestration_manager.mark_failed(
                    tdp_ticket_id,
                    f"工单 {tdp_ticket_id} 被{tdp_type}"
                )

        self._tdp_notification = None  # 消费后清空

        # ── 委托方：TDP 通知已程序化处理，无需 LLM 介入 ──
        if _skip_llm_for_issuer:
            if self.thread_id:
                self.task_buffer.add_step(
                    self.thread_id,
                    f"来自 {self.user_id} 的 TDP 通知（已自动处理）",
                    user_input[:300],
                )
            return True, user_input

        return False, user_input

    async def _llm_review(self, description: str, delivery: str,
                           expected_output: str) -> dict:
        """调用 LLM 执行审核。"""
        prompt = REVIEWER_PROMPT_TEMPLATE.format(
            description=description,
            delivery=delivery,
            expected_output=expected_output,
        )
        try:
            response = await call_big_model_chat(
                prompt,
                model=config.model.memory_extraction_model,
                temperature=0.2,
                is_json=True,
            )
            content = response["choices"][0]["message"]["content"]
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            return json.loads(content)
        except Exception as e:
            logger.warning(f"LLM 审核失败: {e}")
            return {"passed": True, "feedback": f"审核异常自动通过: {e}", "score": 8}

    async def _handle_review_request(self, msg: dict):
        """处理审核请求——由审核 Agent 调用。"""
        plan_id = msg.get("_orchestration", "")
        subtask_id = msg.get("subtask_id", "")
        delivery_summary = msg.get("delivery_summary", "")
        original_description = msg.get("original_description", "")
        expected_output = msg.get("expected_output", "")

        # 调用 LLM 审核
        review_result = await self._llm_review(
            description=original_description,
            delivery=delivery_summary,
            expected_output=expected_output,
        )

        passed = review_result.get("passed", False)
        feedback = review_result.get("feedback", "无具体反馈")
        score = review_result.get("score", 0)

        plan = self.orchestration_manager.get_plan(plan_id)
        if not plan:
            return
        st = plan.get_subtask(subtask_id)
        if not st:
            return

        if passed:
            await self._safe_send(
                f"✅ 审核通过: {subtask_id} (评分: {score}/10)\n{feedback}"
            )
            st.status = SubTaskStatus.APPROVED
            st.review_feedback = feedback
            plan.completed_count += 1
            await self.orchestration_manager._save_plan(plan)

            # 自动生成技能
            self.schedule_background_task(
                self._auto_create_skill(plan_id, subtask_id)
            )

            if plan.is_complete():
                self.schedule_background_task(
                    self._synthesize_plan_results(plan_id)
                )
            else:
                ready = plan.get_ready_subtasks()
                if ready:
                    dispatched = await self._dispatch_ready_subtasks(plan, ready)
                    await self._safe_send(
                        f"📤 子任务审核通过，自动派发下一批: {dispatched}"
                    )
        else:
            # 审核不通过，检查重试次数
            if st.retry_count >= st.max_retries:
                st.status = SubTaskStatus.BLOCKED
                st.blocked_reason = f"审核不通过且已达最大重试次数: {feedback}"
                st.review_feedback = feedback
                await self.orchestration_manager._save_plan(plan)
                await self._safe_send(
                    f"⚠️ 子任务 {subtask_id} 已达最大重试次数 ({st.retry_count}/{st.max_retries})，触发升级"
                )
                self.schedule_background_task(
                    self._handle_escalation(plan_id, subtask_id)
                )
            else:
                st.retry_count += 1
                st.review_feedback = feedback
                st.attempts.append({
                    "time": datetime.now().isoformat(),
                    "feedback": feedback,
                    "result_summary": delivery_summary[:200],
                })
                st.status = SubTaskStatus.PENDING
                await self.orchestration_manager._save_plan(plan)
                await self._safe_send(
                    f"🔄 子任务 {subtask_id} 审核不通过 (评分: {score}/10)，"
                    f"第 {st.retry_count}/{st.max_retries} 次重试:\n{feedback}"
                )
                # 重新派发
                online = list(self.peer_agents)
                if st.assigned_to and st.assigned_to in online:
                    await self._dispatch_single_subtask(
                        plan_id, st, online, allow_parallel=True,
                        assigned_to=st.assigned_to,
                    )

    async def _llm_analyze_escalation(self, subtask_id: str, description: str,
                                        blocked_reason: str,
                                        attempts: list) -> dict:
        """调用 LLM 分析困境并给出决策。"""
        attempts_summary = json.dumps(attempts[-3:], ensure_ascii=False) if attempts else "无"
        prompt = ESCALATION_PROMPT.format(
            subtask_id=subtask_id,
            description=description,
            blocked_reason=blocked_reason,
            attempts_summary=attempts_summary,
        )
        try:
            response = await call_big_model_chat(
                prompt,
                model=config.model.memory_extraction_model,
                temperature=0.2,
                is_json=True,
            )
            content = response["choices"][0]["message"]["content"]
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            return json.loads(content)
        except Exception as e:
            logger.warning(f"LLM 升级分析失败: {e}")
            return {"resolution": "retry", "reason": f"LLM 不可用，自动重试: {e}"}

    async def _handle_escalation(self, plan_id: str, subtask_id: str):
        """处理困境升级请求——由主管 Agent 调用。"""
        plan = self.orchestration_manager.get_plan(plan_id)
        if not plan:
            return
        st = plan.get_subtask(subtask_id)
        if not st:
            return

        # 记录升级日志
        plan.escalation_log.append({
            "time": datetime.now().isoformat(),
            "subtask_id": subtask_id,
            "reason": st.blocked_reason or "未知原因",
            "resolution": "研讨中",
        })

        # 通知用户
        await self._safe_send(
            f"🚨 子任务 {subtask_id}（{st.description[:60]}...）遇到困境，"
            f"主管正在研讨解决方案..."
        )

        # LLM 分析困境
        analysis = await self._llm_analyze_escalation(
            subtask_id=subtask_id,
            description=st.description,
            blocked_reason=st.blocked_reason or "未知",
            attempts=st.attempts,
        )

        resolution = analysis.get("resolution", "retry")

        if resolution == "skip":
            st.skipped = True
            st.status = SubTaskStatus.SKIPPED
            plan.critical_decisions.append(
                f"子任务 {subtask_id} 被跳过: {analysis.get('reason', '无')}"
            )
            await self._safe_send(f"⏭️ 子任务 {subtask_id} 已被跳过，继续执行其他任务")

        elif resolution == "reassign":
            new_agent = analysis.get("new_agent", "")
            if new_agent:
                st.assigned_to = new_agent
                st.status = SubTaskStatus.PENDING
                await self._safe_send(f"🔄 子任务 {subtask_id} 已重新分配给 {new_agent}")
                online = list(self.peer_agents)
                if new_agent in online:
                    await self._dispatch_single_subtask(
                        plan_id, st, online, allow_parallel=True,
                        assigned_to=new_agent,
                    )

        elif resolution == "modify":
            new_description = analysis.get("new_description", st.description)
            st.description = new_description
            st.status = SubTaskStatus.PENDING
            plan.critical_decisions.append(
                f"子任务 {subtask_id} 描述已修正: {new_description[:100]}..."
            )
            await self._safe_send(f"📝 子任务 {subtask_id} 描述已修正，重新执行")

        else:  # retry
            st.retry_count += 1
            st.status = SubTaskStatus.PENDING
            await self._safe_send(
                f"🔄 子任务 {subtask_id} 第 {st.retry_count} 次重试"
            )

        # 保存状态并尝试派发
        await self.orchestration_manager._save_plan(plan)
        if st.status == SubTaskStatus.PENDING:
            online = list(self.peer_agents)
            if st.assigned_to and st.assigned_to in online:
                await self._dispatch_single_subtask(
                    plan_id, st, online, allow_parallel=True,
                    assigned_to=st.assigned_to,
                )
            else:
                ready = plan.get_ready_subtasks()
                if ready:
                    await self._dispatch_ready_subtasks(plan, ready)

    async def _safe_llm_call(self, prompt: str, max_retries: int = 2,
                              prompt_hint: str = "") -> dict:
        """安全的 LLM 调用，含降级策略。"""
        dm = self.degradation_manager
        for attempt in range(max_retries + 1):
            try:
                response = await call_big_model_chat(
                    prompt,
                    model=config.model.memory_extraction_model,
                    temperature=0.2,
                    is_json=True,
                )
                content = response["choices"][0]["message"]["content"]
                if content.startswith("```"):
                    lines = content.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    content = "\n".join(lines)
                return json.loads(content)
            except (asyncio.TimeoutError, ConnectionError,
                    json.JSONDecodeError, Exception) as e:
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                if dm.enabled:
                    dm.set_level(DegradationLevel.DEGRADED_LLM,
                                 f"LLM 调用失败: {e}")
                logger.warning(f"LLM 调用降级 (hint={prompt_hint}): {e}")
                return dm.get_fallback_response(prompt_hint)
        return {"error": "LLM 完全不可用"}

    async def _check_review_deadlock(self, plan_id: str):
        """检查审核死锁，自动终审仲裁。"""
        plan = self.orchestration_manager.get_plan(plan_id)
        if not plan:
            return
        now = datetime.now()
        for st in plan.subtasks:
            if st.status == SubTaskStatus.REVIEWING:
                review_start = st.escalated_at
                if review_start:
                    duration = (now - review_start).total_seconds()
                    if duration > 600:
                        st.status = SubTaskStatus.APPROVED
                        st.review_feedback = (
                            f"审核超时（{int(duration)}s），自动通过（终审仲裁）"
                        )
                        plan.critical_decisions.append(
                            f"子任务 {st.id} 审核超时（{int(duration)}s），自动通过"
                        )
                        await self.orchestration_manager._save_plan(plan)
                        await self._safe_send(
                            f"⚖️ 子任务 {st.id} 审核超时，已自动通过（终审仲裁）"
                        )
                        ready = plan.get_ready_subtasks()
                        if ready:
                            await self._dispatch_ready_subtasks(plan, ready)

    async def ensure_agent_with_fallback(self, plan_id: str, role: str,
                                          system_prompt: str) -> tuple:
        """确保 Agent 可用，含降级策略。返回 (agent_id, 是否降级)。"""
        try:
            agent_id = await self.orchestration_manager.ensure_agent(
                plan_id, role, system_prompt, brain=self)
            return agent_id, False
        except TimeoutError:
            for i in range(3):
                try:
                    await asyncio.sleep(2 ** i)
                    agent_id = await self.orchestration_manager.ensure_agent(
                        plan_id, role, system_prompt, brain=self)
                    return agent_id, True
                except TimeoutError:
                    continue

            await self._safe_send(
                f"⚠️ 无法启动 {role} Agent，主管将接管该任务"
            )
            if self.degradation_manager.enabled:
                self.degradation_manager.set_level(
                    DegradationLevel.DEGRADED_NET,
                    f"无法启动 {role} Agent"
                )
            return self.agent_id, True

    async def _dispatch_single_subtask(self, plan_id, subtask, online_agents,
                                       allow_parallel=True, reassign=False,
                                       assigned_to=None) -> str:
        """派发单个子任务：分配智能体、创建工单、通知目标。

        由 _dispatch_ready_subtasks 和编排工具共用，消除重复逻辑。
        调用者负责异常处理。

        Args:
            plan_id: 编排计划 ID
            subtask: SubTask 对象（含 id, description, suggested_role）
            online_agents: 在线智能体列表
            allow_parallel: 是否允许同对智能体并行工单（编排场景为 True）
            reassign: 是否为重新分配（影响通知文本）
            assigned_to: 若指定则跳过角色匹配直接分配

        Returns:
            结果描述字符串，如 "✅ st_1 → agent_x (工单 `abc`)"
        """
        dm = self.delegation_manager
        om = self.orchestration_manager

        if assigned_to:
            assigned = assigned_to
        else:
            assigned = None
            role_hint = (subtask.suggested_role or "").lower()
            for agent in online_agents:
                if role_hint and role_hint in agent.lower():
                    assigned = agent
                    break
            if not assigned:
                others = [a for a in online_agents if a != self.agent_id]
                assigned = others[0] if others else online_agents[0]

        prefix = "[编排任务 重新分配]" if reassign else "[编排任务]"
        expected_output = f"{'重新分配' if reassign else '完成子任务'}: {subtask.description}"

        ticket = dm.create_ticket(
            issuer=self.agent_id,
            assignee=assigned,
            description=subtask.description,
            expected_output=expected_output,
            max_rounds=config.agent.orchestration_max_rounds,
            allow_parallel=allow_parallel,
        )
        ticket.orchestration_plan_id = plan_id
        dm._save_ticket(ticket)
        await om.dispatch_subtask(plan_id, subtask.id, assigned, ticket.ticket_id)
        await self.comm.send_to_agent(assigned, {
            "text": (
                f"{prefix} 这是计划 {plan_id} 的子任务 {subtask.id}。\n"
                f"任务: {subtask.description}\n"
                f"完成后请使用 deliver_result 交付结果。"
            ),
            "_tdp": "delegation",
            "_orchestration": plan_id,
            "ticket_id": ticket.ticket_id,
            "max_rounds": config.agent.orchestration_max_rounds,
            "expected_output": expected_output,
        })
        return f"✅ {subtask.id} → {assigned} (工单 `{ticket.ticket_id}`)", ticket.ticket_id

    async def _dispatch_ready_subtasks(self, plan, ready_subtasks) -> str:
        """程序化派发就绪子任务（无需 LLM 参与）。
        由 delivery 回调调用，自动推进编排计划。
        返回派发结果描述字符串。
        """
        online = list(self.peer_agents)
        if not online:
            return "（无在线智能体可供派发）"

        results = []
        for st in ready_subtasks:
            try:
                msg, ticket_id = await self._dispatch_single_subtask(
                    plan.plan_id, st, online, allow_parallel=True)
                results.append(msg)
                logger.info(
                    f"编排自动派发: {st.id} → (plan={plan.plan_id}, ticket={ticket_id})"
                )
            except Exception as e:
                results.append(f"❌ {st.id} 派发失败: {e}")
                logger.error(f"编排自动派发失败 {st.id}: {e}")

        if plan.state == PlanState.READY:
            plan.state = PlanState.EXECUTING
        return "; ".join(results) if results else "无子任务被派发"

    async def _synthesize_plan_results(self, plan_id: str):
        """计划全部完成后，后台触发 LLM 汇总所有子任务结果并呈现最终报告给用户。

        作为后台任务运行：等待当前 process() 释放锁后，以各子任务交付结果
        作为输入，让 LLM 整合成一份完整的最终报告发送给 super_user。
        """
        await asyncio.sleep(1.5)  # 等待当前 delivery processing 完成并释放 _process_lock

        plan = self.orchestration_manager.get_plan(plan_id)
        if not plan or not plan.is_complete():
            return

        # 收集所有子任务结果
        results_parts = []
        for st in plan.subtasks:
            agent_info = f" (执行者: {st.assigned_to})" if st.assigned_to else ""
            result_text = st.result or "(未返回结果)"
            results_parts.append(
                f"### {st.id}: {st.description}{agent_info}\n"
                f"**交付摘要**: {result_text[:600]}"
            )
        results_block = "\n\n".join(results_parts)

        synthesis_prompt = (
            f"🎯 你之前发起的编排任务「{plan.description}」已全部完成！\n\n"
            f"以下是各子任务的交付结果：\n\n"
            f"{results_block}\n\n"
            f"---\n"
            f"请整合以上所有子任务的结果，向用户呈现一份完整、清晰的**最终汇总报告**。"
            f"要求：\n"
            f"1. 直接呈现报告内容，不要说'我来汇总'、'好的让我整合'之类的开场白\n"
            f"2. 报告应逻辑连贯，覆盖各子任务的核心发现\n"
            f"3. 保持专业但易读的风格，适当使用表格和分节\n"
            f"4. 最后可以询问用户是否需要深入了解某个具体方面"
        )

        logger.info(f"触发编排结果自动汇总: plan={plan_id}")

        # 清理子 Agent
        await self.orchestration_manager.cleanup_agents(plan_id, brain=self)

        await self.process(
            user_id="super_user",
            user_input=synthesis_prompt,
            new_thread=False,
            thread_id_override=f"private_{self.agent_id}_super_user",
            silent=False,
        )

    def _agent_message_needs_response(self, text: str) -> bool:
        """判断来自其他 Agent 的消息是否需要 LLM 回复。

        纯状态更新（"我在搜索"、"数据已找到"等）不需要 LLM 处理，
        只需记录到 task_buffer。只有明确的求助、提问、指令才需要触发 LLM。
        这避免了 LLM 读取共享 checkpoint 时因角色混淆而扮演错误身份。

        核心原则：
        - TDP 交付 → 委托方需要 LLM 审查结果、派发下一批子任务
        - TDP 接受/拒绝 → 已程序化处理，委托方不需要 LLM
        - 委托方不与被委托方闲聊 — 只响应明确的提问或求助
        """
        if not text:
            return False

        # ── 检查当前对话中的角色关系 ──
        ticket = self.delegation_manager.get_active_ticket(self.agent_id, self.user_id)
        is_issuer = ticket and ticket.issuer == self.agent_id

        # TDP 交付 — 委托方始终需要 LLM 审查结果、推进编排
        if re.search(r'\[TDP\s*交付\]', text):
            return True

        # 委托方：TDP 接受/拒绝通知已程序化处理，不需要 LLM
        if is_issuer and re.search(r'\[TDP\]', text):
            return False

        # TDP 委托/编排任务 — 被委托方需要 LLM 来接受/处理
        if re.search(r'\[任务委托\]|\[TDP|\[编排任务\]', text):
            return True

        # 委托方：拒绝所有来自被委托方的非提问/非求助消息
        if is_issuer:
            if re.search(r'[?？]|能否|可以吗|你能|可否|怎么|如何|帮我|帮我看|需要你', text):
                return True
            if re.search(r'请\s*(调用|执行|使用|尝试|确认|验证|检查|处理|操作|交付|取消|查看|帮|协助|回复)', text):
                return True
            # 纯状态更新/闲聊 — 委托方无需回复
            return False

        # 被委托方或无工单：保持原有逻辑
        if re.search(r'请\s*(调用|执行|使用|尝试|确认|验证|检查|处理|操作|交付|取消|查看|帮|协助|回复)', text):
            return True
        if re.search(r'[?？]|吗[？?]?[\s]*$|能否|可以吗|你能|可否|怎么|如何|帮我|帮我看|需要你', text):
            return True
        return False

    async def _build_system_contexts(self, image_data: str = None) -> list:
        """Build system-level context as a list of strings for SystemMessage injection.

        These are things the agent needs to know *about* the current turn but
        that are NOT part of what the user/other-agent said.  By putting them
        in SystemMessages instead of prepending to HumanMessage we keep the
        role boundary clean — the LLM can distinguish instruction from input.

        Returns a list of non-empty context strings (may be empty).
        """
        parts = []

        # -- conversation partner info -----------------------------------
        if self.user_id == 'super_user':
            parts.append("[当前对话] 你正在和人类用户 (id是super_user) 对话。")
            # ── 编排上下文：提醒 orchestrator 汇报进展 ──
            active_plans = [
                p for p in self.orchestration_manager._plans.values()
                if p.state in (PlanState.READY, PlanState.EXECUTING)
                and p.issuer == self.agent_id
            ]
            if active_plans:
                for p in active_plans[:2]:  # 最多展示 2 个
                    parts.append(
                        f"[编排提醒] 你有一个活跃的编排计划「{p.description[:60]}」"
                        f"（{p.completed_count}/{p.total_count} 完成）。"
                        f"收到子任务交付后请推进计划，进展及时向用户汇报。"
                    )
        else:
            status = "在线" if (self.online_agents and self.user_id in self.online_agents) else "未知"
            parts.append(
                f"[重要身份确认] 你是 {self.agent_id}，你正在和智能体 {self.user_id} 对话。"
                f"你永远只能以 {self.agent_id} 的身份说话和行动，不要替 {self.user_id} 说话，"
                f"不要扮演 {self.user_id} 的角色，不要替对方做决定。"
                f"对方当前状态：{status}。"
            )

            # ── TDP 工单上下文注入 ──
            ticket = self.delegation_manager.get_active_ticket(self.agent_id, self.user_id)
            if ticket:
                role = "委托方" if ticket.issuer == self.agent_id else "被委托方"
                if ticket.issuer == self.agent_id:
                    role_instruction = (
                        f"## 身份边界警告\n"
                        f"你是**委托方（orchestrator）**，{self.user_id} 是**被委托方（worker）**。\n\n"
                        f"**你的核心职责：**\n"
                        f"1. 等待 {self.user_id} 完成工作并交付结果\n"
                        f"2. 接收交付结果后，立即推进编排计划（派发下一批子任务）\n"
                        f"3. 重要进展必须向用户 (super_user) 汇报，而不是和 worker 聊天\n\n"
                        f"**绝对禁止的行为：**\n"
                        f"- 禁止代替 {self.user_id} 执行具体任务（如搜索、写文件、分析数据）\n"
                        f"- 禁止以 {self.user_id} 的身份说话或行动\n"
                        f"- 禁止'接手'对方正在进行的工作\n"
                        f"- **禁止回复 {self.user_id} 的进度消息**（如'我在搜索'、'数据已整理'）\n"
                        f"- **禁止在收到交付后与 worker 进行任何对话**\n"
                        f"  → 收到交付后唯一正确的行动：派发下一批子任务，或向用户汇报完成\n"
                        f"- {self.user_id} 发来的消息中，TDP 接受通知已自动处理，你无需回复\n"
                    )
                else:
                    role_instruction = (
                        f"## 身份边界警告\n"
                        f"你是**被委托方**，{self.user_id} 是**委托方**。你的职责是：\n"
                        f"1. 按照工单描述执行具体任务\n"
                        f"2. 完成后使用 deliver_result 交付结果\n"
                        f"3. 遇到疑问时使用 request_clarification 澄清\n\n"
                        f"**绝对禁止的行为：**\n"
                        f"- 禁止让 {self.user_id} 替你做本该你完成的工作\n"
                        f"- 禁止以 {self.user_id} 的身份说话或行动"
                    )
                parts.append(
                    f"\n{role_instruction}"
                    f"\n[任务委托信息]"
                    f"\n工单ID: {ticket.ticket_id}"
                    f"\n任务描述: {ticket.description}"
                    f"\n期望产出: {ticket.expected_output}"
                    f"\n轮次进度: {ticket.round_count}/{ticket.max_rounds}"
                    f"\n当前状态: {ticket.state.value}"
                )
                if ticket.state == TicketState.PENDING and ticket.assignee == self.agent_id:
                    parts.append("⚠️ 请使用 accept_task 接受此委托，或 decline_task 拒绝。")
                elif ticket.state == TicketState.ACCEPTED and ticket.assignee == self.agent_id:
                    parts.append("委托已接受，请开始执行任务。完成后使用 deliver_result 交付结果。")

                tdp_warning = ticket.get_warning_hint(self.user_id)
                if tdp_warning:
                    parts.append(tdp_warning)

            else:
                # 无活跃工单时使用旧 ConversationTracker 警告
                warning = self.conversation_tracker.get_warning(self.agent_id, self.user_id)
                if warning:
                    parts.append(warning)

            # ── 编排派发提示：子任务完成后，提示推进计划 ──
            if self._orchestration_dispatch_hint:
                parts.append(f"\n⚠️ [编排推进] {self._orchestration_dispatch_hint}")
                self._orchestration_dispatch_hint = ""  # 消费后清空

        # -- group chat context -------------------------------------------
        gc = self._build_group_context_prompt()
        if gc:
            parts.append(gc)

        # -- image description --------------------------------------------
        if image_data:
            image_desc = await self._handle_image(image_data)
            if image_desc:
                parts.append(f"[图片信息] 对方刚上传了一张图片，内容描述如下：\"{image_desc}\"")

        return parts

    async def _build_messages_for_stream(self, user_input: str, image_data: str, original_user_input: str) -> list:
        """组装用于 stream 的消息列表：系统上下文 + 技能经验 + 用户输入。"""
        system_contexts = await self._build_system_contexts(image_data)
        skill_lessons = await self._get_relevant_skill_lessons(original_user_input)
        if skill_lessons:
            system_contexts.append(skill_lessons)
        messages = []
        for ctx in system_contexts:
            if ctx:
                messages.append({"role": "system", "content": ctx})
        messages.append({"role": "user", "content": user_input})
        return messages

    async def _prepare_agent_config(self) -> tuple:
        """创建 config、修复 checkpoint、加载 sent_ids。返回 (config, sent_ids)。"""
        config = {"configurable": {"thread_id": self.thread_id}}
        await self._sanitize_checkpoint(config)
        if self.thread_id not in self.sent_msg_ids_by_thread:
            self.sent_msg_ids_by_thread[self.thread_id] = await self._load_sent_ids_from_checkpoint(config)
        return config, self.sent_msg_ids_by_thread[self.thread_id]

    async def _cleanup_in_flight_tools(self):
        """清理所有未完成的工具调用状态，发送 tool_call_end 通知前端。"""
        if not self._in_flight_tools:
            return
        tools = list(self._in_flight_tools)
        self._in_flight_tools.clear()
        for tool_name in tools:
            await self._safe_send(
                f"🛠️ {tool_name} 因任务中断而终止",
                type="tool_call_end",
                tool_name=tool_name,
            )

    async def _safe_send(self, text: str, **extra) -> bool:
        """安全发送消息给当前用户。WebSocket/网络异常仅记日志，不向上传播。
        自动附加 thread_id，供前端正确路由消息到对应会话。"""
        # 过滤空消息，避免向 Hub 发送无意义的数据包
        if not text and not extra:
            return False
        payload = {"text": text, **extra}
        if self.user_id == 'super_user' and self.thread_id:
            payload['thread_id'] = self.thread_id
        try:
            await self.comm.send_to_agent(self.user_id, payload)
            return True
        except Exception as e:
            logger.warning(f"发送消息失败 (user={self.user_id}): {e}")
            return False

    async def _send_ai_message(self, msg, sent_ids: set) -> bool:
        """去重并发送 AI 消息到当前用户。返回 True 表示实际发送了。"""
        msg_id = getattr(msg, 'id', None) or f"hash_{_stable_hash(msg.content)}"
        if msg.type == "ai" and msg.content and msg_id not in sent_ids:
            sent_ids.add(msg_id)
            await self._safe_send(msg.content)
            return True
        return False

    def _build_system_prompt(self):
        prompt = build_brain_system_prompt(self.agent_id)
        if self.agent_context:
            prompt += self.agent_context
        return prompt

    def _load_agent_context(self) -> str:
        """加载 Agent 专属上下文文件（类似 Claude Code 的 CLAUDE.md）。

        文件路径：agent/agent_context/{agent_id}.md
        如果文件不存在则返回空字符串。
        """
        if not self.agent_id:
            return ""
        context_dir = Path(__file__).parent / "agent_context"
        context_file = context_dir / f"{self.agent_id}.md"
        if context_file.exists():
            try:
                content = context_file.read_text(encoding='utf-8')
                logger.info(f"已加载 Agent 上下文文件: {context_file}")
                return f"\n\n## Agent 专属上下文\n\n{content}"
            except Exception as e:
                logger.warning(f"读取 Agent 上下文文件失败 ({context_file}): {e}")
        return ""


    def _build_group_context_prompt(self) -> str:
        """构建群聊上下文提示，用于注入到系统提示词中。"""
        gc = self.group_context
        if not gc:
            return ""
        room_id = gc.get("room_id", "未知")
        members = gc.get("members", [])
        members_str = "、".join(members) if members else "未知"
        return (
            f"\n\n[群聊上下文]"
            f"\n你正在群聊「{room_id}」中，当前群成员：{members_str}。"
            f"\n群聊规则："
            f"\n1. 只回复与你相关、或明确 @你 的消息，其他消息静默忽略；"
            f"\n2. 回复时必须使用 send_group_message 工具，room_id 为「{room_id}」；"
            f"\n3. 禁止使用 send_to_agent 私聊群成员来绕过群聊；"
            f"\n4. 回复应简洁专业，面向全体群成员。"
        )

    @staticmethod
    def _load_skill_content(skill_name: str, source: str) -> str | None:
        """从磁盘加载技能文件完整内容（优先最新版本）"""
        try:
            if source == 'builtin':
                path = Path(__file__).parent / "skills" / skill_name / "SKILL.md"
            else:
                ver = get_skill_latest_version(skill_name)
                if ver:
                    path = get_skill_file_path(skill_name, ver)
                else:
                    path = SKILLS_DIR / f"{skill_name}.md"
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_skill_lessons(content: str) -> str:
        """从技能文档中提取注意事项和反思优化部分（即'踩过的坑'）"""
        parts = []
        for header in ['执行步骤', '注意事项', '反思与优化', '常见错误', '关键要点']:
            pattern = rf'#{{1,4}}\s+{header}\s*\n(.*?)(?=\n#{{1,4}}\s|\Z)'
            m = re.search(pattern, content, re.DOTALL)
            if m:
                text = m.group(1).strip()
                if len(text) > 500:
                    text = text[:500] + "..."
                parts.append(f"**{header}**: {text}")
        return "\n".join(parts) if parts else ""

    async def _sanitize_checkpoint(self, config: dict) -> bool:
        """检查并修复 checkpoint 中的悬空 tool_calls。

        当用户中断任务（停止按钮/新消息取消）时，checkpoint 可能保存了 AI 的
        tool_calls 消息但对应工具尚未响应，导致 LLM API 报 400 错误：
        "insufficient tool messages following tool_calls message"

        此方法检测该情况并移除悬空的 tool_calls 消息。
        返回 True 表示执行了修复。
        """
        try:
            state = await self.agent.aget_state(config)
            if not state or not state.values:
                return False
            messages = list(state.values.get("messages", []))
            if not messages:
                return False

            # 收集所有 tool 消息已响应的 tool_call_id
            responded_ids = set()
            for msg in messages:
                tc_id = getattr(msg, 'tool_call_id', None)
                if tc_id:
                    responded_ids.add(tc_id)

            # 从后往前找第一个包含悬空 tool_calls 的 AI 消息
            fixed = False
            clean_messages = list(messages)
            for i in range(len(clean_messages) - 1, -1, -1):
                msg = clean_messages[i]
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_call_ids = {
                        tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', '')
                        for tc in msg.tool_calls
                    }
                    pending = tool_call_ids - responded_ids
                    if pending and tool_call_ids and pending == tool_call_ids:
                        # 所有 tool_calls 都没有响应，移除整条 AI 消息
                        clean_messages.pop(i)
                        fixed = True
                        logger.info(f"移除悬空 tool_calls 消息 (ids={pending})")
                    elif pending:
                        # 部分有响应，移除无响应的 tool_call（保守处理：移除整条消息）
                        clean_messages.pop(i)
                        fixed = True
                        logger.info(f"移除部分悬空 tool_calls 消息 (pending={pending})")
                    # 一旦处理了一个悬空消息就停止（只修复最末尾的）
                    if fixed:
                        break

            if fixed:
                await self.agent.aupdate_state(config, {"messages": clean_messages})
                logger.info(f"已修复 checkpoint {config.get('configurable', {}).get('thread_id', '?')}")
            return fixed
        except Exception:
            logger.warning("检查 checkpoint 有效性失败", exc_info=True)
            return False

    async def _auto_create_skill(self, plan_id: str, subtask_id: str):
        """任务审核通过后自动生成技能。"""
        plan = self.orchestration_manager.get_plan(plan_id)
        if not plan:
            return
        st = plan.get_subtask(subtask_id)
        if not st or st.status != SubTaskStatus.APPROVED:
            return

        try:
            task_data = {
                "task_id": f"{plan_id}_{subtask_id}",
                "task_description": st.description,
                "steps": [
                    {"step_description": f"尝试 {i+1}", "result": att.get("result_summary", ""),
                     "tool_calls": []}
                    for i, att in enumerate(st.attempts)
                ] or [{"step_description": "执行", "result": st.result or "", "tool_calls": []}],
                "final_result": "success",
                "user_feedback": st.review_feedback or "审核通过",
            }
            from agent.reflection import reflect_on_task, create_skill_from_reflection
            reflection = await reflect_on_task(task_data)
            if reflection.get("should_create_skill"):
                await create_skill_from_reflection(task_data, reflection)
                plan.critical_decisions.append(
                    f"子任务 {subtask_id} 已生成技能: {reflection.get('skill_name', 'unknown')}"
                )
                await self.orchestration_manager._save_plan(plan)
                logger.info(f"子任务 {subtask_id} 成功生成技能")
        except Exception as e:
            logger.warning(f"自动创建技能失败 ({subtask_id}): {e}")

    async def _get_relevant_skill_lessons(self, user_input: str) -> str:
        """搜索与用户输入相关的技能，提取经验教训注入上下文。
        程序化强制执行，不依赖 Agent 自觉调用 search_skills。"""
        try:
            collection = get_skill_collection()
            if not collection:
                return ""

            results = collection.query(query_texts=[user_input], n_results=3)
            if not results.get('metadatas') or not results['metadatas'][0]:
                return ""

            # 提取相似度距离，过滤低相关性的结果
            # cosine 距离：0=完全相同, 1=正交无关, 2=完全相反。阈值 0.5 以下视为相关
            distances = results.get('distances', [[]])[0] if results.get('distances') else []

            lessons = []
            seen = set()
            for idx, meta in enumerate(results['metadatas'][0]):
                # 检查相似度：距离超过阈值则跳过
                if distances and idx < len(distances) and distances[idx] > 0.5:
                    logger.debug(f"技能 {meta.get('skill_name', '?')} 距离={distances[idx]:.3f} 超过阈值，跳过")
                    continue
                skill_name = meta.get('skill_name', '')
                source = meta.get('source', 'learned')
                if not skill_name or skill_name in seen:
                    continue
                seen.add(skill_name)

                content = self._load_skill_content(skill_name, source)
                if not content:
                    continue

                extracted = self._extract_skill_lessons(content)
                if extracted:
                    lessons.append(f"### {skill_name}\n{extracted}")
                    if len(lessons) >= 2:
                        break

            if not lessons:
                return ""

            return "\n\n[相关经验] 以下是你过去处理类似任务时积累的经验教训，请在执行时特别注意避开已知的坑：\n\n" + "\n\n".join(lessons)
        except Exception as e:
            logger.warning(f"获取相关技能经验失败: {e}")
            return ""

    async def update_memory(self, user_id: str, user_input: str, thread_id: str):
        """静默更新指定线程的记忆"""
        await self.process(user_id, user_input, thread_id_override=thread_id, silent=True)

    async def _get_clarification_context(self, thread_id: str = None, max_messages: int = 10) -> str:
        """从当前会话 checkpoints 中提取最近对话，为澄清判断提供上下文。

        使 LLM 能理解"那个文件"、"加个功能"等指代词所指的具体内容，
        避免对同一对话中的自然延续重复追问。
        """
        effective_id = thread_id or self.thread_id
        if not hasattr(self, 'agent') or not effective_id:
            return "（无上文对话记录）"
        try:
            config = {"configurable": {"thread_id": effective_id}}
            state = await self.agent.aget_state(config)
            if not state or not state.values:
                return "（无上文对话记录）"
            messages = state.values.get("messages", [])
            if not messages:
                return "（无上文对话记录）"

            # 取最近 N 条 human/ai 消息，格式化为简洁对话记录
            recent = []
            for msg in messages[-max_messages:]:
                role = msg.type
                content = msg.content if msg.content else ""
                if role == "human":
                    recent.append(f"用户: {content}")
                elif role == "ai":
                    # AI 消息可能很长，截取前 300 字符
                    short = content[:300] + ("..." if len(content) > 300 else "")
                    recent.append(f"助手: {short}")
            if not recent:
                return "（无上文对话记录）"
            return "\n".join(recent)
        except Exception as e:
            logger.debug(f"获取澄清上下文失败: {e}")
            return "（获取上文对话失败）"

    async def clarify(self, user_id: str, user_input: str, thread_id: str = None, max_rounds: int = 5) -> tuple:
        """多轮澄清循环。
        返回: (clarified_input: str, is_clear: bool)

        使用轻量模型判断用户目标的模糊度。若模糊则生成追问，
        每轮更新上下文直到明确或达到最大轮次。澄清环节不计入 TDP 轮次。
        """
        # 初始化或获取澄清状态
        if not hasattr(self, '_clarify_state') or self._clarify_state is None:
            self._clarify_state = {
                "round": 0,
                "history": [],
                "original_input": user_input,
                "conversation_context": await self._get_clarification_context(thread_id),
            }

        state = self._clarify_state

        # 若上一轮有未回答的追问，则当前 user_input 就是用户的回答
        if state["history"]:
            last_entry = state["history"][-1]
            if not last_entry.get("answer"):
                last_entry["answer"] = user_input

        state["round"] += 1

        # 构建历史字符串
        history_str = ""
        for entry in state["history"]:
            history_str += f"第{entry['round']}轮——问: {entry['question']}\\n答: {entry['answer']}\\n"

        # 调用轻量模型判断
        prompt = CLARIFICATION_PROMPT.format(
            conversation_context=state.get("conversation_context", "（无上文对话记录）"),
            user_input=state["original_input"],
            history=history_str if history_str else "（首次澄清）",
        )

        try:
            response = await call_big_model_chat(
                prompt,
                model=config.model.memory_extraction_model,
                temperature=0.2,
                is_json=True,
            )
            content = response["choices"][0]["message"]["content"]
            # 清理可能的 markdown 代码块包装
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            result = json.loads(content)
        except Exception as e:
            logger.warning(f"澄清判断失败: {e}")
            # 降级：当作已明确，直接通过
            self._clarify_state = None
            return user_input, True

        is_clear = result.get("clear", True)
        questions = result.get("questions", [])
        assumption = result.get("assumption", "")
        clarified_goal = result.get("clarified_goal", "")

        if is_clear:
            self._clarify_state = None
            # 用 LLM 合成的完整目标描述替代原始输入，避免丢失澄清内容
            final_goal = clarified_goal or user_input
            logger.info(f"澄清完成: 用户目标已明确，共 {state['round']} 轮")
            return final_goal, True

        # 达到上限，使用假设
        if state["round"] >= max_rounds:
            self._clarify_state = None
            fallback = assumption or clarified_goal or user_input
            await self._safe_send(
                f"经过 {state['round']} 轮澄清，我将基于以下理解继续：\n\n"
                f"**{fallback}**\n\n如有偏差，请随时纠正。"
            )
            logger.info(f"澄清达上限 ({max_rounds} 轮)，使用假设: {fallback[:100]}")
            return fallback, False

        # 发送追问
        questions_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        await self._safe_send(
            f"为了更好地帮你完成任务，我需要确认以下几点：\n\n"
            f"{questions_text}\n\n请逐一回答。"
        )

        # 记录本轮问题（回答留空，下一轮由 user_input 回填）
        state["history"].append({
            "round": state["round"],
            "question": " | ".join(questions),
            "answer": "",
        })

        return "", False  # is_clear=False，调用方应提前返回

    def _inject_time_context(self, text: str) -> str:
        """将相对时间词替换为带具体日期的标注，帮助模型理解当前时间。
        使用正则负向后顾避免误替换（如"如今天气"中的"今天"不会被替换）。"""
        now = datetime.now()
        # 负向后顾字符集：避免"今"前接如/而/至/当/现/迄/古（如今、而今、至今...）
        #                   避免"明"前接说/证/声/表/聪/文/光/发（说明、证明、声明...）
        #                   避免"昨"前接其他字、避免"前/后"与其他字组合
        rules = [
            (r'(?<![如今而至当现迄古今])今天', 0),
            (r'(?<![昨])昨天', -1),
            (r'(?<![前])前天', -2),
            (r'(?<![后])后天', 2),
            (r'(?<![说证声表聪文明光发])明天', 1),
        ]
        for pattern, offset in rules:
            date_str = (now + timedelta(days=offset)).strftime('%Y%m%d')
            text = re.sub(pattern, rf'\g<0>（{date_str}）', text)
        return text
        
    async def process(self, user_id: str, user_input: str, image_data: str = None, new_thread: bool = False,
                      thread_id_override: str = None, silent: bool = False,
                      group_context: dict = None, tdp_notification: dict = None) -> str:
        async with self._process_lock:
            # ── 在锁内原子设置 TDP 通知，消除竞态条件 ──
            if tdp_notification is not None:
                self._tdp_notification = tdp_notification
            self.is_busy = True
            self._current_task = asyncio.current_task()
            await self._notify_status("busy")
            self.group_context = group_context  # 群聊上下文，注入系统提示词
            user_input = self._inject_time_context(user_input)
            try:
                self.user_id = user_id
                effective_thread_id = thread_id_override if thread_id_override is not None else self.thread_id

                # ── 智能澄清循环：仅 super_user 的普通文本消息触发 ──
                # 被打断后的新消息视为明确指令，跳过澄清直接执行
                if self._skip_clarify:
                    self._skip_clarify = False
                elif self.user_id == 'super_user' and not tdp_notification and not image_data:
                    clarified_input, is_clear = await self.clarify(user_id, user_input, thread_id=effective_thread_id)
                    if not clarified_input and not is_clear:
                        # 仍在澄清中，已发送追问，等待用户回复
                        return ""
                    user_input = clarified_input

                if self.user_id != 'super_user':
                    intent_data = IntentType.COMPLEX_TASKS.value
                else:
                    intent_data = await self._classify_intent(user_input)
                response = await self._handle_intent(intent_data, user_id, user_input, image_data, new_thread,
                                                     effective_thread_id, silent)
                return response if not silent else ""
            except asyncio.CancelledError:
                await self._safe_send("⏹ 任务已停止。")
                return "" if not silent else None
            finally:
                self._current_task = None
                self.is_busy = False
                self.last_run_time = datetime.now()
                await self._notify_status("idle")
            # 注意：不在这里清除 group_context，因为 agent 消息走延迟处理路径，
            # 实际处理在 delayed_process 中异步发生，此时 group_context 仍需保留。
            # group_context 在 _process_agent_message / _handle_with_agent 使用后由下次
            # process() 调用覆盖，或在 _process_agent_message 中主动清除。

    async def recover_state(self):
        """从数据库恢复活跃的工单和编排计划（进程重启后调用）。"""
        await self.delegation_manager.load_active()
        await self.orchestration_manager.load_active()
        # 如果恢复了活跃工单，启动超时检测循环
        if self.delegation_manager._by_id:
            self._ensure_timeout_loop_started()

    async def _notify_status(self, status: str):
        """通知前端智能体状态变化（busy / idle）"""
        if self.comm:
            try:
                await self.comm.send({
                    "type": "agent_status",
                    "agent_id": self.agent_id,
                    "status": status,
                })
            except Exception as e:
                logger.warning(f"通知状态变化失败 ({status}): {e}")

    def stop_current_task(self):
        """取消当前正在执行的任务（由外部 stop_task 消息触发）"""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            logger.info(f"Task cancelled for agent {self.agent_id}")
        # 清除澄清状态，避免新消息被误判为澄清轮次
        self._clarify_state = None
        # 被打断后的下一条消息跳过澄清，直接执行
        self._skip_clarify = True

    def _ensure_timeout_loop_started(self):
        """惰性启动工单超时检测后台循环（仅首次调用生效）"""
        if not self._timeout_loop_started:
            self._timeout_loop_started = True
            self.schedule_background_task(self._ticket_timeout_loop())

    async def _ticket_timeout_loop(self, interval_seconds: int = None):
        if interval_seconds is None:
            interval_seconds = config.agent.ticket_timeout_interval
        """后台循环：定期检查工单是否空闲超时并自动终止。"""
        from agent.delegation import DEFAULT_IDLE_TIMEOUT_MINUTES
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                timed_out = self.delegation_manager.check_timeouts(DEFAULT_IDLE_TIMEOUT_MINUTES)
                for ticket in timed_out:
                    logger.info(f"工单 {ticket.ticket_id} 空闲超时，自动终止")
                    # ── 编排回调：标记子任务失败 ──
                    await self.orchestration_manager.mark_failed(
                        ticket.ticket_id,
                        f"工单超时 ({DEFAULT_IDLE_TIMEOUT_MINUTES}分钟无活动)"
                    )
                    await self._safe_send(
                        f"⏰ 工单 {ticket.ticket_id}（{ticket.description[:50]}...）空闲超时，已自动终止。"
                    )
                    # 通知双方
                    try:
                        other = ticket.issuer if self.agent_id == ticket.assignee else ticket.assignee
                        await self.comm.send_to_agent(other, {
                            "text": f"[TDP] 工单 {ticket.ticket_id} 因 {DEFAULT_IDLE_TIMEOUT_MINUTES} 分钟无活动而自动超时。",
                            "_tdp": "cancel",
                            "ticket_id": ticket.ticket_id,
                        })
                    except Exception:
                        pass
                    try:
                        await self.comm.send_to_agent("super_user", {
                            "text": (
                                f"⏰ [TDP超时] 工单 {ticket.ticket_id} "
                                f"({ticket.issuer} -> {ticket.assignee}) 因 {DEFAULT_IDLE_TIMEOUT_MINUTES} 分钟无活动而超时。"
                            )
                        })
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"工单超时检测异常: {e}")

    def get_thread_id(self, new_thread, chat_id):
        # new_thread 优先级最高：强制开启新对话
        if new_thread:
            # 仅当 thread_id 是前端通过 createConversation 预生成的唯一 ID（含 UUID 后缀）时才保留
            # core.py 的兜底格式 "private_agent_user" 不含 UUID，排除之，避免复用旧 checkpoint
            bare_default = f"private_{chat_id}"
            if self.thread_id and self.thread_id != bare_default and chat_id in self.thread_id:
                logger.debug(f'使用前端指定的新 thread_id: {self.thread_id}')
                self.memory.set_user_metadata(chat_id, "last_thread_id", self.thread_id)
                return
            # 否则生成新 ID
            logger.debug('new_thread: 生成新 ID')
            self.thread_id = f"{chat_id}_{uuid.uuid4()}"
            self.memory.set_user_metadata(chat_id, "last_thread_id", self.thread_id)
            return

        # 如果前端显式传了 thread_id（非 new_thread 情况），直接使用
        if self.thread_id and chat_id in self.thread_id:
            logger.debug(f'使用前端指定的 thread_id: {self.thread_id}')
            self.memory.set_user_metadata(chat_id, "last_thread_id", self.thread_id)
            return

        # 尝试从长期记忆恢复上次的 thread_id
        last_thread = self.memory.get_user_metadata(f'{chat_id}', "last_thread_id")
        if last_thread:
            logger.debug('加载从长期记忆中的last_thread_id')
            self.thread_id = last_thread
        else:
            logger.debug('首次对话，生成新 ID')
            self.thread_id = f"{chat_id}_{uuid.uuid4()}"
            self.memory.set_user_metadata(chat_id, "last_thread_id", self.thread_id)

    async def _detect_reminder_intent(self, user_input: str) -> dict:
        """调用模型判断是否是定时任务，并提取时间和消息"""
        prompt = build_reminder_detection_prompt(user_input)
        try:
            content = await call_big_model_chat(prompt, model=config.model.default_model, temperature=config.model.model_temperature, is_json=True)
            # 提取 choices[0].message.content
            content = content["choices"][0]["message"]["content"]
            # 例如：```json\n{...}\n```
            if content.startswith("```") and content.endswith("```"):
                # 去掉第一行（```json）和最后一行（```）
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            data = json.loads(content)
            # print(f"data={data}, type={type(data)}")
            # 确保字段存在
            if "reminders" not in data:
                data["reminders"] = []
            if "has_other" not in data:
                data["has_other"] = False
            return data
        except Exception as e:
            logger.warning(f"意图识别失败: {e}")
            # 出错时默认当作普通任务处理
            return {"reminders": [], "has_other": True}
    
    async def _classify_intent(self, user_input: str) -> dict:
        """调用大模型进行意图分类，返回包含intent和可能参数的字典，先使用关键词进行提取，不行再做大模型意图识别"""
        # 时间词正则（包含自然语言）
        time_patterns = [
            r'(\d+)\s*分钟[后内]',
            r'(\d+)\s*小时[后内]',
            r'(\d+)\s*天[后内]',
            r'明天', r'后天', r'今天', r'下周', r'下个月',
            r'(\d{1,2})点',
            r'(\d+)\s*秒[后内]',
        ]
        has_time = any(re.search(p, user_input) for p in time_patterns)
        
        # 设置提醒：必须有时间词 + 提醒/记/闹钟
        if has_time and ('提醒' in user_input or '记' in user_input or '闹钟' in user_input):
            return IntentType.SET_REMINDER.value
        
        # 查询提醒：精确的关键词（避免误匹配）
        query_keywords = ['我的提醒', '查看提醒', '有哪些提醒', '未到期的提醒', '提醒列表', '提醒我什么', '待办',
                          '提醒一下']
        if any(kw in user_input for kw in query_keywords):
            return IntentType.QUERY_REMINDER.value
        
        # 构建意图选项字符串
        intent_lines = []
        intent_str = ""
        for intent, desc in INTENT_DESCRIPTIONS.items():
            line = f"- {intent.value}: {desc}"
            intent_str += f'，{intent.value}'
            intent_lines.append(line)
        intent_options = "\n".join(intent_lines)
        
        prompt = build_intent_classification_prompt(user_input, intent_options, intent_str)
        # print(prompt)
        try:
            response = await call_big_model_chat(prompt, model=config.model.intent_model, temperature=config.model.model_temperature)
            content = response["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            logger.warning(f"意图分类失败: {e}")
            return IntentType.COMPLEX_TASKS.value
    
    async def _handle_intent(self, intent: str, user_id: str, user_input: str, image_data: str = None,
                             new_thread: bool = False, thread_id: str = None, silent: bool = False) -> str:
        # 临时覆盖 self.thread_id
        original_thread_id = self.thread_id
        if thread_id:
            self.thread_id = thread_id
        elif new_thread:
            self.thread_id = None  # 清除残留旧值，由 get_thread_id 生成新 ID
        try:
            if intent == IntentType.SET_REMINDER.value:
                reminders = await self._detect_reminder_intent(user_input)
                reminder_list = reminders.get('reminders', [])
                has_other = reminders.get('has_other', False)

                if not reminder_list:
                    # 未提取到提醒内容：如果夹杂其他意图，交给 Agent 正常处理；
                    # 纯粹无法解析时才返回错误。
                    return await self._handle_with_agent(user_input, image_data, new_thread, silent) if has_other \
                        else "未能理解提醒的时间和内容，请重新描述。"

                reminder_response = await self._handle_set_reminder(reminders)
                if has_other:
                    # 提醒之外还有其他内容，继续交给 Agent 处理
                    await self._handle_with_agent(user_input, image_data, new_thread, silent)
                return reminder_response
            elif intent == IntentType.QUERY_REMINDER.value:
                return await self._handle_query_reminder(user_id)
            else:
                return await self._handle_with_agent(user_input, image_data, new_thread, silent)
        finally:
            self.thread_id = original_thread_id

    async def _handle_set_reminder(self, reminders):
        responses = []
        # 处理所有提醒
        scheduler = get_scheduler()
        pool = get_pool()
        for r in reminders.get('reminders', []):
            time_str = r.get("time")
            message = r.get("message")
            if time_str and message:
                # 以 UTC 当前时间为基准解析时间
                remind_time = dateparser.parse(
                    time_str,
                    settings={
                        'PREFER_DATES_FROM': 'future',
                        'RELATIVE_BASE': datetime.now(timezone.utc)
                    }
                )
                if not remind_time:
                    return f" 无法解析时间：{time_str}"
                # 转换为 UTC naive datetime（移除时区信息）
                if remind_time.tzinfo is not None:
                    remind_time = remind_time.astimezone(timezone.utc).replace(tzinfo=None)
                
                if remind_time:
                    from sqlalchemy.exc import OperationalError
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            job_id = f"reminder_{self.user_id}_{int(remind_time.timestamp())}"
                            scheduler.add_job(
                                send_reminder,
                                trigger='date',
                                run_date=remind_time,
                                args=[self.user_id, message],
                                id=job_id,
                                replace_existing=True,
                                misfire_grace_time=30   # 如果在 misfire_grace_time 时间差内，依然运行
                            )
                            # 2. 插入 reminders 表
                            async with pool.connection() as conn:
                                async with conn.cursor() as cur:
                                    await cur.execute(
                                        "INSERT INTO reminders (user_id, reminder_time, message) VALUES (%s, %s, %s)",
                                        (self.user_id, remind_time, message)
                                    )
                            responses.append(f"在 {remind_time.strftime('%Y-%m-%d %H:%M:%S')} 提醒你 {message}")
                            break
                        except OperationalError as e:
                            if attempt == max_retries - 1:
                                raise
                            logger.warning(f"数据库连接错误，重试 {attempt + 1}/{max_retries}...")
                            await asyncio.sleep(2**attempt)
                        except Exception as e:
                            logger.error(f"设置提醒失败 ({type(e).__name__}): {e}")
                            responses.append(f"设置提醒失败: {e}")
                            break
                else:
                    responses.append(f"出错了，无法理解这个时间：{time_str}")
            else:
                responses.append("出错了 提醒信息不完整")
        return "好的，我会在" + "，".join(responses)
    
    async def _handle_query_reminder(self, user_id: str) -> str:
        from agent.db import get_pool
        from psycopg.rows import dict_row
        pool = get_pool()
        try:
            async with pool.connection() as conn:
                # 标记已过期的提醒（使用 UTC 时间）
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE reminders SET triggered = TRUE WHERE user_id = %s AND NOT triggered AND reminder_time <= (NOW() AT TIME ZONE 'UTC')",
                        (user_id,)
                    )
                    updated = cur.rowcount
                    if updated > 0:
                        logger.info(f"已标记 {updated} 条过期提醒")
                
                # 查询未触发且未过期的提醒
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT reminder_time, message FROM reminders WHERE user_id = %s AND NOT triggered AND reminder_time > (NOW() AT TIME ZONE 'UTC') ORDER BY reminder_time",
                        (user_id,)
                    )
                    rows = await cur.fetchall()
        except Exception as e:
            return f"查询提醒时出错: {e}"
        
        if not rows:
            return "您当前没有未到期的提醒。"
        result = "您当前的提醒：\n"
        for row in rows:
            dt = row['reminder_time'].strftime('%Y-%m-%d %H:%M:%S')
            result += f"- {dt} UTC：{row['message']}\n"
        return result
     
    async def _astream_with_timeout(self, input_state, runnable_config, stream_mode="updates",
                                     per_event_timeout: float = None,
                                     idle_timeout: float = None,
                                     max_total_timeout: float = None):
        """对 agent.astream 的每次 __anext__ 调用增加超时保护，避免流永久挂起。

        三层超时设计（按优先级）：
        1. per_event_timeout (180s): 单次 __anext__() 最大等待时间，覆盖慢 LLM/tool 调用
        2. idle_timeout (300s): 持续无事件的空闲时间，每次 yield 事件后重置
        3. max_total_timeout (1800s): 整条流的最大生存时间，绝对安全网

        核心理念：只要在持续产出事件（工具调用/LLM响应），就视为"健康推进"，
        idle_timer 不断刷新。只有彻底无响应时才判定为挂死。

        使用 asyncio.wait 代替 wait_for，避免 Python 3.12 将内部 CancelledError 掩码为 TimeoutError。
        """
        if per_event_timeout is None:
            per_event_timeout = config.agent.stream_per_event_timeout
        if idle_timeout is None:
            idle_timeout = config.agent.stream_idle_timeout
        if max_total_timeout is None:
            max_total_timeout = config.agent.stream_max_timeout
        agen = self.agent.astream(input_state, runnable_config, stream_mode=stream_mode)
        total_deadline = asyncio.get_event_loop().time() + max_total_timeout
        last_event_time = asyncio.get_event_loop().time()

        while True:
            now = asyncio.get_event_loop().time()

            # ── 检查总安全网 ──
            if now >= total_deadline:
                elapsed = int(max_total_timeout)
                raise asyncio.TimeoutError(
                    f"astream ({stream_mode}) 达到最大总时长限制 ({elapsed}s)，任务可能过于复杂，请拆分后重试。"
                )

            # ── 检查空闲超时（自上次事件以来的沉寂时间）──
            idle_elapsed = now - last_event_time
            if idle_elapsed >= idle_timeout:
                raise asyncio.TimeoutError(
                    f"astream ({stream_mode}) 已 {int(idle_elapsed)}s 无响应（空闲阈值 {idle_timeout}s），判定为挂死。"
                )

            # ── 本次等待的事件级超时：取 per_event 和 剩余空闲时间的最小值 ──
            event_wait = min(per_event_timeout, idle_timeout - idle_elapsed)

            next_task = asyncio.ensure_future(agen.__anext__())
            try:
                done, _ = await asyncio.wait([next_task], timeout=event_wait)
            except asyncio.CancelledError:
                next_task.cancel()
                raise

            if not done:
                next_task.cancel()
                raise asyncio.TimeoutError(
                    f"astream ({stream_mode}) 等待下一个事件超过 {event_wait:.0f}s，超时。"
                )

            exc = next_task.exception()
            if exc is not None:
                if isinstance(exc, StopAsyncIteration):
                    break
                raise exc

            # 事件产出 → 刷新空闲计时器
            last_event_time = asyncio.get_event_loop().time()
            yield next_task.result()

    def _enqueue_agent_message(self, user_input: str, image_data: str = None, new_thread: bool = False):
        """将 agent-to-agent 消息加入延迟合并缓冲区。"""

        async def on_process(full_text: str, saved_ctx: dict):
            if saved_ctx:
                self.group_context = saved_ctx
            if not self._agent_message_needs_response(full_text):
                logger.debug(f"跳过 agent-to-agent 状态消息 (from={self.user_id}): {full_text[:100]}")
                if self.thread_id:
                    self.task_buffer.add_step(self.thread_id, f"来自 {self.user_id} 的状态更新", full_text[:300])
                return
            await self._process_agent_message(full_text, image_data, new_thread)

        self.msg_buffer.enqueue(self.user_id, user_input, self.group_context, on_process)

    def _check_conversation_limits(self) -> bool:
        """检查对话限制：super_user 重置所有限制，硬截断检查。
        返回 True 表示应中止当前消息处理。
        """
        if self.user_id == 'super_user':
            self.conversation_tracker.reset_all_for(self.agent_id)
            return False
        if self.conversation_tracker.is_capped(self.agent_id, self.user_id):
            logger.info(f"对话 {self.agent_id}<->{self.user_id} 已截断，丢弃收到的消息")
            return True
        return False

    async def _handle_with_agent(self, user_input: str, image_data: str = None, new_thread: bool = False,
                                    silent: bool = False):
        # ---- P0: 对话限制检查 ----
        if self._check_conversation_limits():
            return

        # 如果是 Agent 之间的对话，委托给 MessageBuffer 延迟合并
        if self.user_id != 'super_user':
            # 惰性启动工单超时循环
            self._ensure_timeout_loop_started()

            # ── TDP 通知处理 ──
            should_skip, user_input = await self._handle_tdp_notification(user_input)
            if should_skip:
                return

            # ── 编排模式守卫：正在编排时，非 TDP 消息也禁止唤醒 LLM ──
            # 例外：如果 self 是消息发送方的被委托方（assignee），需要 LLM 来决定接受/拒绝
            if self._is_orchestrating:
                ticket = self.delegation_manager.get_active_ticket(self.agent_id, self.user_id)
                is_assignee = ticket and ticket.assignee == self.agent_id
                if not is_assignee:
                    logger.info(
                        f"编排模式: 跳过 agent-to-agent 消息的 LLM 处理 "
                        f"(from={self.user_id}, agent={self.agent_id})"
                    )
                    if self.thread_id:
                        self.task_buffer.add_step(
                            self.thread_id,
                            f"编排模式 — 来自 {self.user_id} 的消息（已静默）",
                            user_input[:300],
                        )
                    return

            self._enqueue_agent_message(user_input, image_data, new_thread)
            return  # 等待定时器，不立即处理
        
        
        chat_id = f'{self.agent_id}_{self.user_id}'
        self.get_thread_id(new_thread, chat_id)
        
        # ========== 修改：智能创建/更新任务 ==========
        existing_task = self.task_buffer.get_current_task(self.thread_id)
        if existing_task is None:
            self.task_buffer.start_task(self.thread_id, user_input)
        else:
            # 已有任务，只记录步骤并刷新活跃时间
            self.task_buffer.add_step(self.thread_id, "用户继续输入（补充信息或修正指令）", user_input[:500])
        # 保存原始用户输入，用于精确的技能检索
        original_user_input = user_input

        # Build messages with system contexts + skill lessons
        messages = await self._build_messages_for_stream(user_input, image_data, original_user_input)

        # Proactive token budget check (one-time, before the stream loop)
        if self.user_id == 'super_user' and not silent:
            budget_warning = self.context_manager.check_budget(messages)
            if budget_warning:
                await self._safe_send(budget_warning)

        config, sent_ids = await self._prepare_agent_config()
        
        # HITL 循环：初始输入为 messages
        input_state = {"messages": messages}
        command = None
        
        try:
            while True:
                # 如果有恢复命令，则使用 Command 作为输入
                if command:
                    input_state = Command(resume=command)

                async for event in self._astream_with_timeout(input_state, config, stream_mode="updates"):
                    # 中断可能在 event 的某个节点值中
                    interrupt_data = None
                    for key, node_output in event.items():
                        if node_output is None:
                            continue
                        if "__interrupt__" == key:
                            interrupt_data = node_output
                            break
                    if interrupt_data:
                        # 处理中断
                        decisions = await self._process_interrupts(interrupt_data)
                        if decisions:
                            command = {"decisions": decisions}
                        else:
                            command = None
                        break
                    else:
                        # 正常消息：遍历所有节点输出中的 messages
                        for node_output in event.values():
                            if node_output is None:
                                continue
                            if "messages" in node_output:
                                messages_obj = node_output["messages"]
                                if messages_obj is None:  # 也检查 messages 是否为 None
                                    continue
                                # 处理 Overwrite 对象
                                if hasattr(messages_obj, 'value') and not isinstance(messages_obj, list):
                                    messages_list = messages_obj.value
                                else:
                                    messages_list = messages_obj

                                for msg in messages_list:
                                    await self._send_ai_message(msg, sent_ids)
                                    msg_id = getattr(msg, 'id', None) or f"hash_{_stable_hash(msg.content)}"
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls and self.user_id == 'super_user' and not silent:
                                        for tc in msg.tool_calls:
                                            tc_id = tc.get('id', '') or f"tc_{_stable_hash(str(tc))}"
                                            if tc_id not in sent_ids:
                                                sent_ids.add(tc_id)
                                                tool_name = tc.get("name")
                                                self._in_flight_tools.append(tool_name)
                                                await self._safe_send(
                                                    f"🔧 调用工具: {tool_name}",
                                                    type="tool_call_start",
                                                    tool_name=tool_name,
                                                    tool_args=tc.get("args", {}),
                                                )
                                    elif msg.type == "tool" and self.user_id == 'super_user' and not silent \
                                            and msg_id not in sent_ids and msg_id not in self._sent_tool_msgs:
                                        sent_ids.add(msg_id)
                                        self._sent_tool_msgs.add(msg_id)
                                        tool_name = getattr(msg, 'name', 'unknown_tool')
                                        # 从 in-flight 列表中移除匹配的工具
                                        try:
                                            self._in_flight_tools.remove(tool_name)
                                        except ValueError:
                                            pass  # 可能已被清理
                                        await self._safe_send(
                                            f"🛠️ 工具返回: {msg.content}",
                                            type="tool_call_end",
                                            tool_name=tool_name,
                                        )
                                        self.task_buffer.add_step(
                                            self.thread_id,
                                            f"调用工具： {tool_name}",
                                            msg.content[:500],
                                        )

                else:
                    # 没有中断，流正常结束
                    break
        except asyncio.TimeoutError:
            logger.error("Agent stream timed out in _handle_with_agent")
            if self.user_id == 'super_user' and not silent:
                await self._safe_send("处理超时，请重试或简化请求。")
            await self._cleanup_in_flight_tools()
        except asyncio.CancelledError:
            await self._cleanup_in_flight_tools()
            raise
        except (OSError, ConnectionError) as e:
            logger.error(f"Agent stream network error in _handle_with_agent: {e}")
            if self.user_id == 'super_user' and not silent:
                await self._safe_send(f"网络连接失败：{e}。请检查代理设置或网络连接后重试。")
            await self._cleanup_in_flight_tools()
        except Exception as e:
            logger.error(f"Agent stream error in _handle_with_agent: {type(e).__name__}: {e}", exc_info=True)
            if self.user_id == 'super_user' and not silent:
                await self._safe_send(f"处理请求时遇到错误 ({type(e).__name__})，请重试。")
            await self._cleanup_in_flight_tools()
    
    async def _process_agent_message(self, user_input: str, image_data: str = None, new_thread: bool = False):
        # ---- TDP 优先：使用活跃工单的 thread_id ----
        active_ticket = self.delegation_manager.get_active_ticket(self.agent_id, self.user_id)
        if active_ticket and active_ticket.thread_id:
            self.thread_id = active_ticket.thread_id
            active_ticket.last_activity = time.time()
        else:
            # 兜底：使用旧 tracker 管理的 thread_id
            self.thread_id = self.conversation_tracker.get_or_create_thread_id(self.agent_id, self.user_id)

        # ===== 记录收到的消息（有助于任务活跃检测） =====
        if hasattr(self, 'task_buffer') and self.thread_id:
            self.task_buffer.add_step(
                self.thread_id,
                f"收到来自 {self.user_id} 的消息",
                user_input[:500]
            )

        # ---- P0: 双重门控（TDP + 旧 tracker）----
        if self.conversation_tracker.is_capped(self.agent_id, self.user_id):
            logger.info(f"对话 {self.agent_id}<->{self.user_id} 已被旧 tracker 截断，跳过处理")
            return
        # ----------------------

        config, sent_ids = await self._prepare_agent_config()

        system_contexts = await self._build_system_contexts(image_data)

        messages = []
        for ctx in system_contexts:
            if ctx:
                messages.append({"role": "system", "content": ctx})
        messages.append({"role": "user", "content": user_input})

        input_state = {"messages": messages}
        command = None
        message_sent = False  # 追踪本轮是否发了 AI 消息

        try:
            while True:
                if command:
                    input_state = Command(resume=command)

                async for event in self._astream_with_timeout(input_state, config, stream_mode="values"):
                    if "__interrupt__" in event:
                        interrupts = event["__interrupt__"]
                        decisions = await self._process_interrupts(interrupts)
                        command = decisions if decisions else None
                        break
                    else:
                        if "messages" in event:
                            for msg in event["messages"]:
                                if await self._send_ai_message(msg, sent_ids):
                                    message_sent = True
                                    self.task_buffer.add_step(
                                        self.thread_id,
                                        f"向 {self.user_id} 发送消息",
                                        msg.content[:500]
                                    )
                else:
                    break
        except asyncio.TimeoutError:
            logger.error("Agent stream timed out in _process_agent_message")
        except asyncio.CancelledError:
            raise
        except (OSError, ConnectionError) as e:
            logger.error(f"Agent stream network error in _process_agent_message: {e}")
        except Exception as e:
            logger.error(f"Agent stream error in _process_agent_message: {type(e).__name__}: {e}", exc_info=True)

        # ---- TDP 轮次记录：Agent 发了消息就记录一轮 ----
        if message_sent and active_ticket and active_ticket.is_active:
            try:
                self.delegation_manager.record_round(active_ticket.ticket_id)
            except TicketError:
                pass

        # ---- 对话终止检测（语义停滞）----
        if active_ticket and active_ticket.is_active and active_ticket.round_count >= 3:
            should_end = await self._should_terminate_conversation(active_ticket.thread_id)
            if should_end:
                try:
                    self.delegation_manager.cancel_ticket(
                        active_ticket.ticket_id,
                        "语义停滞检测：对话陷入重复循环",
                        "system"
                    )
                    await self._safe_send(
                        f"⚠️ 工单 {active_ticket.ticket_id} 检测到对话停滞，已自动终止。"
                        f"如需继续，请创建新工单。"
                    )
                except TicketError:
                    pass

    async def _handle_image(self, image_data: str) -> str | None:
        """处理图片输入，返回视觉模型的结果。失败时返回 None。"""
        model = model_config.get_model("vision")
        content = [
            {"type": "text", "text": "读取图片的内容，尽可能完整地描述出图片的内容。"},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]
        messages = [{"role": "user", "content": content}]
        try:
            response = await model.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            return None
        
    async def _process_interrupts(self, interrupts):
        """处理中断列表，返回决策列表"""
        decisions = []
        for interrupt in interrupts:
            value = interrupt.value
            action_requests = value.get("action_requests", [])
            review_configs = value.get("review_configs", [])
            for action in action_requests:
                tool_name = action.get("name")
                tool_args = action.get("args")
                config = next((cfg for cfg in review_configs if cfg["action_name"] == tool_name), {})
                allowed = config.get("allowed_decisions", ["approve", "reject"])
                # 为每个工具生成唯一 ID
                tool_call_id = action.get("id") or f"{tool_name}_{uuid.uuid4()}"
                # 发送审批请求给前端，带上 from 字段（当前 Agent ID）
                msg = {
                    "type": "approval_request",
                    "from": self.agent_id,  # 关键：指明发起者
                    "tool": tool_name,
                    "args": tool_args,
                    "allowed": allowed,
                    "tool_call_id": tool_call_id,
                }
                try:
                    await self.comm.send_to_agent(self.user_id, msg)
                except Exception as e:
                    logger.warning(f"发送审批请求失败: {e}")
                # 等待用户决策
                decision = await self._wait_for_user_decision(tool_call_id)
                decisions.append(decision)
        return decisions
    
    async def _wait_for_user_decision(self, tool_call_id: str):
        future = asyncio.get_event_loop().create_future()
        self._pending_approvals[tool_call_id] = future
        try:
            decision = await asyncio.wait_for(future, timeout=config.agent.approval_timeout)
            return decision
        except asyncio.TimeoutError:
            return {"type": "reject"}
        except asyncio.CancelledError:
            return {"type": "reject"}
    
    async def _complete_approval(self, tool_call_id: str, decision):
        if tool_call_id in self._pending_approvals:
            self._pending_approvals[tool_call_id].set_result(decision)
            del self._pending_approvals[tool_call_id]
        else:
            logger.warning(f"No pending approval for {tool_call_id}")
        
    def schedule_background_task(self, coro) -> asyncio.Task:
        """创建带生命周期跟踪的后台任务，异常自动记日志。"""

        async def _wrapper():
            try:
                await coro
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"后台任务异常: {type(e).__name__}: {e}", exc_info=True)

        task = asyncio.create_task(_wrapper())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def close(self):
        self.msg_buffer.cancel_all()
        # 取消所有活跃工单
        self.delegation_manager.cancel_all_for_agent(self.agent_id, "Agent shutting down")
        # 触发闲置对话的记忆提取（在取消后台任务之前）
        for thread_id, task in list(self.task_buffer.buffers.items()):
            if task.get("status") == "in_progress":
                idle = time.time() - task.get("last_active_time", 0)
                if idle > 3600:  # 1小时无活动
                    self.schedule_background_task(
                        trigger_memory_extraction(thread_id, self.user_id, self.agent_id)
                    )
                    self.task_buffer.finish_task(thread_id, "timeout", user_feedback="任务因长时间无活动而终止")
        # 等待记忆提取完成
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        # 取消可能残留的后台任务
        for task in list(self._bg_tasks):
            task.cancel()
        await close_browser_session()
            
    async def _process_reflection(self, task_data):
        """异步处理反思 + 触发对话记忆提取（在任务完成时）。"""
        await asyncio.to_thread(submit_task_for_reflection, task_data)
        # 任务完成后，从对话中提取长期记忆（facts + events）
        if self.thread_id:
            self.schedule_background_task(
                trigger_memory_extraction(self.thread_id, self.user_id, self.agent_id)
            )
        
    async def _load_sent_ids_from_checkpoint(self, config):
        """从 checkpoint 加载已发送的消息 ID 集合（含 tool_call ID）。"""
        try:
            state = await self.agent.aget_state(config)
            if state and state.values and "messages" in state.values:
                ids = set()
                for msg in state.values["messages"]:
                    msg_id = getattr(msg, 'id', None)
                    if msg_id:
                        ids.add(msg_id)
                    elif hasattr(msg, 'content') and msg.content:
                        # 备用方案：内容哈希
                        ids.add(_stable_hash(msg.content))
                    # 恢复 tool_call ID，防止对话恢复时重发 tool_call_start 事件
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tc_id = tc.get('id', '') or f"tc_{_stable_hash(str(tc))}"
                            ids.add(tc_id)
                return ids
        except Exception as e:
            logger.warning(f"加载已发送ID失败: {e}")
        return set()
    
    async def _get_conversation_history_for_termination(self, thread_id: str, max_messages: int = 20):
        """
        从 checkpoint 中提取对话历史，返回适合终止判断的格式：
        [{"speaker": "AgentA", "message": "..."}, ...]
        规则：将 AI 消息视为当前 Agent（self.agent_id），将 Human 消息视为对方 Agent。
        """
        configs = {"configurable": {"thread_id": thread_id}}
        try:
            state = await self.agent.aget_state(configs)
        except Exception as e:
            logger.warning(f"读取对话历史失败 (thread={thread_id}): {e}")
            return []
        if not state or not state.values:
            return []

        messages = state.values.get("messages", [])
        if not messages:
            return []

        # 对方 Agent ID：在 agent-to-agent 上下文中，self.user_id 就是对方
        other_agent_id = self.user_id if self.user_id != 'super_user' else "user"

        history = []
        for msg in messages[-max_messages:]:
            if msg.type == "ai":
                speaker = self.agent_id
            elif msg.type == "human":
                speaker = other_agent_id
            else:
                continue  # 忽略 system/tool 消息
            content = msg.content if msg.content else ""
            history.append({"speaker": speaker, "message": content})
        return history
    
    async def _should_terminate_conversation(self, thread_id: str) -> bool:
        """
        判断指定 thread_id 的对话是否应终止。
        返回 True 表示应终止，False 表示可继续。
        """
        # ===== 新增：如果有进行中且未超时的任务，不允许终止 =====
        if hasattr(self, 'task_buffer') and self.task_buffer.has_active_task(thread_id, min_rounds=8, max_idle_seconds=600):
            logger.debug(f"Thread {thread_id} has active task, skip termination")
            return False
        
        # 限频缓存（在类中增加属性 self._termination_cache: dict）
        now = time.time()
        if thread_id in self._termination_cache:
            result, timestamp = self._termination_cache[thread_id]
            if now - timestamp < 30:  # 30 秒内复用结果
                return result
        
        history = await self._get_conversation_history_for_termination(thread_id, max_messages=20)
        if len(history) < 10:  # 对话太短，不判断
            return False
            
        # 获取任务信息
        task = self.task_buffer.get_current_task(thread_id)
        task_context = ""
        if task:
            steps = task.get("steps", [])
            completed = len(steps)
            description = task.get("task_description", "未知")
            task_context = (f"当前进行中的任务：{description}，已执行 {completed} 步。"
                            f"请根据对话历史判断任务是否仍在有序推进，或者已经陷入无意义的重复循环。"
                            f"如果对话明显重复、无新信息，即使任务未完成也应该终止。")
        
        # 构造提示词
        prompt = build_termination_judge_prompt(history, task_context)
        try:
            response = await call_big_model_chat(
                prompt,
                model=config.model.default_model,
                temperature=0.2,
                is_json=True
            )
            content = response["choices"][0]["message"]["content"]
            # 提取 JSON
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            data = json.loads(content)
            terminate = data.get("should_terminate", False)
            # 缓存结果（先清理过期条目，防止内存无限增长）
            if len(self._termination_cache) > 100:
                cutoff = now - 300  # 5 分钟
                self._termination_cache = {k: v for k, v in self._termination_cache.items() if v[1] > cutoff}
            self._termination_cache[thread_id] = (terminate, now)
            return terminate
        except Exception as e:
            logger.warning(f"终止判断失败: {e}")
            return False  # 出错时不终止

