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

from agent.conversation_tracker import ConversationTracker
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
    create_load_user_profile_tool,
    create_list_online_agents_tool,
    create_send_to_agent_tool,
    create_room_tools,
    create_log_memory_tool,
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
from langchain_core.runnables import RunnableConfig
import chromadb

import logging
logging.getLogger('langgraph').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

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
        
        # 智能体对话追踪器（轮次计数 + 逐级降级 + 硬上限）
        self.conversation_tracker = ConversationTracker()
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
            }
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
        tools = [self.send_to_agent_tool, create_list_online_agents_tool(self), TavilySearch(max_results=5), create_log_memory_tool(self), create_load_user_profile_tool(self), launch_agent, stop_agent, stop_all_agents_impl, browser] + room_tools + COMPUTER_TOOLS
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
    
    def get_platform(self):
        if os.name == 'nt':
            return "Windows"
        elif os.name == 'posix':
            return "Linux"
        else:
            return "Unknown OS"

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
        else:
            status = "在线" if (self.online_agents and self.user_id in self.online_agents) else "未知"
            parts.append(
                f"[当前对话] 你正在和智能体 {self.user_id} 对话。对方当前状态：{status}。"
            )
            warning = self.conversation_tracker.get_warning(self.agent_id, self.user_id)
            if warning:
                parts.append(warning)

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

    async def _prepare_agent_config(self) -> tuple:
        """创建 config、修复 checkpoint、加载 sent_ids。返回 (config, sent_ids)。"""
        config = {"configurable": {"thread_id": self.thread_id}}
        await self._sanitize_checkpoint(config)
        if self.thread_id not in self.sent_msg_ids_by_thread:
            self.sent_msg_ids_by_thread[self.thread_id] = await self._load_sent_ids_from_checkpoint(config)
        return config, self.sent_msg_ids_by_thread[self.thread_id]

    async def _safe_send(self, text: str, **extra) -> bool:
        """安全发送消息给当前用户。WebSocket/网络异常仅记日志，不向上传播。"""
        payload = {"text": text, **extra}
        try:
            await self.comm.send_to_agent(self.user_id, payload)
            return True
        except Exception as e:
            logger.warning(f"发送消息失败 (user={self.user_id}): {e}")
            return False

    async def _send_ai_message(self, msg, sent_ids: set) -> bool:
        """去重并发送 AI 消息到当前用户。返回 True 表示实际发送了。"""
        msg_id = getattr(msg, 'id', None) or f"hash_{hash(msg.content)}"
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
                      group_context: dict = None) -> str:
        async with self._process_lock:
            self.is_busy = True
            self._current_task = asyncio.current_task()
            await self._notify_status("busy")
            self.group_context = group_context  # 群聊上下文，注入系统提示词
            user_input = self._inject_time_context(user_input)
            try:
                self.user_id = user_id
                effective_thread_id = thread_id_override if thread_id_override is not None else self.thread_id
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
     
    async def _astream_with_timeout(self, input_state, config, stream_mode="updates",
                                     per_event_timeout: float = 300):
        """对 agent.astream 的每次 __anext__ 调用增加超时保护，避免流永久挂起。
        使用 asyncio.wait 代替 wait_for，避免 Python 3.12 将内部 CancelledError 掩码为 TimeoutError。"""
        agen = self.agent.astream(input_state, config, stream_mode=stream_mode)
        deadline = asyncio.get_event_loop().time() + per_event_timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"astream ({stream_mode}) timed out after {per_event_timeout}s")

            next_task = asyncio.ensure_future(agen.__anext__())
            try:
                done, _ = await asyncio.wait([next_task], timeout=min(remaining, 120))
            except asyncio.CancelledError:
                next_task.cancel()
                raise

            if not done:
                next_task.cancel()
                raise asyncio.TimeoutError(f"astream ({stream_mode}) timed out waiting for next event")

            exc = next_task.exception()
            if exc is not None:
                if isinstance(exc, StopAsyncIteration):
                    break
                raise exc
            yield next_task.result()

    async def _handle_with_agent(self, user_input: str, image_data: str = None, new_thread: bool = False,
                                    silent: bool = False):
        # ---- P0: 用户消息自动解除该智能体的所有对话限制 ----
        if self.user_id == 'super_user':
            self.conversation_tracker.reset_all_for(self.agent_id)

        # ---- P0: 检查该对话对是否已被硬截断 ----
        if self.user_id != 'super_user' and self.conversation_tracker.is_capped(self.agent_id, self.user_id):
            logger.info(f"对话 {self.agent_id}<->{self.user_id} 已截断，丢弃收到的消息")
            return

        # 如果是 Agent 之间的对话，委托给 MessageBuffer 延迟合并
        if self.user_id != 'super_user':
            async def on_process(full_text: str, saved_ctx: dict):
                if saved_ctx:
                    self.group_context = saved_ctx
                await self._process_agent_message(full_text, image_data, new_thread)

            self.msg_buffer.enqueue(self.user_id, user_input, self.group_context, on_process)
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

        # Build system-level context as dedicated SystemMessages (not user message prefixes).
        # Inspired by Claude Code: system instructions stay in system role,
        # user messages contain only what the user actually said.
        system_contexts = await self._build_system_contexts(image_data)

        # 程序化注入相关技能经验：作为系统消息而非用户消息后缀
        if self.user_id == 'super_user':
            skill_lessons = await self._get_relevant_skill_lessons(original_user_input)
            if skill_lessons:
                system_contexts.append(skill_lessons)

        messages = []
        for ctx in system_contexts:
            if ctx:
                messages.append({"role": "system", "content": ctx})
        messages.append({"role": "user", "content": user_input})

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
                                    msg_id = getattr(msg, 'id', None) or f"hash_{hash(msg.content)}"
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls and self.user_id == 'super_user' and not silent:
                                        for tc in msg.tool_calls:
                                            tc_id = tc.get('id', '') or f"tc_{hash(str(tc))}"
                                            if tc_id not in sent_ids:
                                                sent_ids.add(tc_id)
                                                await self._safe_send(
                                                    f"🔧 调用工具: {tc['name']}",
                                                    type="tool_call_start",
                                                    tool_name=tc.get("name"),
                                                    tool_args=tc.get("args", {}),
                                                )
                                    elif msg.type == "tool" and self.user_id == 'super_user' and not silent and msg_id not in sent_ids:
                                        sent_ids.add(msg_id)
                                        tool_name = getattr(msg, 'name', 'unknown_tool')
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
        except asyncio.CancelledError:
            raise
        except (OSError, ConnectionError) as e:
            logger.error(f"Agent stream network error in _handle_with_agent: {e}")
            if self.user_id == 'super_user' and not silent:
                await self._safe_send(f"网络连接失败：{e}。请检查代理设置或网络连接后重试。")
        except Exception as e:
            logger.error(f"Agent stream error in _handle_with_agent: {type(e).__name__}: {e}", exc_info=True)
            if self.user_id == 'super_user' and not silent:
                await self._safe_send(f"处理请求时遇到错误 ({type(e).__name__})，请重试。")
    
    async def _process_agent_message(self, user_input: str, image_data: str = None, new_thread: bool = False):
        # ---- 使用 tracker 管理的 thread_id，每次新会话自动隔离 checkpoint ----
        self.thread_id = self.conversation_tracker.get_or_create_thread_id(self.agent_id, self.user_id)
        
        # ===== 新增：记录收到的消息（有助于任务活跃检测） =====
        if hasattr(self, 'task_buffer') and self.thread_id:
            self.task_buffer.add_step(
                self.thread_id,
                f"收到来自 {self.user_id} 的消息",
                user_input[:500]
            )
            
        # ---- P0: 检查对话是否已被硬截断 ----
        if self.conversation_tracker.is_capped(self.agent_id, self.user_id):
            logger.info(f"对话 {self.agent_id}<->{self.user_id} 已截断，跳过消息处理")
            return

        # ---- P0: 对话上限检查（轮次硬截断 + 逐级警告） ----
        if self.conversation_tracker.is_capped(self.agent_id, self.user_id):
            logger.info(f"对话 {self.thread_id} 已达轮次上限，终止处理")
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
            decision = await asyncio.wait_for(future, timeout=60)
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
        # 取消并等待 Brain 自身的后台任务
        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        for thread_id, task in list(self.task_buffer.buffers.items()):
            if task.get("status") == "in_progress":
                idle = time.time() - task.get("last_active_time", 0)
                if idle > 3600:  # 1小时无活动
                    self.task_buffer.finish_task(thread_id, "timeout", user_feedback="任务因长时间无活动而终止")
        await close_browser_session()
            
    async def _process_reflection(self, task_data):
        """异步处理反思（避免在工具内部阻塞）"""
        await asyncio.to_thread(submit_task_for_reflection, task_data)
        
    async def _load_sent_ids_from_checkpoint(self, config):
        """从 checkpoint 加载已发送的消息 ID 集合"""
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
                        ids.add(hash(msg.content))
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

