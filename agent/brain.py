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
from agent.utils import call_big_model_chat
import dateparser
from datetime import datetime, timezone, timedelta
from langgraph.types import Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain.agents.middleware import SummarizationMiddleware
from agent.model_config import model_config  # 导入配置
from agent.memory import get_memory
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
    build_proactive_chat_prompt,
    build_termination_judge_prompt,
)
from config import config
from langchain.tools import tool
from langchain_tavily import TavilySearch
from agent.sandboxed_backend import DockerSandboxBackend
from agent.tools import (launch_agent, stop_agent, stop_all_agents_impl)
from agent.reflection import init_chroma, submit_task_for_reflection
from agent.browser_tools import browser, close_browser_session
from agent.task_buffer import TaskBuffer
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
        self.user_id = None
        # 获取模型
        self.model = model_config.get_model(config.model.default_provider)  # model_config 仍需按需
        self.thread_id = None
        
        self.comm = comm
        self.is_busy = False  # 标记是否正在处理用户请求
        self.last_run_time = datetime.now()
        self.recent_active_messages = {}  # AI主动发起的对话记录 格式 {user_id: {"content": str, "timestamp": datetime}}
        
        self.memory = get_memory() if use_long_term_memory else None
        # # 初始化反思模块的向量库
        init_chroma(self.memory.client)
        # 检查点
        if db_pool is None:
            from agent.db import get_pool
            db_pool = get_pool()
        self.checkpointer = AsyncPostgresSaver(db_pool)
        
        # 和其他Agent交互工具
        self.send_to_agent_tool = self._create_send_to_agent_tool()
        # 创建群组相关的工具
        room_tools = self._create_room_tools()
        
        self.agent_msg_cache = {}  # user_id -> 累积的消息文本
        self.agent_msg_timer = {}  # user_id -> asyncio.Task
        
        self._pending_approvals = {}
        # 用于去重
        self.sent_msg_ids_by_thread = {}  # thread_id -> set
        
        self._termination_cache = {}
        # 任务缓冲模块
        self.task_buffer = TaskBuffer()
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
        
        # 2. 指定技能目录路径 (相对于 backend 的根目录)
        skills_dir = "/agent/skills/"  # 注意：路径以 "/" 开头，相对于 backend 的 root_dir
        # 自定义工具
        tools = [self.send_to_agent_tool, TavilySearch(max_results=5), self._create_log_memory(), launch_agent, stop_agent, stop_all_agents_impl, browser] + room_tools
        tools = tools + [list_skills, load_skill, search_skills, skill_stats, upgrade_skill, report_skill_result]
        self.agent = create_deep_agent(
            model=self.model,
            tools=tools,
            system_prompt=self._build_system_prompt() if custom_system_prompt is None else custom_system_prompt,
            # backend=backend,
            backend=self.docker_backend,
            skills=[str(skills_dir)],
            checkpointer=self.checkpointer,
            subagents=[reflection_subagent],  # 在线反思子代理
            interrupt_on={
                "windows_automation": {"allowed_decisions": ["approve", "reject"]},
                "launch_agent": {"allowed_decisions": ["approve", "reject"]},
                # "browser": {"allowed_decisions": ["approve", "reject"]},
            },
            middleware=[
                    SummarizationMiddleware(
                    model=self.model,
                    trigger=("tokens", 20000),  # 当历史超过 20000 token 时触发
                    keep=("messages", 30),  # 保留最近 30 条消息，其余用摘要代替
                ),
            ]
        )
    
    def get_platform(self):
        if os.name == 'nt':
            return "Windows"
        elif os.name == 'posix':
            return "Linux"
        else:
            return "Unknown OS"

    def _build_system_prompt(self):
        return build_brain_system_prompt(self.agent_id)
    
    async def update_memory(self, user_id: str, user_input: str, thread_id: str):
        """静默更新指定线程的记忆"""
        await self.process(user_id, user_input, thread_id_override=thread_id, silent=True)
        
    async def process(self, user_id: str, user_input: str, image_data: str = None, new_thread: bool = False,
                      thread_id_override: str = None, silent: bool = False) -> str:
        self.is_busy = True
        try:
            self.user_id = user_id
            effective_thread_id = thread_id_override if thread_id_override else self.thread_id
            if self.user_id != 'super_user':
                intent_data = IntentType.COMPLEX_TASKS.value
            else:
                intent_data = await self._classify_intent(user_input)
            response = await self._handle_intent(intent_data, user_id, user_input, image_data, new_thread,
                                                 effective_thread_id, silent)
            return response if not silent else ""
        finally:
            self.is_busy = False
            self.last_run_time = datetime.now()
    
    async def process_group_message(self, user_id: str, user_input: str, image_data: str = None, new_thread: bool = False) -> str:
        # 处理群组消息
        self.is_busy = True
        try:
            self.user_id = user_id
            if self.user_id != 'super_user':        # 如果是Agent之间的交互，则跳过意图识别，直接认为是复杂任务
                intent_data = IntentType.COMPLEX_TASKS.value
            else:
                intent_data = await self._classify_intent(user_input)
            # print(f'意图识别为：{intent_data}')
            return await self._handle_intent(intent_data, user_id, user_input, image_data, new_thread)
        finally:
            self.is_busy = False
            self.last_run_time = datetime.now()
    
    def get_thread_id(self, new_thread, chat_id):
        if new_thread:
            # 用户要求新对话：生成新 ID，并更新元数据
            print(f'new_thread: {new_thread}', flush=True)
            self.thread_id = f"{chat_id}_{uuid.uuid4()}"
            self.memory.set_user_metadata(chat_id, "last_thread_id", self.thread_id)
        else:
            # 尝试从长期记忆恢复上次的 thread_id
            if not (self.thread_id and self.thread_id.startswith(f'{chat_id}')):
                last_thread = self.memory.get_user_metadata(f'{chat_id}', "last_thread_id")
                if last_thread:
                    print(f'加载从长期记忆中的last_thread_id')
                    self.thread_id = last_thread
                else:
                    print(f'首次对话，生成新 ID')
                    # 首次对话，生成新 ID
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
            print(f"意图识别失败: {e}")
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
            print(f"意图分类失败: {e}")
            return IntentType.CHAT.value
    
    async def _handle_intent(self, intent: str, user_id: str, user_input: str, image_data: str = None,
                             new_thread: bool = False, thread_id: str = None, silent: bool = False) -> str:
        # 临时覆盖 self.thread_id
        original_thread_id = self.thread_id
        if thread_id:
            self.thread_id = thread_id
        try:
            if intent == IntentType.SET_REMINDER.value:
                reminders = await self._detect_reminder_intent(user_input)
                if reminders:
                    return await self._handle_set_reminder(reminders)
                else:
                    return "未能理解提醒的时间和内容，请重新描述。"
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
                            print(f"数据库连接错误，重试 {attempt + 1}/{max_retries}...")
                            time.sleep(2**attempt)
                else:
                    responses.append(f"出错了，无法理解这个时间：{time_str}")
            else:
                responses.append("出错了 提醒信息不完整")
        return "好的，reminder_bot会" + "，".join(responses)
    
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
                        print(f"已标记 {updated} 条过期提醒", flush=True)
                
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
     
    async def _handle_with_agent(self, user_input: str, image_data: str = None, new_thread: bool = False,
                                    silent: bool = False):
        # 如果是 Agent 之间的对话，执行缓存合并逻辑
        if self.user_id != 'super_user':
            # 取消已有的定时器（新消息到来，重新计时）
            if self.user_id in self.agent_msg_timer:
                self.agent_msg_timer[self.user_id].cancel()
            
            strings = ['检索', '记忆', '关于', '经验', '参考']
            if all(string in user_input for string in strings) and len(user_input) <= 50:     # 废话提取掉
                return
            
            # 累积消息
            if self.user_id not in self.agent_msg_cache:
                self.agent_msg_cache[self.user_id] = user_input
            else:
                self.agent_msg_cache[self.user_id] += "\n" + user_input
            
            # 创建新的延时处理任务
            async def delayed_process():
                await asyncio.sleep(5)
                if hasattr(self, 'agent_msg_cache'):
                    full_input = self.agent_msg_cache.pop(self.user_id, "")
                    if full_input:
                        await self._process_agent_message(full_input, image_data, new_thread)
                    if self.user_id in self.agent_msg_timer:
                        del self.agent_msg_timer[self.user_id]
            
            if user_input:
                self.agent_msg_timer[self.user_id] = asyncio.create_task(delayed_process())
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
        memories = []
        if self.memory and self.user_id == 'super_user':
            memories = self.memory.query_relevant(user_input, self.user_id, n_results=3)
        base_prompt = self._build_system_prompt()
        if memories:
            memory_text = "\n\n## 关于用户的长期记忆：\n" + "\n".join([
                f"- {m['content']} (来自 {m['metadata'].get('timestamp', '过去')})"
                for m in memories
            ])
            base_prompt += memory_text
        if image_data:
            image_desc = await self._handle_image(image_data)
            base_prompt += f"\n\n[图片信息] 对方刚上传了一张图片，内容描述如下：“{image_desc}”"
        
        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # 注意：必须使用同一个 thread_id，不能动态生成
        config = {"configurable": {"thread_id": self.thread_id}}
        if self.thread_id not in self.sent_msg_ids_by_thread:
            self.sent_msg_ids_by_thread[self.thread_id] = await self._load_sent_ids_from_checkpoint(config)
        sent_ids = self.sent_msg_ids_by_thread[self.thread_id]
        
        # HITL 循环：初始输入为 messages
        input_state = {"messages": messages}
        command = None
        
        while True:
            # 如果有恢复命令，则使用 Command 作为输入
            if command:
                input_state = Command(resume=command)
            
            async for event in self.agent.astream(input_state, config, stream_mode="updates"):
                # print(f"DEBUG event: {event}")  # 添加这一行
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
                                msg_id = getattr(msg, 'id', None) or f"hash_{hash(msg.content)}"
                                if msg.type == "ai" and msg.content:
                                    if msg_id not in sent_ids:
                                        await self.comm.send_to_agent(self.user_id, {"text": msg.content})
                                        sent_ids.add(msg_id)
                                if (hasattr(msg, 'tool_calls') and msg.tool_calls and self.user_id == 'super_user' and not silent
                                        and msg_id not in sent_ids):
                                    for tc in msg.tool_calls:
                                        tool_call_info = f"🔧 调用工具: {tc}"
                                        await self.comm.send_to_agent(self.user_id, {"text": tool_call_info})
                                    sent_ids.add(msg_id)
                                elif msg.type == "tool" and self.user_id == 'super_user' and not silent and msg_id not in sent_ids:
                                    tool_result = f"🛠️ 工具返回: {msg.content}"
                                    await self.comm.send_to_agent(self.user_id, {"text": tool_result})
                                    sent_ids.add(msg_id)
                                    # 获取工具名称（可以从 msg.name 或 context 中，这里简单从 sent_ids 推断）
                                    tool_name = getattr(msg, 'name', 'unknown_tool')
                                    self.task_buffer.add_step(
                                        self.thread_id,
                                        f"调用工具： {tool_name}",
                                        msg.content[:500],  # 截断过长结果
                                    )
            
            else:
                # 没有中断，流正常结束
                break
    
    async def _process_agent_message(self, user_input: str, image_data: str = None, new_thread: bool = False):
        chat_id = f'{self.agent_id}_{self.user_id}'
        self.get_thread_id(new_thread, chat_id)
        
        # ===== 新增：记录收到的消息（有助于任务活跃检测） =====
        if hasattr(self, 'task_buffer') and self.thread_id:
            self.task_buffer.add_step(
                self.thread_id,
                f"收到来自 {self.user_id} 的消息",
                user_input[:500]
            )
            
        # ---- 新增终止检查 ----
        if await self._should_terminate_conversation(self.thread_id):
            print(f"对话 {self.thread_id} 被判定为死循环，终止处理。")
            # 可选：发送终止通知给对方
            # await self.comm.send_to_agent(self.user_id, {"text": "我认为我们应该停止交流，我不会再对该话题进行回复"})
            return
        # ----------------------
        
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # 加载已发送消息 ID 集合（可选，用于去重）
        if self.thread_id not in self.sent_msg_ids_by_thread:
            self.sent_msg_ids_by_thread[self.thread_id] = await self._load_sent_ids_from_checkpoint(config)
        sent_ids = self.sent_msg_ids_by_thread[self.thread_id]
        
        # 构建系统提示（仅超级用户才添加记忆）
        base_prompt = self._build_system_prompt()
        if self.memory and self.user_id == 'super_user':
            memories = self.memory.query_relevant(user_input, self.user_id, n_results=3)
            if memories:
                memory_text = "\n\n## 关于用户的长期记忆：\n" + "\n".join(...)
                base_prompt += memory_text

        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_input}
        ]
        
        input_state = {"messages": messages}
        command = None
        
        while True:
            if command:
                input_state = Command(resume=command)
            
            async for event in self.agent.astream(input_state, config, stream_mode="values"):
                if "__interrupt__" in event:
                    interrupts = event["__interrupt__"]
                    decisions = await self._process_interrupts(interrupts)
                    command = decisions if decisions else None
                    break
                else:
                    if "messages" in event:
                        for msg in event["messages"]:
                            # 只发送新消息（AI 内容）
                            if msg.type == "ai" and msg.content:
                                msg_id = getattr(msg, 'id', None) or f"hash_{hash(msg.content)}"
                                if msg_id not in sent_ids:
                                    sent_ids.add(msg_id)
                                    # 发送给目标 Agent（self.user_id 是其他 Agent ID）
                                    await self.comm.send_to_agent(self.user_id, {"text": msg.content})
                                    # ===== 新增：记录本 Agent 发送的消息 =====
                                    self.task_buffer.add_step(
                                        self.thread_id,
                                        f"向 {self.user_id} 发送消息",
                                        msg.content[:500]
                                    )
            else:
                break
                
    async def _handle_chat(self, user_input: str, image_data: str = None, new_thread: bool = False, silent: bool = False):
        """聊天"""
        chat_id = f'{self.agent_id}_{self.user_id}'
        self.get_thread_id(new_thread, chat_id)
        memories = []
        if self.memory and self.user_id == 'super_user':
            memories = self.memory.query_relevant(user_input, self.user_id, n_results=3)
        
        # 2. 构建系统提示（基础提示 + 长期记忆信息）
        base_prompt = self._build_system_prompt()
        if memories and self.user_id == 'super_user':
            memory_text = "\n\n## 关于用户的长期记忆：\n" + "\n".join([
                f"- {m['content']} (来自 {m['metadata'].get('timestamp', '过去')})"
                for m in memories
            ])
            base_prompt += memory_text
            
        # 添加最近主动消息（5分钟内有效）
        recent = self.recent_active_messages.get(self.user_id)
        if recent and (datetime.now() - recent["timestamp"]) < timedelta(minutes=60):
            base_prompt += f"\n\n[主动消息] AI刚才主动对用户说过：“{recent['content']}”"
            # 使用后立即删除，避免每条消息都重复出现（也可保留到过期，根据需要调整）
            del self.recent_active_messages[self.user_id]
        
        if image_data:
            image_desc = await self._handle_image(image_data)
            base_prompt += f"\n\n[图片信息] 用户刚上传了一张图片，内容描述如下：“{image_desc}”"
            
        # print(f'base_prompt: {base_prompt}', flush=True)
        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_input}
        ]
        
        current_ai_message = ""  # 累积当前 AI 消息的文本
        async for chunk, metadata in self.agent.astream(
                {"messages": messages},
                {"configurable": {"thread_id": self.thread_id}},
                stream_mode="messages",
        ):
            # 处理 AI 消息块（可能是文本片段或工具调用）
            if chunk.type == "AIMessageChunk":
                if chunk.content:
                    # 实时发送每个文本片段给用户（打字机效果）
                    current_ai_message += chunk.content
                if chunk.tool_calls:
                    if current_ai_message and not silent:
                        await self.comm.send_to_agent(self.user_id, {"text": current_ai_message})
                        current_ai_message = ""
                    # 发送工具调用信息
                    if self.user_id == 'super_user' and not silent:  # 只有和人类交互才返回工具调用信息
                        tool_call_info = f"🔧 调用工具: {chunk.tool_calls}"
                        # tool_call_info = tool_call_info[:40] + '......(已省略部分消息)'
                        await self.comm.send_to_agent(self.user_id, {"text": tool_call_info})
                    # 工具调用本身可能不包含文本，但如果有内容也累积
            # 处理工具返回消息块
            elif chunk.type == "ToolMessageChunk":
                if current_ai_message and not silent:
                    await self.comm.send_to_agent(self.user_id, {"text": current_ai_message})
                    current_ai_message = ""
                if self.user_id == 'super_user' and not silent:
                    tool_result = f"🛠️ 工具返回: {chunk.content}"
                    await self.comm.send_to_agent(self.user_id, {"text": tool_result})
        # 流结束后，current_ai_message 即为完整的 AI 回复（包含思考和最终答案）
        # if current_ai_message and not silent:
        #     await self.comm.send_to_agent(self.user_id, {"text": current_ai_message})
        return current_ai_message if not silent else ""
    
    async def _handle_image(self, image_data: str) -> str:
        """处理图片输入，返回视觉模型的结果"""
        # 获取视觉模型（需要在 model_config.py 中预先配置）
        model = model_config.get_model("vision")
        # 构造多模态消息：文本 + 图片
        content = [
            {"type": "text", "text": "读取图片的内容，尽可能完整地描述出图片的内容。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
        messages = [{"role": "user", "content": content}]
        try:
            response = await model.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            return f"图片处理失败: {e}"
        
    async def _generate_thought(self, user_id: str = "super_user") -> str:
        """生成一个随机想法，结合记忆和最近对话"""
        thought_types = [
            "基于用户的长期记忆，给用户一个生活建议或提醒，或者找一个有趣的事实，然后以此为主题闲聊。",
            "反思一下今天的对话，有没有什么可以改进的地方。",
            "想一个搞笑的笑话，活跃一下气氛",
            "提出一个哲学问题，和用户一起讨论",
            "找一个近期的网络热点话题，进行讨论",
            "提出关于未来畅想的讨论",
            "回忆过去的事情，童年的趣事",
        ]
        
        thought_type = random.choice(thought_types)
        
        if thought_type in [thought_types[0], thought_types[1]]:
            # 获取长期记忆
            memories_text = "暂无"
            if self.memory:
                facts = self.memory.get_random_facts(user_id, n=3)
                if facts:
                    memories_text = "\n".join([f"- {fact}" for fact in facts])
            
            # 获取最近对话
            recent_msgs = await self._get_recent_messages(user_id, limit=3)
            recent_text = "\n".join([f"- {msg}" for msg in recent_msgs]) if recent_msgs else "暂无"
        else:
            memories_text = recent_text = "暂无"
        
        prompt = build_proactive_chat_prompt(thought_type, memories_text, recent_text)
        try:
            response = await call_big_model_chat(prompt, model=config.model.default_model, temperature=0.8)      # temperature高一点
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # print(f"生成想法失败: {e}")
            return "今天天气不错，想出去走走。"
    
    async def _think_and_act(self):
        """思考并采取行动（如主动发送消息）"""
        if self.is_busy:
            return  # 正在忙，跳过
        if datetime.now() - self.last_run_time < timedelta(minutes=5):  # 刚忙完五分钟内不主动发消息
            return
        self.last_run_time = datetime.now()
        thought = await self._generate_thought()
        # print(thought)
        target_user = "super_user"
        if self.comm:
            await self.send_ai_message(target_user, f"{thought}")

    async def _get_recent_messages(self, user_id: str, limit: int = 5) -> list:
        """获取用户最近对话的最后 limit 条消息内容"""
        # 获取用户的 last_thread_id
        thread_id = self.memory.get_user_metadata(f"{self.agent_id}_{user_id}", "last_thread_id")
        if not thread_id:
            return []
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = await self.checkpointer.aget_tuple(config)
            if not state:
                return []
            # 从 checkpoint 中提取消息
            # state 是 CheckpointTuple 对象，其 checkpoint 字段包含 channel_values
            if hasattr(state, 'checkpoint') and state.checkpoint:
                channel_values = state.checkpoint.get('channel_values', {})
                messages = channel_values.get('messages', [])
            else:
                messages = []
            
            recent = []
            for msg in reversed(messages):
                if len(recent) >= limit:
                    break
                # msg 可能是 BaseMessage 对象，有 type 和 content 属性
                if hasattr(msg, 'type') and hasattr(msg, 'content') and msg.type in ["human", "ai"]:
                    recent.insert(0, msg.content)
            return recent
        except Exception as e:
            print(f"获取最近消息失败: {e}")
            return []
    
    async def send_ai_message(self, user_id: str, content: str):
        """主动发送消息并记录到内存"""
        from datetime import datetime
        self.recent_active_messages[user_id] = {
            "content": content,
            "timestamp": datetime.now()
        }
        # 发送消息
        await self.comm.send_to_agent(user_id, {"text": content})
        print(f"[主动消息已记录] {content}")
    
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
                await self.comm.send_to_agent(self.user_id, msg)
                # 等待用户决策
                decision = await self._wait_for_user_decision(tool_call_id)
                decisions.append(decision)
        return decisions
    
    async def _wait_for_user_decision(self, tool_call_id: str):
        future = asyncio.get_event_loop().create_future()
        self._pending_approvals[tool_call_id] = future  # 用唯一 ID 存储
        try:
            decision = await asyncio.wait_for(future, timeout=60)
            return decision
        except asyncio.TimeoutError:
            return {"type": "reject"}
    
    async def _complete_approval(self, tool_call_id: str, decision):
        if tool_call_id in self._pending_approvals:
            self._pending_approvals[tool_call_id].set_result(decision)
            del self._pending_approvals[tool_call_id]
        else:
            print(f"[WARN] No pending approval for {tool_call_id}")
    
    def _create_send_to_agent_tool(self):
        @tool
        async def send_to_agent(target_agent_id: str, message: str) -> str:
            """
            向指定的 Agent 发送消息，不等待对方回复。
            :param target_agent_id: 目标 Agent 的 ID，例如 'debater' 或 'researcher'。
            :param message: 要发送的消息内容，例如 '请反驳我的观点'。
            :return: 发送成功的标志
            示例: send_to_agent(target_agent_id='debater', message='现在请开始反驳。')
            """
            if message and '[停止交流]' in message:
                return f'已和 {target_agent_id} 停止交流'
            await self.comm.send_to_agent(target_agent_id, {"text": message})
            return f'消息已经发送给了 Agent : {target_agent_id}，请等待对方回复。'
        return send_to_agent
    
    async def close(self):
        for task in self.agent_msg_timer.values():
            task.cancel()
        for thread_id, task in list(self.task_buffer.buffers.items()):
            if task.get("status") == "in_progress":
                idle = time.time() - task.get("last_active_time", 0)
                if idle > 3600:  # 1小时无活动
                    self.task_buffer.finish_task(thread_id, "timeout", user_feedback="任务因长时间无活动而终止")
        await close_browser_session()
    
    def _create_room_tools(self):
        # @tool
        # async def create_room(room_id: str) -> str:
        #     """创建一个新群组。"""
        #     await self.comm.send({"type": "create_room", "room_id": room_id, "agent_id": self.agent_id})
        #     return f"已创建群组 {room_id}"
        
        @tool
        async def join_room(room_id: str) -> str:
            """加入一个已有群组。"""
            await self.comm.send({"type": "join_room", "room_id": room_id, "agent_id": self.agent_id})
            return f"已请求加入群组 {room_id}"
        
        @tool
        async def leave_room(room_id: str) -> str:
            """离开群组。"""
            await self.comm.send({"type": "leave_room", "room_id": room_id, "agent_id": self.agent_id})
            return f"已离开群组 {room_id}"
        
        @tool
        async def send_group_message(room_id: str, message: str) -> str:
            """向群组发送消息。"""
            await self.comm.send({
                "type": "group_message",
                "room_id": room_id,
                "from": self.agent_id,
                "payload": {"text": message}
            })
            return f"消息已发送到群组 {room_id}"
        
        # return [create_room, join_room, leave_room, send_group_message]
        return [join_room, leave_room, send_group_message]
    
    def _create_log_memory(self):
        @tool
        async def log_memory(description: str, result: str, task_complete: bool = False, config: RunnableConfig = None) -> str:
            """
            记录当前任务的执行步骤或最终总结。
            当 task_complete=True 时，将整个任务提交给反思模块进行离线分析。

            :param description: 步骤描述或任务总结
            :param result: 执行结果
            :param task_complete: 是否完成任务
            :return: 提示信息
            """
            # 从 config 中提取 thread_id
            thread_id = config.get("configurable", {}).get("thread_id") if config else None
            if not thread_id:
                return "错误：无法获取当前对话 ID。"

            self.task_buffer.add_step(thread_id, description, result)
            
            if task_complete:
                final_result = "success" if "成功" in result else "failure"
                task_data = self.task_buffer.finish_task(thread_id, final_result)
                if task_data:
                    # 启动后台任务（不等待）
                    asyncio.create_task(self._process_reflection(task_data))
                return f"步骤已记录，任务结束，将进行经验反思。"
            else:
                return "步骤已记录。"
        
        return log_memory
    
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
            print(f"加载已发送ID失败: {e}")
        return set()
    
    async def _get_conversation_history_for_termination(self, thread_id: str, max_messages: int = 20):
        """
        从 checkpoint 中提取对话历史，返回适合终止判断的格式：
        [{"speaker": "AgentA", "message": "..."}, ...]
        规则：将 AI 消息视为当前 Agent（self.agent_id），将 Human 消息视为对方 Agent。
        """
        configs = {"configurable": {"thread_id": thread_id}}
        state = await self.agent.aget_state(configs)
        if not state or not state.values:
            return []
        
        messages = state.values.get("messages", [])
        if not messages:
            return []
        
        # 从 thread_id 中解析对方 Agent ID（格式：f"{self.agent_id}_{other_agent_id}_{uuid}"）
        parts = thread_id.split("_")
        if len(parts) >= 2 and parts[0] == self.agent_id:
            other_agent_id = parts[1]
        else:
            other_agent_id = "unknown"
        
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
            # 缓存结果
            self._termination_cache[thread_id] = (terminate, time.time())
            return terminate
        except Exception as e:
            print(f"终止判断失败: {e}")
            return False  # 出错时不终止

