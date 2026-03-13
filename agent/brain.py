"""
大脑决策层
"""
import os
import sys
import uuid
import json
import random
import copy
from agent.utils import call_zhipu_chat
import dateparser
from datetime import datetime, timezone, timedelta
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain.agents.middleware import SummarizationMiddleware
from agent.model_config import model_config  # 导入配置
from agent.memory import get_memory
from deepagents import create_deep_agent
# from deepagents.backends.filesystem import FilesystemBackend
# from deepagents.backends import LocalShellBackend
from agent.scheduler import get_scheduler
from agent.tasks import send_reminder
from agent.db import get_pool
from agent.intent import IntentType, INTENT_DESCRIPTIONS
from config import config
from langchain.tools import tool
from agent.sandboxed_backend import DockerSandboxBackend


import logging
logging.getLogger('langgraph').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
    

class Brain:
    
    def __init__(
            self,
            comm = None,
            model_config_key: str = "zhipu",
            db_pool=None,
            use_long_term_memory=True,
            agent_id=None
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
        # 检查点
        if db_pool is None:
            from agent.db import get_pool
            db_pool = get_pool()
        self.checkpointer = AsyncPostgresSaver(db_pool)
        
        # 1. 配置后端 (FilesystemBackend 允许技能脚本访问本地文件)
        #    这里需要根据你的项目结构调整根目录
        # root_dir = os.path.expanduser("~")  # 这会得到当前用户的家目录
        root_dir = os.getcwd()
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        self.docker_backend = DockerSandboxBackend(
            image="python:3.12-slim",  # 可自定义镜像
            mem_limit="512m",
            cpu_limit=1.0,
            network_disabled=True,  # 根据需要允许或禁用网络
            desktop_path=config.backend.docker_volumes,      # 如果需要控制电脑桌面文件夹，需要配置
            # env={
            #     "PATH": f"{os.path.dirname(sys.executable)};{os.environ.get('PATH', '')}",
            #     "PYTHONPATH": root_dir,
            #     "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            #     "POSTGRES_URI": os.environ.get("POSTGRES_URI", ""),
            # }
        )
        
        # backend = LocalShellBackend(
        #     root_dir=config.backend.backend_root_dir,
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
       
        # 2. 指定技能目录路径 (相对于 backend 的根目录)
        skills_dir = "/agent/skills/"  # 注意：路径以 "/" 开头，相对于 backend 的 root_dir
        
        self.agent = create_deep_agent(
            model=self.model,
            # tools=self._create_custom_tools(),  # 添加自定义工具
            system_prompt=self._build_system_prompt(),
            # backend=backend,
            backend=self.docker_backend,
            skills=[str(skills_dir)],
            checkpointer=self.checkpointer,
            interrupt_on={
                "delete_file": True,  # Default: approve, edit, reject
                "execute": False,
                "read_file": False,  # No interrupts needed
            },
            middleware=[
                    SummarizationMiddleware(
                    model=self.model,
                    trigger=("tokens", 4000),  # 当历史超过 4000 token 时触发
                    keep=("messages", 20),  # 保留最近 20 条消息，其余用摘要代替
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
        base = (f"你是一个有帮助的AI助手，可以调用工具来完成任务。"
                f"你当前的运行环境是{self.get_platform()}。"
                f"当你不知道该如何处理任务时，可以尝试从skill中加载技能来辅助你完成任务。"
                "在调用工具或者skill之前，请先写下你的思考过程。"
                "如果工具调用出错，请分析错误原因，并尝试其他方法。"
                )
        instructions = """
        注意：你的文件系统环境中，宿主机的桌面目录被挂载在 `/desktop` 下。因此，当用户提到“桌面”上的文件时，你应该使用 `/desktop/文件名` 的路径来读取或写入文件。

        例如：
        - 用户说“读取桌面上的 test.txt”，你应该调用 `read_file` 工具，路径为 `/desktop/test.txt`。
        - 用户说“修改桌面上李白古诗.txt 的内容”，你应该使用 `/desktop/李白古诗.txt`。

        不要使用 Windows 路径（如 C:\\Users...），因为容器内无法识别。
        """
        return base + instructions
    
    async def process(self, user_id: str, user_input: str, image_data: str = None, new_thread: bool = False) -> str:
        self.is_busy = True
        try:
            self.user_id = user_id
            if image_data:
                return await self._handle_image(user_input, image_data)
            else:
                intent_data = await self._classify_intent(user_input)
                return await self._handle_intent(intent_data, user_id, user_input, new_thread)
        finally:
            self.is_busy = False
            self.last_run_time = datetime.now()
    
    async def _detect_reminder_intent(self, user_input: str) -> dict:
        """调用模型判断是否是定时任务，并提取时间和消息"""
        prompt = f"""
        请注意，当前时间为{datetime.now()}
        请分析以下用户输入，用户希望在未来某个时间收到提醒，需要提取提醒的时间和内容，请仔细思考时间。

        请以JSON格式输出，包含两个字段：
        - reminders: 一个数组，每个元素是一个对象，包含 "time" (需要你将自然语言转为代码可解析的时间，格式如：2026-03-09 10:10:19) 和 "message" (提醒内容)。
        - has_other: 布尔值，表示是否包含其他任务。

        如果用户输入没有明确的时间或提醒内容，则不应归类为reminder。

        用户输入："{user_input}"

        只输出JSON，不要任何额外文字。"""
        try:
            content = await call_zhipu_chat(prompt, model=config.model.default_model, temperature=config.model.model_temperature)
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
            print(f"data={data}, type={type(data)}")
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
        """调用大模型进行意图分类，返回包含intent和可能参数的字典"""
        # 构建意图选项字符串
        intent_lines = []
        intent_str = ""
        for intent, desc in INTENT_DESCRIPTIONS.items():
            line = f"- {intent.value}: {desc}"
            intent_str += f'，{intent.value}'
            intent_lines.append(line)
        intent_options = "\n".join(intent_lines)
        
        prompt = f"""
        请注意，当前时间为{datetime.now()},
        请分析以下用户输入，判断其属于哪一种意图。意图选项如下：
        {intent_options}

        请以字符串回复意图，必须是以下选项中的一个{intent_str}。

        用户输入："{user_input}"

        只输出答案，不要任何额外文字。"""
        # print(prompt)
        try:
            response = await call_zhipu_chat(prompt, model=config.model.intent_model, temperature=config.model.model_temperature)
            content = response["choices"][0]["message"]["content"]
            print(f"content={content}")
            return content
        except Exception as e:
            print(f"意图分类失败: {e}")
            return IntentType.CHAT.value
    
    async def _handle_intent(self, intent: str, user_id: str, user_input: str, new_thread) -> str:
        """根据意图分发到对应的处理函数"""
        if intent == IntentType.SET_REMINDER.value:
            reminders = await self._detect_reminder_intent(user_input)
            if reminders:
                print(f'reminders={reminders}')
                return await self._handle_set_reminder(reminders)
            else:
                return "未能理解提醒的时间和内容，请重新描述。"
            
        elif intent == IntentType.COMPLEX_TASKS.value:
            return await self._handle_complex_tasks(user_input, new_thread)
        
        elif intent == IntentType.QUERY_REMINDER.value:
            return await self._handle_query_reminder(user_id)
        
        else:  # chat 或其他
            return await self._handle_chat(user_input, new_thread)

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
                    job_id = f"reminder_{self.user_id}_{int(remind_time.timestamp())}"
                    scheduler.add_job(
                        send_reminder,
                        trigger='date',
                        run_date=remind_time,
                        args=[self.user_id, message],
                        id=job_id,
                        replace_existing=True
                    )
                    # 2. 插入 reminders 表
                    async with pool.connection() as conn:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                "INSERT INTO reminders (user_id, reminder_time, message) VALUES (%s, %s, %s)",
                                (self.user_id, remind_time, message)
                            )
                    responses.append(f"在 {remind_time.strftime('%Y-%m-%d %H:%M:%S')} 提醒你 {message}")
                else:
                    responses.append(f"出错了，无法理解这个时间：{time_str}")
            else:
                responses.append("出错了 提醒信息不完整")
        return "好的，我会" + "，".join(responses)
    
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
    
    async def _handle_complex_tasks(self, user_input: str, new_thread: bool = False):
        """处理复杂推理任务时，默认不需要上下文。"""
        memories = []
        if self.memory:
            memories = self.memory.query_relevant(user_input, self.user_id, n_results=3)
        
        # 2. 构建系统提示（基础提示 + 长期记忆信息）
        base_prompt = self._build_system_prompt()
        if memories:
            memory_text = "\n\n## 关于用户的长期记忆：\n" + "\n".join([
                f"- {m['content']} (来自 {m['metadata'].get('timestamp', '过去')})"
                for m in memories
            ])
            base_prompt += memory_text
        print(f'base_prompt: {base_prompt}', flush=True)
        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_input}
        ]
        
        current_ai_message = ""  # 累积当前 AI 消息的文本
        async for chunk, metadata in self.agent.astream(
                {"messages": messages},
                {"configurable": {"thread_id": uuid.uuid4()}},      # 独立的上下文
                stream_mode="messages",
        ):
            # 处理 AI 消息块（可能是文本片段或工具调用）
            if chunk.type == "AIMessageChunk":
                if chunk.content:
                    # 实时发送每个文本片段给用户（打字机效果）
                    current_ai_message += chunk.content
                if chunk.tool_calls:
                    if current_ai_message:
                        await self.comm.send_to_agent(self.user_id, {"text": current_ai_message})
                        current_ai_message = ""
                    # 发送工具调用信息
                    tool_call_info = f"🔧 调用工具: {chunk.tool_calls}"
                    await self.comm.send_to_agent(self.user_id, {"text": tool_call_info})
                    # 工具调用本身可能不包含文本，但如果有内容也累积
            # 处理工具返回消息块
            elif chunk.type == "ToolMessageChunk":
                if current_ai_message:
                    await self.comm.send_to_agent(self.user_id, {"text": current_ai_message})
                    current_ai_message = ""
                tool_result = f"🛠️ 工具返回: {chunk.content}"
                await self.comm.send_to_agent(self.user_id, {"text": tool_result})
        # 流结束后，current_ai_message 即为完整的 AI 回复（包含思考和最终答案）
        return current_ai_message
    
    async def _handle_chat(self, user_input: str, new_thread: bool = False):
        """聊天"""
        chat_id = f'{self.agent_id}_{self.user_id}'
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
        
        memories = []
        if self.memory:
            memories = self.memory.query_relevant(user_input, self.user_id, n_results=3)
        
        # 2. 构建系统提示（基础提示 + 长期记忆信息）
        base_prompt = self._build_system_prompt()
        if memories:
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
                    if current_ai_message:
                        await self.comm.send_to_agent(self.user_id, {"text": current_ai_message})
                        current_ai_message = ""
                    # 发送工具调用信息
                    tool_call_info = f"🔧 调用工具: {chunk.tool_calls}"
                    await self.comm.send_to_agent(self.user_id, {"text": tool_call_info})
                    # 工具调用本身可能不包含文本，但如果有内容也累积
            # 处理工具返回消息块
            elif chunk.type == "ToolMessageChunk":
                if current_ai_message:
                    await self.comm.send_to_agent(self.user_id, {"text": current_ai_message})
                    current_ai_message = ""
                tool_result = f"🛠️ 工具返回: {chunk.content}"
                await self.comm.send_to_agent(self.user_id, {"text": tool_result})
        # 流结束后，current_ai_message 即为完整的 AI 回复（包含思考和最终答案）
        return current_ai_message
    
    async def _handle_image(self, user_input: str, image_data: str) -> str:
        """处理图片输入，返回视觉模型的结果"""
        # 获取视觉模型（需要在 model_config.py 中预先配置）
        model = model_config.get_model("vision")
        # 构造多模态消息：文本 + 图片
        content = [
            {"type": "text", "text": user_input},
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
        
        prompt = f"""你是一个有内在思考能力的AI。请根据以下背景生成一个简短的闲聊的口语内容，用于主动和用户沟通（不超过50字）。

            背景信息：
            - 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - 思考类型：{thought_type}
            - 用户的长期记忆：
            {memories_text}
            - 最近对话：
            {recent_text}
        
            请生成一个自然、有温度、可能带有好奇心的内心想法。不要以“作为AI”开头，直接说出想法。"""
        try:
            response = await call_zhipu_chat(prompt, model=config.model.default_model, temperature=0.8)      # temperature高一点
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
    
    