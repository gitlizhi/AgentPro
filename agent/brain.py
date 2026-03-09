"""
大脑决策层
"""
import os
import sys
import uuid
import json
from agent.utils import call_zhipu_chat
import dateparser
from datetime import datetime, timezone
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain.agents.middleware import SummarizationMiddleware
from agent.model_config import model_config  # 导入配置
from agent.memory import get_memory
from agent.skill_loader import SkillRegistry
from deepagents import create_deep_agent
# from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends import LocalShellBackend
from agent.scheduler import get_scheduler
from agent.tasks import send_reminder
from agent.db import get_pool
from agent.intent import IntentType, INTENT_DESCRIPTIONS


import logging
# logging.getLogger('psycopg.pool').setLevel(logging.DEBUG)
logging.getLogger('langgraph').setLevel(logging.DEBUG)

skills_dir = os.path.join(os.path.dirname(__file__), "skills")
skill_registry = SkillRegistry(skills_dir)

class Brain:
    
    def __init__(
            self,
            model_config_key: str = "zhipu",
            db_pool=None,
            use_long_term_memory=True,
            agent_id=None
    ):
        self.agent_id = agent_id
        self.user_id = None
        # 获取模型
        self.model = model_config.get_model(model_config_key)
        self.thread_id = None
        
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
        
        backend = LocalShellBackend(
            root_dir=root_dir,
            virtual_mode=True,
            timeout=30,
            max_output_bytes=10000,
            env={
                "PATH": f"{os.path.dirname(sys.executable)};{os.environ.get('PATH', '')}",
                "PYTHONPATH": root_dir,
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
                "POSTGRES_URI": os.environ.get("POSTGRES_URI", ""),  # 关键添加
            }
        )

        # 2. 指定技能目录路径 (相对于 backend 的根目录)
        skills_dir = "/agent/skills/"  # 注意：路径以 "/" 开头，相对于 backend 的 root_dir
        
        # 基础系统提示
        base_system_prompt = (
            "你是一个有帮助的AI助手，可以调用工具来完成任务。"
            # "如果需要特定领域的详细指导，请先调用 load_skill 工具加载对应技能。"
        )
        
        self.agent = create_deep_agent(
            model=self.model,
            # tools=self.concrete_tools,
            system_prompt=base_system_prompt,
            backend=backend,
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
        return (f"你是一个有帮助的AI助手，可以调用工具来完成任务。"
                f"你当前的运行环境是{self.get_platform()}。"
                f"当你不知道该如何处理任务时，可以尝试从skill中加载技能来辅助你完成任务。"
                )
    
    async def process(self, user_id: str, user_input: str, new_thread: bool = False) -> str:
        self.user_id = user_id
        intent_data = await self._classify_intent(user_input)
        return await self._handle_intent(intent_data, user_id, user_input, new_thread)
    
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
            content = await call_zhipu_chat(prompt, model="glm-4.7", temperature=0.0)
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
        print(prompt)
        try:
            response = await call_zhipu_chat(prompt, model="glm-4-flash", temperature=0.0)
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
    
    async def _handle_chat(self, user_input: str, new_thread: bool = False):
        # 1. 决定使用哪个 thread_id
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
        
        # 2. 构建系统提示（基础提示 + 记忆信息）
        base_prompt = self._build_system_prompt()  # 你已有的方法
        if memories:
            memory_text = "\n\n## 你可能需要知道的过往信息：\n" + "\n".join([
                f"- {m['content']} (来自 {m['metadata'].get('timestamp', '过去')})"
                for m in memories
            ])
            base_prompt += memory_text
        print(f'base_prompt: {base_prompt}', flush=True)
        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_input}
        ]
        
        final_answer = ""
        async for event in self.agent.astream(
                {"messages": messages},
                {"configurable": {"thread_id": self.thread_id}},
                stream_mode="values",  # 或省略，默认是 "values"
        ):
            if "__end__" in event:  # 某些流模式下会有结束标记
                print(" 图执行完毕", flush=True)
            if "messages" in event:
                for msg in event["messages"]:
                    if msg.type == "ai":
                        if msg.content:
                            # print(f"🤖 AI: {msg.content}", flush=True)
                            final_answer = msg.content  # 注意：每次是整个消息，不是增量
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"🔧 工具调用: {msg.tool_calls}", flush=True)
                    elif msg.type == "tool":
                        print(f"🛠️ 工具返回: {msg.content}", flush=True)
        return final_answer
