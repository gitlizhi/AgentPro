"""
大脑决策层
"""
import os
import sys
import uuid
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain.agents.middleware import SummarizationMiddleware

import config
from agent.model_config import model_config  # 导入配置
from agent.memory import get_memory
from agent.skill_loader import SkillRegistry
from deepagents import create_deep_agent
# from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends import LocalShellBackend
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
        
    async def process(self, user_id: str, user_input: str, new_thread: bool = False) -> str:
        self.user_id = user_id
        chat_id = f'{self.agent_id}_{user_id}'
        # 1. 决定使用哪个 thread_id
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
                print("✅ 图执行完毕", flush=True)
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
    
    def _build_system_prompt(self):
        return f"你是一个有帮助的AI助手，可以调用工具来完成任务。\n\n当前用户 ID 是 {self.user_id}。当你要记住信息时，必须使用 --user_id {self.user_id} 参数调用 remember-fact 工具。"