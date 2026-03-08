"""
大脑决策层
"""
import os
import traceback
from typing import List, Any, Dict, Optional, Callable, Awaitable
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import config
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain.agents.middleware import SummarizationMiddleware, AgentMiddleware
from agent.model_config import model_config  # 导入配置
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from agent.skill_loader import SkillRegistry

import logging
# logging.getLogger('psycopg.pool').setLevel(logging.DEBUG)
logging.getLogger('langgraph').setLevel(logging.DEBUG)

skills_dir = os.path.join(os.path.dirname(__file__), "skills")
skill_registry = SkillRegistry(skills_dir)

@tool
async def load_skill(skill_name: str) -> str:
    """加载指定技能的完整内容到上下文。当你需要处理特定类型的请求时，调用此工具获取详细指导。

    Args:
        skill_name: 技能名称，例如 "read_file", "write_file"
    """
    skill = skill_registry.get_skill(skill_name)
    if not skill:
        available = ", ".join(skill_registry.skills.keys())
        return f"技能 '{skill_name}' 不存在。可用技能：{available}"
    return skill.get_instructions()  # 返回技能的完整内容（如 SKILL.md）
class SkillMetadataMiddleware(AgentMiddleware):
    """中间件：将技能列表注入系统提示"""

    def __init__(self, skills_prompt: str):
        super().__init__()  # 调用父类初始化（如果有必要）
        self.skills_prompt = skills_prompt

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """同步包装：在系统提示后追加技能列表"""
        skills_addendum = (
            f"\n\n## 可用技能\n\n{self.skills_prompt}\n\n"
            "要使用某个技能，请先调用 `load_skill` 工具加载其完整说明。"
        )
        # 假设 request.system_message 是 SystemMessage 对象
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """异步包装（ainvoke 会使用这个）"""
        skills_addendum = (
            f"\n\n## 可用技能\n\n{self.skills_prompt}\n\n"
            "要使用某个技能，请先调用 `load_skill` 工具加载其完整说明。"
        )
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return await handler(modified_request)

class SkillMetadataMiddleware1(AgentMiddleware):
    """中间件：在每次模型调用前，将技能列表注入系统提示"""

    def __init__(self, ):
        # 从注册中心生成技能描述列表
        skills_list = [
            f"- **{name}**: {skill.metadata.description}"
            for name, skill in skill_registry.skills.items()
        ]
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """同步包装（如果 agent 是同步调用）"""
        # 构建技能附录
        skills_addendum = (
            f"\n\n## 可用技能\n\n{self.skills_prompt}\n\n"
            "要使用某个技能，请先调用 `load_skill` 工具加载其完整说明。"
            "加载后，你可以根据说明调用对应的具体工具（如 read_file）来执行任务。"
        )

        # 追加到系统消息
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)

        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)

    # 如果需要异步调用，还需要实现 awrap_model_call
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """异步包装（ainvoke 会使用这个）"""
        skills_addendum = (
            f"\n\n## 可用技能\n\n{self.skills_prompt}\n\n"
            "要使用某个技能，请先调用 `load_skill` 工具加载其完整说明。"
            "加载后，你可以根据说明调用对应的具体工具（如 read_file）来执行任务。"
        )
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return await handler(modified_request)

class Brain:
    
    def __init__(
            self,
            model_config_key: str = "zhipu",
            thread_id: Optional[str] = None,
            db_pool=None,
            skill_registry: SkillRegistry = None,  # 必须传入已加载的注册中心
    ):
        print(f"Brain received skill_registry id: {id(skill_registry)}")  # 添加调试
        if skill_registry is None:
            raise ValueError("skill_registry must be provided")
        self.skill_registry = skill_registry
        
        # 获取模型
        self.model = model_config.get_model(model_config_key)
        self.thread_id = thread_id or "default_thread"
        
        # 创建 load_skill 工具
        self.load_skill_tool = self._create_load_skill_tool()
        
        # 创建具体工具（从已加载的技能中生成）
        self.concrete_tools = self._create_concrete_tools()
        
        # 所有工具 = load_skill + 具体工具
        all_tools = [self.load_skill_tool] + self.concrete_tools
        print("Available tools:", [tool.name for tool in all_tools])
        # 检查点
        if db_pool is None:
            from agent.db import get_pool
            db_pool = get_pool()
        self.checkpointer = AsyncPostgresSaver(db_pool)
        
        # 生成技能列表提示（用于中间件）
        skills_prompt = self._build_system_prompt()
        
        # 创建中间件实例
        skill_middleware = SkillMetadataMiddleware(skills_prompt)
        
        # 基础系统提示
        base_system_prompt = (
            "你是一个有帮助的AI助手，可以调用工具来完成任务。"
            "如果需要特定领域的详细指导，请先调用 load_skill 工具加载对应技能。"
        )
        
        # 创建 agent
        self.agent = create_agent(
            model=self.model,
            tools=all_tools,
            system_prompt=base_system_prompt,
            checkpointer=self.checkpointer,
            middleware=[skill_middleware],  # 只用一个中间件
        )
    
        # SummarizationMiddleware(
        #     model=self.model,
        #     trigger=("tokens", 4000),
        #     keep=("messages", 20),
        # ),
    def _create_load_skill_tool(self):
        @tool
        async def load_skill(skill_name: str) -> str:
            """加载指定技能的完整内容。"""
            skill = self.skill_registry.get_skill(skill_name)
            if not skill:
                available = ", ".join(self.skill_registry.skills.keys())
                return f"技能 '{skill_name}' 不存在。可用技能：{available}"
            return skill.get_instructions()
        
        return load_skill
        
    def _create_concrete_tools(self):
        tools = []
        for skill in self.skill_registry.list_skills():
            skill_tools = skill.get_tools()
            print(f"Skill '{skill.metadata.name}' returned {len(skill_tools)} tools")
            tools.extend(skill_tools)
        return tools
	
    def _build_system_prompt(self):
        skills_info = "\n".join([
            f"- {s.metadata.name}: {s.metadata.description}"
            for s in self.skill_registry.skills.values()
        ])
        return f"""
        你是一个有帮助的AI助手，必须通过调用工具来完成任务。你有以下技能可用：
        {skills_info}
    
        要使用某个技能，**必须首先调用对应的 load_xxx 工具**（例如 load_read_file）来激活它。激活后，该技能的具体工具（如 read_file）就会变得可用，然后你再调用具体工具执行任务。
    
        请始终使用工具调用，不要生成代码示例或伪代码。如果你不确定如何操作，请调用 load_xxx 工具并遵循其返回的指示。
        """
            
    async def process(self, user_input: str) -> str:
        try:
            messages = [HumanMessage(content=user_input)]
            result = await self.agent.ainvoke(
                {"messages": messages},
                {"configurable": {"thread_id": self.thread_id}}
            )
            output_messages = result.get("messages", [])
            if output_messages and isinstance(output_messages[-1], AIMessage):
                reply = output_messages[-1].content
                return reply
            return "代理执行完成，但没有返回可读的消息。"
        except Exception as e:
            traceback.print_exc()
            return f"处理时发生错误: {str(e)}"