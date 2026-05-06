"""
智能体主类
"""
import re
import json
import asyncio
import logging
import random
from agent.brain import Brain
from agent.communication import Communication
from psycopg_pool import ConnectionPool
from config import config
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, agent_id: str, db_pool=None, custom_system_prompt: str = None):
        self.agent_id = agent_id
        self._think_task = None
        self.comm = Communication(
            agent_id=agent_id,
            hub_url=f"ws://{config.hub.hub_host}:{config.hub.hub_port}",
            on_message=self._handle_message
        )
        self.brain = Brain(
            comm=self.comm,
            db_pool=db_pool,
            agent_id=agent_id,
            custom_system_prompt=custom_system_prompt  # 传递
        )
        self._running = False

    async def _handle_message(self, data: dict):
        try:
            if not isinstance(data, dict):
                logger.warning(f"Received non-dict message, ignoring: {data}")
                return
            msg_type = data.get("type")
            if msg_type == "message":
                payload = data.get("payload", {})
                user_input = payload.get("text", "")
                new_thread = payload.get("new_thread", False)
                image_data = payload.get("image")
                sender = data.get("from")
                # 如果是来自其他 Agent 的消息（非自身），也作为新消息处理
                if user_input and sender != self.agent_id:
                    asyncio.create_task(self._process_message(sender, user_input, image_data, new_thread))
                    return
                # 审批命令必须立即处理，不能放到后台任务
                if user_input.startswith("/approve"):
                    tool_name = user_input.split()[1] if len(user_input.split()) > 1 else None
                    await self.brain._complete_approval(tool_name, {"type": "approve"})
                    return
                if user_input.startswith("/approve"):
                    parts = user_input.split()
                    tool_name = parts[1] if len(parts) > 1 else None
                    if tool_name:
                        await self.brain._complete_approval(tool_name, {"type": "approve"})
                    else:
                        await self.comm.send_to_agent(sender, {"text": "错误：未指定工具名称"})
                    return
                
                elif user_input.startswith("/reject"):
                    tool_name = user_input.split()[1] if len(user_input.split()) > 1 else None
                    await self.brain._complete_approval(tool_name, {"type": "reject"})
                    return
                elif user_input.startswith("/edit"):
                    parts = user_input.split(maxsplit=2)
                    tool_name = parts[1] if len(parts) > 1 else None
                    edited_args = json.loads(parts[2]) if len(parts) > 2 else None
                    if edited_args:
                        decision = {
                            "type": "edit",
                            "edited_action": {
                                "name": tool_name,
                                "args": edited_args
                            }
                        }
                        await self.brain._complete_approval(tool_name, decision)
                    else:
                        await self.comm.send_to_agent(sender, {"text": "编辑命令格式错误"})
                    return
    
                # 普通消息：后台处理，不阻塞
                asyncio.create_task(self._process_message(sender, user_input, image_data, new_thread))
                # 可选：立即回复“已收到”
                # await self.comm.send_to_agent(sender, {"text": "✅ 已收到，正在处理..."})
    
            elif msg_type == "register_ack":
                logger.info("Registration acknowledged by hub")
            else:
                logger.warning(f"Unknown message type: {msg_type}")
        except Exception as e:
            logger.error(f"Unhandled exception in _handle_message: {e}", exc_info=True)

    async def _process_message(self, sender, user_input, image_data, new_thread):
        """后台处理普通消息（包括可能触发 HITL 的任务）"""
        try:
            response = await self.brain.process(
                user_id=sender,
                user_input=user_input,
                image_data=image_data,
                new_thread=new_thread
            )
            if response:
                response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL).strip()
            if not response and sender != 'super_user':
                return
            reply_payload = {"text": response if response else ''}
            await self.comm.send_to_agent(sender, reply_payload)
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self.comm.send_to_agent(sender, {"text": f"处理出错: {e}"})

    async def _periodic_think(self):
        while self._running:
            await asyncio.sleep(random.randint(60, 600))
            await self.brain._think_and_act()

    async def run(self):
        self._running = True
        self._think_task = asyncio.create_task(self._periodic_think())
        logger.info(f"Agent {self.agent_id} starting...")
        await self.comm.connect()

    async def stop(self):
        self._running = False
        if self._think_task:
            self._think_task.cancel()
        await self.comm.close()
        await self.brain.close()