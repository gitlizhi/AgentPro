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
        # 本地缓存房间成员信息 (可选)
        self._room_members = {}

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
                if user_input and sender not in [self.agent_id, 'super_user']:
                    asyncio.create_task(self._process_message(sender, user_input, image_data, new_thread))
                    return
                if user_input.startswith("/approve"):
                    tool_name = user_input.split()[1] if len(user_input.split()) > 1 else None
                    decision = {"decisions": [{"type": "approve"}]}
                    await self.brain._complete_approval(tool_name, decision)
                    return
                
                elif user_input.startswith("/reject"):
                    tool_name = user_input.split()[1] if len(user_input.split()) > 1 else None
                    decision = {"decisions": [{"type": "reject"}]}
                    await self.brain._complete_approval(tool_name, decision)
                    return
                
                elif user_input.startswith("/edit"):
                    parts = user_input.split(maxsplit=2)
                    tool_name = parts[1] if len(parts) > 1 else None
                    edited_args = json.loads(parts[2]) if len(parts) > 2 else None
                    decision = {
                        "decisions": [
                            {
                                "type": "edit",
                                "edited_action": {
                                    "name": tool_name,
                                    "args": edited_args
                                }
                            }
                        ]
                    }
                    await self.brain._complete_approval(tool_name, decision)
                    return
                # 普通消息：后台处理，不阻塞
                asyncio.create_task(self._process_message(sender, user_input, image_data, new_thread))
                # 可选：立即回复“已收到”
                # await self.comm.send_to_agent(sender, {"text": "✅ 已收到，正在处理..."})
            
            elif msg_type == "group_message":
                room_id = data.get("room_id")
                payload = data.get("payload", {})
                user_input = payload.get("text", "")
                image_data = payload.get("image")
                sender = data.get("from")
                asyncio.create_task(self._process_group_message(room_id, sender, user_input, image_data))
                return
            
            elif msg_type == "invite_to_room":
                room_id = data.get("room_id")
                target_agent = data.get("target_agent")  # 可能是自己
                if target_agent == self.agent_id:
                    # 被邀请，主动加入房间
                    await self.comm.send_to_room(room_id, {"type": "join_room", "room_id": room_id, "agent_id": self.agent_id})
            
            elif msg_type in ("room_members_update", "room_members"):
                room_id = data.get("room_id")
                members = data.get("members", [])
                # 更新本地缓存
                self._room_members[room_id] = members
                logger.info(f"Room '{room_id}' members updated: {members}")
                # 可选：如果 Agent 需要感知成员变化（例如更新上下文），可以在此添加逻辑
            
            elif msg_type == "rooms_list":
                rooms = data.get("rooms", [])
                logger.info(f"Received rooms list: {rooms}")
                # 可以根据需要初始化房间成员缓存
                for rid in rooms:
                    if rid not in self._room_members:
                        self._room_members[rid] = []
                        
            elif msg_type == "register_ack":
                logger.info("Registration acknowledged by hub")
            else:
                logger.warning(f"Unknown message type: {msg_type}")
        except Exception as e:
            logger.error(f"Unhandled exception in _handle_message: {e}", exc_info=True)

    async def _process_message(self, sender, user_input, image_data, new_thread):
        """后台处理普通消息（包括可能触发 HITL 的任务）"""
        private_thread_id = f"private_{self.agent_id}_{sender}"
        try:
            response = await self.brain.process(
                user_id=sender,
                user_input=user_input,
                image_data=image_data,
                new_thread=new_thread,
                thread_id_override=private_thread_id,
                silent=False
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

    async def _process_group_message(self, room_id, sender, user_input, image_data):
        group_thread_id = f"group_{room_id}"
        mention = f"@{self.agent_id}"
        silent = mention not in user_input  # 未被点时静默
        response = await self.brain.process(
            user_id=sender,
            user_input=user_input + '\n' + f'[当前群聊内成员如下：{self._room_members.get(room_id, '无')}]',
            image_data=image_data,
            new_thread=False,
            thread_id_override=group_thread_id,
            silent=silent
        )
        if not silent and response:
            await self.comm.send_to_room(room_id,{"text": response})
    
    async def _periodic_think(self):
        while self._running:
            await asyncio.sleep(random.randint(60, 600))
            await self.brain._think_and_act()

    async def run(self):
        self._running = True
        # self._think_task = asyncio.create_task(self._periodic_think())
        logger.info(f"Agent {self.agent_id} starting...")
        await self.comm.connect()

    async def stop(self):
        self._running = False
        # if self._think_task:
        #     self._think_task.cancel()
        await self.comm.close()
        await self.brain.close()