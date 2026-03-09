"""
智能体主类
"""
import asyncio
import logging
import random
from agent.brain import Brain
from agent.communication import Communication
from psycopg_pool import ConnectionPool
import config

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
            self,
            agent_id: str,
            db_pool=None,
            model_config_key: str = "zhipu",
    ):
        self.agent_id = agent_id
        self._think_task = None
        
        self.comm = Communication(
            agent_id=agent_id,
            hub_url=f"ws://{config.HUB_HOST}:{config.HUB_PORT}",
            on_message=self._handle_message
        )
        self.brain = Brain(
            comm=self.comm,
            model_config_key=model_config_key,
            db_pool=db_pool,
            agent_id=agent_id
        )
        self._running = False
    
    async def _handle_message(self, data: dict):
        msg_type = data.get("type")
        if msg_type == "message":
            payload = data.get("payload", {})
            user_input = payload.get("text", "")
            new_thread = payload.get("new_thread", False)  # 读取标记
            if user_input:
                logger.info(f"Received message: {user_input}, (new_thread={new_thread})")
                sender = data.get("from")
                response = await self.brain.process(sender, user_input, new_thread=new_thread)
                if sender:
                    await self.comm.send_to_agent(sender, {"text": response})
        elif msg_type == "register_ack":
            logger.info("Registration acknowledged by hub")
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def _periodic_think(self):
        """随机触发一次话题"""
        while self._running:
            await asyncio.sleep(random.randint(60, 600))  # 1分钟到10分钟随机
            # 在 brain 中执行，不干涉主线程
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