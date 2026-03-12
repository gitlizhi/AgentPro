"""
websocket 服务端，负责转发消息
"""
import asyncio
import json
import logging
import websockets
from config import config
from websockets.legacy.server import WebSocketServerProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Hub:
    def __init__(self):
        self.clients = {}  # agent_id -> websocket
    
    async def register(self, agent_id: str, websocket: WebSocketServerProtocol):
        self.clients[agent_id] = websocket
        logger.info(f"Agent {agent_id} registered. Total clients: {len(self.clients)}")
        # 发送确认
        await websocket.send(json.dumps({"type": "register_ack"}))
    
    async def unregister(self, agent_id: str):
        if agent_id in self.clients:
            del self.clients[agent_id]
            logger.info(f"Agent {agent_id} unregistered")
    
    async def route_message(self, data: dict):
        """根据 data['to'] 转发消息"""
        if not isinstance(data, dict):
            logger.warning(f"Received non-dict message: {data}")
            return
        target = data.get("to")
        logger.info(f"Routing message to {target}: {data.get('payload')}")
        if target == "broadcast":
            # 广播给除发送者外的所有人
            sender = data.get("from")
            for aid, ws in self.clients.items():
                if aid != sender:
                    await ws.send(json.dumps(data, ensure_ascii=False))
        elif target in self.clients:
            await self.clients[target].send(json.dumps(data, ensure_ascii=False))
        else:
            logger.warning(f"Target agent {target} not found")
    
    async def handler(self, websocket: WebSocketServerProtocol):
        """处理每个客户端连接"""
        agent_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "register":
                    agent_id = data["agent_id"]
                    await self.register(agent_id, websocket)
                elif data.get("type") == "message":
                    await self.route_message(data)
                else:
                    logger.warning(f"Unknown message type: {data.get('type')}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if agent_id:
                await self.unregister(agent_id)


async def main():
    hub = Hub()
    async with websockets.serve(hub.handler, config.hub.hub_host, config.hub.hub_port):
        logger.info(f"Hub started on ws://{config.hub.hub_host}:{config.hub.hub_port}")
        await asyncio.Future()  # 运行 forever


if __name__ == "__main__":
    asyncio.run(main())