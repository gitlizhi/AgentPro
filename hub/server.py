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
        self.observers = set()
    
    async def register(self, agent_id: str, websocket: WebSocketServerProtocol):
        self.clients[agent_id] = websocket
        logger.info(f"Agent {agent_id} registered. Total clients: {len(self.clients)}")
        if agent_id in ["super_user", "reminder_bot"]:  # 需要接收列表更新的客户端
            self.observers.add(websocket)
        # 发送确认
        await websocket.send(json.dumps({"type": "register_ack"}))
        await self.broadcast_agents()

    async def unregister(self, agent_id: str):
        if agent_id in self.clients:
            ws = self.clients.pop(agent_id, None)
            if ws:
                self.observers.discard(ws)
            logger.info(f"Agent {agent_id} unregistered")
        await self.broadcast_agents()
    
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
            try:
                await self.clients[target].send(json.dumps(data, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Failed to send to {target}: {e}")
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
                elif data.get("type") == "get_agents":
                    # 返回当前所有在线 agent_id 列表
                    agents = list(self.clients.keys())
                    await websocket.send(json.dumps({"type": "agents_list", "agents": agents}))
                else:
                    logger.warning(f"Unknown message type: {data.get('type')}")
        except websockets.exceptions.ConnectionClosed:
            print('=============== Connection closed ===============')
        finally:
            if agent_id:
                await self.unregister(agent_id)
    
    async def broadcast_agents(self):
        agents = list(self.clients.keys())
        message = json.dumps({"type": "agents_list", "agents": agents})
        for ws in list(self.observers):
            try:
                await ws.send(message)
            except:
                self.observers.discard(ws)
            
async def main():
    hub = Hub()
    async with websockets.serve(hub.handler, config.hub.hub_host, config.hub.hub_port):
        logger.info(f"Hub started on ws://{config.hub.hub_host}:{config.hub.hub_port}")
        await asyncio.Future()  # 运行 forever


if __name__ == "__main__":
    asyncio.run(main())