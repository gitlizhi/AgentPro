"""
websocket 服务端，负责转发消息
"""
import asyncio
import json
import logging
import websockets
from config import config
import asyncpg
from websockets.legacy.server import WebSocketServerProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Hub:
    def __init__(self):
        self.clients = {}  # agent_id -> websocket
        self.observers = set()
        self.rooms = {}  # 内存缓存，加速消息转发
    
    async def register(self, agent_id: str, websocket: WebSocketServerProtocol):
        self.clients[agent_id] = websocket
        logger.info(f"Agent {agent_id} registered. Total clients: {len(self.clients)}")
        if agent_id in ["super_user", "reminder_bot"]:  # 需要接收列表更新的客户端
            self.observers.add(websocket)
        # 发送确认
        await websocket.send(json.dumps({"type": "register_ack"}))
        await self.broadcast_agents()
        # 广播上线事件给所有已连接的智能体
        await self._broadcast_to_all({"type": "agent_online", "agent_id": agent_id}, exclude=agent_id)

    async def unregister(self, agent_id: str):
        if agent_id in self.clients:
            ws = self.clients.pop(agent_id, None)
            if ws:
                self.observers.discard(ws)
            logger.info(f"Agent {agent_id} unregistered")
        await self.broadcast_agents()
        # 广播离线事件给所有剩余的智能体
        await self._broadcast_to_all({"type": "agent_offline", "agent_id": agent_id})

    async def _broadcast_to_all(self, message: dict, exclude: str = None):
        """向所有已连接客户端广播消息，可排除指定 agent_id"""
        data = json.dumps(message, ensure_ascii=False)
        for aid, ws in list(self.clients.items()):
            if aid != exclude:
                try:
                    await ws.send(data)
                except Exception:
                    pass
    
    async def route_message(self, data: dict):
        """根据 data['to'] 转发消息"""
        if not isinstance(data, dict):
            logger.warning(f"Received non-dict message: {data}")
            return
        target = data.get("to")
        sender = data.get("from", "unknown")
        logger.info(f"Routing message from {sender} to {target}: {data.get('payload')}")
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
            await self.clients[sender].send(json.dumps({"error": f"Target agent {target} not found"}, ensure_ascii=False))
    
    async def handler(self, websocket: WebSocketServerProtocol):
        """处理每个客户端连接"""
        agent_id = None
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message}")
                    continue
                if not isinstance(data, dict):
                    logger.warning(f"Received non-dict message: {data}")
                    continue
                if data.get("type") == "register":
                    agent_id = data["agent_id"]
                    await self.register(agent_id, websocket)
                elif data.get("type") == "message":
                    await self.route_message(data)
                elif data.get("type") == "get_agents":
                    # 返回当前所有在线 agent_id 列表
                    agents = list(self.clients.keys())
                    await websocket.send(json.dumps({"type": "agents_list", "agents": agents}))
                elif data.get("type") == "create_room":
                    room_id = data.get("room_id")
                    creator = data.get("agent_id")
                    try:
                        async with self.db_pool.acquire() as conn:
                            await self._db_execute("INSERT INTO rooms (room_id) VALUES ($1) ON CONFLICT DO NOTHING", room_id)
                            await self._db_execute("INSERT INTO room_members (room_id, agent_id) VALUES ($1, $2)", room_id,
                                               creator)
                    except Exception as e:
                        # 向创建者返回失败
                        await websocket.send(json.dumps({"type": "room_created", "error": str(e)}))
                        return
                    if room_id not in self.rooms:
                        self.rooms[room_id] = set()
                    self.rooms[room_id].add(creator)
                    # 向创建者返回成功
                    await websocket.send(json.dumps({"type": "room_created", "room_id": room_id}))
                    # 可选：广播成员更新
                    await self.broadcast_room_members(room_id)
                
                elif data.get("type") == "join_room":
                    room_id = data.get("room_id")
                    agent_id = data.get("agent_id")
                    async with self.db_pool.acquire() as conn:
                        exists = await conn.fetchval("SELECT 1 FROM rooms WHERE room_id = $1", room_id)
                        if not exists:
                            await websocket.send(json.dumps({"type": "error", "msg": "Room not found"}))
                            return
                        await self._db_execute(
                            "INSERT INTO room_members (room_id, agent_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                            room_id, agent_id)
                    if room_id not in self.rooms:
                        self.rooms[room_id] = set()
                    self.rooms[room_id].add(agent_id)
                    # 广播成员列表更新
                    members = list(self.rooms[room_id])
                    for member in self.rooms[room_id]:
                        if member not in self.clients:
                            continue
                        await self.clients[member].send(json.dumps({
                            "type": "room_members_update",
                            "room_id": room_id,
                            "members": members
                        }))
                
                elif data.get("type") == "leave_room":
                    room_id = data.get("room_id")
                    agent_id = data.get("agent_id")
                    async with self.db_pool.acquire() as conn:
                        await self._db_execute("DELETE FROM room_members WHERE room_id = $1 AND agent_id = $2", room_id,
                                           agent_id)
                        count = await self._db_fetchval("SELECT COUNT(*) FROM room_members WHERE room_id = $1", room_id)
                        if count == 0:
                            await self._db_execute("DELETE FROM rooms WHERE room_id = $1", room_id)
                    if room_id in self.rooms:
                        self.rooms[room_id].discard(agent_id)
                        if not self.rooms[room_id]:
                            del self.rooms[room_id]
                        else:
                            # 广播成员列表更新
                            members = list(self.rooms[room_id])
                            for member in self.rooms[room_id]:
                                if member not in self.clients:
                                    continue
                                await self.clients[member].send(json.dumps({
                                    "type": "room_members_update",
                                    "room_id": room_id,
                                    "members": members
                                }))
                
                elif data.get("type") == "group_message":
                    room_id = data.get("room_id")
                    payload = data.get("payload")
                    sender = data.get("from")
                    logger.info(f"Routing group_message from {sender} to room_id {room_id}: {data.get('payload')}")
                    if room_id in self.rooms and sender in self.rooms[room_id]:
                        for member in self.rooms[room_id]:
                            if member != sender:
                                if member not in self.clients:
                                    continue
                                await self.clients[member].send(json.dumps({
                                    "type": "group_message",
                                    "room_id": room_id,
                                    "from": sender,
                                    "payload": payload
                                }))
                
                elif data.get("type") == "get_my_rooms":
                    agent_id = data.get("agent_id")
                    if not agent_id:
                        await websocket.send(json.dumps({"type": "error", "msg": "Missing agent_id"}))
                        return
                    async with self.db_pool.acquire() as conn:
                        rows = await conn.fetch("SELECT DISTINCT room_id FROM room_members WHERE agent_id = $1",
                                                agent_id)
                        room_ids = [row['room_id'] for row in rows]
                    await websocket.send(json.dumps({"type": "rooms_list", "rooms": room_ids}))
                
                elif data.get("type") == "invite_to_room":
                    room_id = data.get("room_id")
                    inviter = data.get("inviter")  # 实际上可以是 super_user
                    target_agent = data.get("target_agent")
                    if not room_id or not target_agent:
                        await websocket.send(json.dumps({"type": "error", "msg": "Missing room_id or target_agent"}))
                        return
                    # 检查房间是否存在
                    if room_id not in self.rooms:
                        await websocket.send(json.dumps({"type": "error", "msg": "Room not found"}))
                        return
                    # 将 target_agent 加入房间（如果尚未在）
                    async with self.db_pool.acquire() as conn:
                        await self._db_execute(
                            "INSERT INTO room_members (room_id, agent_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                            room_id, target_agent)
                    self.rooms[room_id].add(target_agent)
                    # 广播成员更新
                    await self.broadcast_room_members(room_id)
                    # 可选：向被邀请的 Agent 发送系统消息（通知对方已加入群聊），可通过直接发送私聊消息
                    if target_agent in self.clients:
                        await self.clients[target_agent].send(json.dumps({
                            "type": "message",
                            "from": inviter,
                            "payload": {"text": f"你已被邀请加入群聊 {room_id}"}
                        }))
                
                elif data.get("type") == "get_room_members":
                    room_id = data.get("room_id")
                    if not room_id:
                        await websocket.send(json.dumps({"type": "error", "msg": "Missing room_id"}))
                        return
                    if room_id not in self.rooms:
                        await websocket.send(json.dumps({"type": "error", "msg": "Room not found"}))
                        return
                    members = list(self.rooms[room_id])
                    await websocket.send(json.dumps({"type": "room_members", "room_id": room_id, "members": members}))
                    
                else:
                    logger.warning(f"Unknown message type: {data.get('type')}")
        except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as e:
            # 连接关闭是正常情况，静默处理或仅记录 info 级别日志
            logger.info(
                f"WebSocket connection closed for agent {agent_id if agent_id else 'unknown'}: {type(e).__name__}: {e}")
        except Exception as e:
            # 其他意外错误才记录堆栈
            logger.error(f"Unexpected error in handler for agent {agent_id}: {e}", exc_info=True)
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
    
    async def init_db(self):
        self.db_pool = await asyncpg.create_pool(
            config.db.postgres_uri,
            min_size=2,                           # 保持少量热连接，避免频繁建连
            max_size=10,                          # 根据并发量调整
            max_queries=50000,                    # 每个连接执行 5w 次查询后自动关闭重建
            max_inactive_connection_lifetime=300, # 空闲超过 5 分钟自动从池中移除
            command_timeout=30,                   # 命令超时 30 秒
            server_settings={
                'tcp_keepalives_idle': '60',      # 60 秒空闲后发送 keepalive 探测
                'tcp_keepalives_interval': '10',  # 探测间隔 10 秒
                'tcp_keepalives_count': '6',      # 连续 6 次失败后断开连接
            }
        )
        async with self.db_pool.acquire() as conn:
            await self._db_execute("""
                CREATE TABLE IF NOT EXISTS rooms (
                    room_id VARCHAR(255) PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db_execute("""
                CREATE TABLE IF NOT EXISTS room_members (
                    room_id VARCHAR(255) REFERENCES rooms(room_id) ON DELETE CASCADE,
                    agent_id VARCHAR(255),
                    PRIMARY KEY (room_id, agent_id)
                )
            """)
            # 加载现有群组到内存
            rows = await conn.fetch("SELECT room_id, agent_id FROM room_members")
            for row in rows:
                room_id = row['room_id']
                agent_id = row['agent_id']
                if room_id not in self.rooms:
                    self.rooms[room_id] = set()
                self.rooms[room_id].add(agent_id)
    
    async def broadcast_room_members(self, room_id: str):
        if room_id not in self.rooms:
            return
        members = list(self.rooms[room_id])
        message = json.dumps({"type": "room_members_update", "room_id": room_id, "members": members})
        for member in members:
            if member in self.clients:
                await self.clients[member].send(message)
    
    async def _db_execute(self, query, *args, retries=2):
        """带重试的数据库执行，捕获 ConnectionDoesNotExistError"""
        for attempt in range(retries):
            try:
                async with self.db_pool.acquire() as conn:
                    return await conn.execute(query, *args)
            except asyncpg.exceptions.ConnectionDoesNotExistError as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"Database connection lost, retrying... (attempt {attempt + 1}/{retries})")
                await asyncio.sleep(0.5)
    
    async def _db_fetch(self, query, *args, retries=2):
        """带重试的 fetch"""
        for attempt in range(retries):
            try:
                async with self.db_pool.acquire() as conn:
                    return await conn.fetch(query, *args)
            except asyncpg.exceptions.ConnectionDoesNotExistError as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(0.5)
    
    async def _db_fetchval(self, query, *args, retries=2):
        """带重试的 fetchval"""
        for attempt in range(retries):
            try:
                async with self.db_pool.acquire() as conn:
                    return await conn.fetchval(query, *args)
            except asyncpg.exceptions.ConnectionDoesNotExistError as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(0.5)
    
    
async def main():
    hub = Hub()
    await hub.init_db()  # 创建连接池并加载现有群组到内存
    async with websockets.serve(hub.handler, config.hub.hub_host, config.hub.hub_port, max_size=20 * 1024 * 1024):
        logger.info(f"Hub started on ws://{config.hub.hub_host}:{config.hub.hub_port}")
        await asyncio.Future()  # 运行 forever


if __name__ == "__main__":
    asyncio.run(main())