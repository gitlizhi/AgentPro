import asyncio
import sys
import json
import re
from contextlib import asynccontextmanager
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel
from datetime import datetime
import psycopg
from config import config  # 导入你的配置

# 获取数据库连接字符串（同步方式）
DB_URI = config.db.postgres_uri

def get_db_connection():
    return psycopg.connect(DB_URI)

# 在 lifespan 中初始化表（同步）
def init_db():
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                thread_id VARCHAR(255) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()   # 同步初始化
    asyncio.create_task(connect_to_hub())
    yield

    
class MessageIn(BaseModel):
    thread_id: str
    role: str   # 'user' or 'assistant'
    content: str

class MessageOut(MessageIn):
    id: int
    created_at: datetime
    
# 全局变量
hub_ws = None
online_agents = set()
frontend_connections = set()
my_agent_id = 'super_user'


def clean_agent_response(text: str) -> str:
    """移除模型内部生成的摘要块"""
    text = re.sub(r'## SESSION INTENT.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'## SUMMARY.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'## ARTIFACTS.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'## NEXT STEPS.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*##\s*$', '', text, flags=re.MULTILINE)
    lines = [line for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def get():
    with open("client.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frontend_connections.add(websocket)
    try:
        # 发送初始在线 Agent 列表
        await websocket.send_json({"type": "agents", "agents": list(online_agents)})
        # 可以请求用户所在的房间列表（Hub 应支持 get_my_rooms）
        if hub_ws:
            await hub_ws.send(json.dumps({"type": "get_my_rooms", "agent_id": my_agent_id}))
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            
            # 私聊发送
            if msg_type == "send":
                target = message.get("to")
                text = message.get("text")
                image = message.get("image")
                new_thread = message.get("new_thread", False)
                if hub_ws and target:
                    payload = {"text": text, "new_thread": new_thread}
                    if image:
                        payload["image"] = image
                    await hub_ws.send(json.dumps({
                        "type": "message",
                        "from": my_agent_id,
                        "to": target,
                        "payload": payload
                    }))
            
            # 审批决策
            elif msg_type == "approval_decision":
                target = message.get("to")
                decision = message.get("decision")
                tool = message.get("tool")
                tool_call_id = message.get("tool_call_id")
                edited_args = message.get("edited_args")
                command = f"/{decision} {tool_call_id}"
                if edited_args:
                    command += f" {json.dumps(edited_args)}"
                await hub_ws.send(json.dumps({
                    "type": "message",
                    "from": my_agent_id,
                    "to": target,
                    "payload": {"text": command}
                }))
            
            # 群组：创建房间
            elif msg_type == "create_room":
                room_id = message.get("room_id")
                if room_id and hub_ws:
                    await hub_ws.send(json.dumps({
                        "type": "create_room",
                        "room_id": room_id,
                        "agent_id": my_agent_id
                    }))
            
            # 群组：加入房间
            elif msg_type == "join_room":
                room_id = message.get("room_id")
                if room_id and hub_ws:
                    await hub_ws.send(json.dumps({
                        "type": "join_room",
                        "room_id": room_id,
                        "agent_id": my_agent_id
                    }))
            
            # 群组：离开房间
            elif msg_type == "leave_room":
                room_id = message.get("room_id")
                if room_id and hub_ws:
                    await hub_ws.send(json.dumps({
                        "type": "leave_room",
                        "room_id": room_id,
                        "agent_id": my_agent_id
                    }))
            
            # 群组：发送群消息
            elif msg_type == "group_message":
                room_id = message.get("room_id")
                text = message.get("text")
                image = message.get("image")
                if room_id and hub_ws:
                    payload = {"text": text}
                    if image:
                        payload["image"] = image
                    await hub_ws.send(json.dumps({
                        "type": "group_message",
                        "room_id": room_id,
                        "from": my_agent_id,
                        "payload": payload
                    }))
            
            # 群组：邀请进群
            elif msg_type == "invite_to_room":
                room_id = message.get("room_id")
                target_agent = message.get("target_agent")
                if room_id and target_agent and hub_ws:
                    await hub_ws.send(json.dumps({
                        "type": "invite_to_room",
                        "room_id": room_id,
                        "target_agent": target_agent,
                        "inviter": my_agent_id
                    }))
            
            # 群组：邀请进群
            elif msg_type == "get_room_members":
                room_id = message.get("room_id")
                if room_id and hub_ws:
                    await hub_ws.send(json.dumps({
                        "type": "get_room_members",
                        "room_id": room_id
                    }))
    
    except WebSocketDisconnect:
        frontend_connections.remove(websocket)


async def connect_to_hub():
    global hub_ws, online_agents
    uri = "ws://localhost:8765"
    while True:
        try:
            async with websockets.connect(uri) as ws:
                hub_ws = ws
                await ws.send(json.dumps({"type": "register", "agent_id": my_agent_id}))
                await ws.send(json.dumps({"type": "get_agents"}))
                
                async for message in ws:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "agents_list":
                        online_agents = set(data.get("agents", []))
                        if my_agent_id in online_agents:
                            online_agents.remove(my_agent_id)
                        for conn in frontend_connections:
                            await conn.send_json({"type": "agents", "agents": list(online_agents)})
                    
                    elif msg_type == "message":
                        sender = data.get("from")
                        payload = data.get("payload", {})
                        text = payload.get("text", "")
                        inner_type = payload.get("type", "")
                        if inner_type == 'approval_request':
                            for conn in frontend_connections:
                                await conn.send_json({
                                    "type": "approval_request",
                                    "from": payload.get("from"),
                                    "tool": payload.get("tool"),
                                    "args": payload.get("args"),
                                    "allowed": payload.get("allowed"),
                                    "tool_call_id": payload.get("tool_call_id"),
                                })
                        else:
                            cleaned = clean_agent_response(text)
                            if not cleaned:
                                continue
                            for conn in frontend_connections:
                                await conn.send_json({
                                    "type": "message",
                                    "from": sender,
                                    "text": cleaned
                                })
                    
                    # 群组相关转发
                    elif msg_type == "room_created":
                        room_id = data.get("room_id")
                        for conn in frontend_connections:
                            await conn.send_json({
                                "type": "room_created",
                                "room_id": room_id
                            })
                    
                    elif msg_type == "room_joined":
                        room_id = data.get("room_id")
                        for conn in frontend_connections:
                            await conn.send_json({
                                "type": "room_joined",
                                "room_id": room_id
                            })
                    
                    elif msg_type == "room_members_update":
                        room_id = data.get("room_id")
                        members = data.get("members", [])
                        for conn in frontend_connections:
                            await conn.send_json({
                                "type": "room_members_update",
                                "room_id": room_id,
                                "members": members
                            })
                    
                    elif msg_type == "room_members":
                        room_id = data.get("room_id")
                        members = data.get("members", [])
                        for conn in frontend_connections:
                            await conn.send_json({
                                "type": "room_members",
                                "room_id": room_id,
                                "members": members
                            })
                    
                    elif msg_type == "rooms_list":
                        rooms = data.get("rooms", [])
                        for conn in frontend_connections:
                            await conn.send_json({
                                "type": "rooms_list",
                                "rooms": rooms
                            })
                    
                    elif msg_type == "group_message":
                        room_id = data.get("room_id")
                        sender = data.get("from")
                        payload = data.get("payload", {})
                        text = payload.get("text", "")
                        cleaned = clean_agent_response(text)
                        if cleaned:
                            for conn in frontend_connections:
                                await conn.send_json({
                                    "type": "group_message",
                                    "room_id": room_id,
                                    "from": sender,
                                    "text": cleaned
                                })
        
        except Exception as e:
            print(f"Hub 连接失败: {e}")
            await asyncio.sleep(5)
    
@app.post("/chat/history")
async def save_message(msg: MessageIn):
    if 'super_user' not in msg.thread_id:
        return {"status": "ignored"}
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_messages (thread_id, role, content) VALUES (%s, %s, %s)",
            (msg.thread_id, msg.role, msg.content)
        )
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/chat/history")
async def get_history(thread_id: str, limit: int = 100):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, role, content, created_at FROM chat_messages WHERE thread_id = %s ORDER BY created_at LIMIT %s",
            (thread_id, limit)
        )
        rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "created_at": row[3].isoformat()
        }
        for row in rows
    ]
@app.get("/chat/threads")
async def get_threads():
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT thread_id FROM chat_messages ORDER BY thread_id")
        rows = cur.fetchall()
    conn.close()
    return {"threads": [row[0] for row in rows]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)