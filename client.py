import asyncio
import base64
import hashlib
import os
import sys
import json
import re
import signal
import subprocess
from contextlib import asynccontextmanager
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from datetime import datetime, timezone
import psycopg
from config import config  # 导入你的配置
from agent.prompts import build_launch_agent_prompt

# 获取数据库连接字符串（同步方式）
DB_URI = config.db.postgres_uri

def get_db_connection():
    return psycopg.connect(DB_URI)

def fmt_dt(dt):
    """将 naive datetime 标记为 UTC 并返回带时区的 ISO 字符串"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

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
                image TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_threads (
                id SERIAL PRIMARY KEY,
                thread_id VARCHAR(255) UNIQUE NOT NULL,
                agent_id VARCHAR(255) NOT NULL,
                title VARCHAR(500) DEFAULT 'New Chat',
                user_id VARCHAR(255) DEFAULT 'super_user',
                is_archived BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_threads_agent
            ON conversation_threads(agent_id, user_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_threads_updated
            ON conversation_threads(updated_at DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created
            ON chat_messages(thread_id, created_at DESC)
        """)
        # Migration: add image column if not exists
        cur.execute("""
            ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS image TEXT
        """)
    conn.commit()
    conn.close()
    migrate_legacy_threads()

def migrate_legacy_threads():
    """将旧格式 thread_id（如 private_agent_super_user）迁移到 conversation_threads 表"""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT thread_id FROM chat_messages
            WHERE thread_id LIKE 'private\\_%\\_super\\_user'
            AND thread_id NOT LIKE 'private\\_%\\_super\\_user\\_%'
        """)
        old_threads = [row[0] for row in cur.fetchall()]
        for thread_id in old_threads:
            parts = thread_id.split('_')
            # 格式: private_{agent_id}_super_user
            if len(parts) >= 3 and parts[0] == 'private' and parts[-1] == 'user' and parts[-2] == 'super':
                agent_id = '_'.join(parts[1:-2])  # 处理 agent_id 中可能包含下划线
                cur.execute(
                    "SELECT 1 FROM conversation_threads WHERE thread_id = %s",
                    (thread_id,)
                )
                if not cur.fetchone():
                    cur.execute("""
                        INSERT INTO conversation_threads (thread_id, agent_id, title, created_at, updated_at)
                        SELECT %s, %s, %s,
                            (SELECT MIN(created_at) FROM chat_messages WHERE thread_id = %s),
                            (SELECT MAX(created_at) FROM chat_messages WHERE thread_id = %s)
                    """, (thread_id, agent_id, f'Chat with {agent_id}', thread_id, thread_id))
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
    image: str | None = None

class MessageOut(MessageIn):
    id: int
    created_at: datetime

class ConversationCreate(BaseModel):
    agent_id: str
    thread_id: str

class ConversationUpdate(BaseModel):
    title: str | None = None
    is_archived: bool | None = None

class LaunchAgentRequest(BaseModel):
    agent_id: str
    expertise: str  # 专长描述
    
# 全局变量
hub_ws = None
online_agents = set()
frontend_connections = set()
my_agent_id = 'super_user'
_launched_agents: dict[str, subprocess.Popen] = {}  # agent_id -> Popen


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

# 挂载 screenshots 目录，让前端能通过 /screenshots/filename.png 访问截图
import pathlib
_screenshots_dir = pathlib.Path(__file__).parent / "screenshots"
_screenshots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=str(_screenshots_dir)), name="screenshots")

# 挂载 chat_images 目录，存储用户上传的聊天图片
_chat_images_dir = pathlib.Path(__file__).parent / "chat_images"
_chat_images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/chat_images", StaticFiles(directory=str(_chat_images_dir)), name="chat_images")


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
                thread_id = message.get("thread_id")
                if hub_ws and target:
                    payload = {"text": text, "new_thread": new_thread, "thread_id": thread_id}
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

            # 停止任务
            elif msg_type == "stop_task":
                agent_id = message.get("agent_id")
                if agent_id and hub_ws:
                    await hub_ws.send(json.dumps({
                        "type": "stop_task",
                        "to": agent_id,
                        "from": my_agent_id
                    }))
    
    except WebSocketDisconnect:
        frontend_connections.remove(websocket)


async def connect_to_hub():
    global hub_ws, online_agents
    uri = "ws://localhost:8765"
    while True:
        try:
            async with websockets.connect(uri, max_size=20 * 1024 * 1024) as ws:
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

                    elif msg_type == "agent_online":
                        agent = data.get("agent_id")
                        if agent and agent != my_agent_id:
                            for conn in frontend_connections:
                                await conn.send_json({"type": "agent_online", "agent_id": agent})

                    elif msg_type == "agent_offline":
                        agent = data.get("agent_id")
                        if agent:
                            for conn in frontend_connections:
                                await conn.send_json({"type": "agent_offline", "agent_id": agent})

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
                            if not cleaned and not payload.get("image"):
                                continue
                            msg = {
                                "type": "message",
                                "from": sender,
                                "text": cleaned
                            }
                            if payload.get("image"):
                                msg["image"] = payload["image"]
                            for conn in frontend_connections:
                                await conn.send_json(msg)
                    
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
                        if cleaned or payload.get("image"):
                            msg = {
                                "type": "group_message",
                                "room_id": room_id,
                                "from": sender,
                                "text": cleaned
                            }
                            if payload.get("image"):
                                msg["image"] = payload["image"]
                            for conn in frontend_connections:
                                await conn.send_json(msg)
        
                    elif msg_type == "agent_status":
                        for conn in frontend_connections:
                            await conn.send_json({
                                "type": "agent_status",
                                "agent_id": data.get("agent_id"),
                                "status": data.get("status")
                            })

        except Exception as e:
            print(f"Hub 连接失败: {e}")
            await asyncio.sleep(5)

@app.post("/chat/upload-image")
async def upload_image(data: dict):
    """接收 base64 图片，保存到 chat_images/ 目录，返回访问路径"""
    import base64 as b64
    import time as _time
    image_b64 = data.get("image", "")
    if not image_b64:
        return JSONResponse({"error": "缺少 image 字段"}, status_code=400)

    # 解析 data URL: data:image/png;base64,xxxxx
    header = "base64,"
    idx = image_b64.find(header)
    if idx != -1:
        mime_part = image_b64[:idx]
        ext = "png"
        for mime_ext in [("jpeg", "jpg"), ("jpg", "jpg"), ("png", "png"), ("gif", "gif"), ("webp", "webp")]:
            if mime_ext[0] in mime_part:
                ext = mime_ext[1]
                break
        raw = image_b64[idx + len(header):]
    else:
        ext = "png"
        raw = image_b64

    try:
        img_bytes = b64.b64decode(raw)
    except Exception:
        return JSONResponse({"error": "base64 解码失败"}, status_code=400)

    if len(img_bytes) > 10 * 1024 * 1024:
        return JSONResponse({"error": "图片过大，最大 10MB"}, status_code=400)

    # 用内容哈希命名以防重复
    file_hash = hashlib.sha256(img_bytes).hexdigest()[:16]
    filename = f"{file_hash}.{ext}"
    filepath = _chat_images_dir / filename
    if not filepath.exists():
        filepath.write_bytes(img_bytes)

    return {"url": f"/chat_images/{filename}"}


@app.post("/chat/history")
async def save_message(msg: MessageIn):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_messages (thread_id, role, content, image) VALUES (%s, %s, %s, %s)",
            (msg.thread_id, msg.role, msg.content, msg.image)
        )
        cur.execute(
            "UPDATE conversation_threads SET updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
            (msg.thread_id,)
        )
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/chat/history")
async def get_history(thread_id: str, limit: int = 100):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, role, content, created_at, image FROM chat_messages WHERE thread_id = %s ORDER BY created_at LIMIT %s",
            (thread_id, limit)
        )
        rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "created_at": fmt_dt(row[3]),
            "image": row[4]
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

# ── 对话管理 API ──

@app.get("/chat/conversations")
async def list_conversations(agent_id: str = None):
    conn = get_db_connection()
    with conn.cursor() as cur:
        if agent_id:
            cur.execute(
                """SELECT thread_id, agent_id, title, is_archived, created_at, updated_at
                   FROM conversation_threads
                   WHERE agent_id = %s AND user_id = 'super_user' AND is_archived = FALSE
                   ORDER BY updated_at DESC""",
                (agent_id,)
            )
        else:
            cur.execute(
                """SELECT thread_id, agent_id, title, is_archived, created_at, updated_at
                   FROM conversation_threads
                   WHERE user_id = 'super_user' AND is_archived = FALSE
                   ORDER BY updated_at DESC"""
            )
        rows = cur.fetchall()
    conn.close()
    return {
        "conversations": [
            {
                "thread_id": r[0], "agent_id": r[1], "title": r[2],
                "is_archived": r[3], "created_at": fmt_dt(r[4]), "updated_at": fmt_dt(r[5])
            }
            for r in rows
        ]
    }

@app.post("/chat/conversations")
async def create_conversation(data: ConversationCreate):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO conversation_threads (thread_id, agent_id, title, user_id)
               VALUES (%s, %s, 'New Chat', 'super_user')
               ON CONFLICT (thread_id) DO NOTHING""",
            (data.thread_id, data.agent_id)
        )
    conn.commit()
    conn.close()
    return {"status": "ok", "thread_id": data.thread_id}

@app.patch("/chat/conversations/{thread_id}")
async def update_conversation(thread_id: str, data: ConversationUpdate):
    conn = get_db_connection()
    with conn.cursor() as cur:
        updates = []
        params = []
        if data.title is not None:
            updates.append("title = %s")
            params.append(data.title[:500])
        if data.is_archived is not None:
            updates.append("is_archived = %s")
            params.append(data.is_archived)
        if updates:
            params.append(thread_id)
            cur.execute(
                f"UPDATE conversation_threads SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                params
            )
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.delete("/chat/conversations/{thread_id}")
async def delete_conversation(thread_id: str, permanent: bool = False):
    conn = get_db_connection()
    with conn.cursor() as cur:
        if permanent:
            cur.execute("DELETE FROM chat_messages WHERE thread_id = %s", (thread_id,))
            cur.execute("DELETE FROM conversation_threads WHERE thread_id = %s", (thread_id,))
        else:
            cur.execute(
                "UPDATE conversation_threads SET is_archived = TRUE, updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                (thread_id,)
            )
    conn.commit()
    conn.close()
    return {"status": "ok"}

# ── 智能体管理 API ──

@app.post("/agents/launch")
async def launch_agent_endpoint(data: LaunchAgentRequest):
    """直接启动一个新的智能体实例"""
    cmd = [
        sys.executable, "main.py",
        "--agent-id", data.agent_id,
        "--system-prompt", build_launch_agent_prompt(data.expertise)
    ]
    try:
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            process = subprocess.Popen(cmd, startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            process = subprocess.Popen(cmd)
        _launched_agents[data.agent_id] = process
        return {"status": "ok", "agent_id": data.agent_id, "pid": process.pid}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _find_agent_pid(agent_id: str) -> int | None:
    """通过命令行参数查找智能体进程 PID，用于终止 LLM 工具启动的智能体"""
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 f"Get-CimInstance Win32_Process -Filter \"commandline like '%--agent-id {agent_id}%' and name = 'python.exe'\" | Select-Object -ExpandProperty ProcessId"],
                capture_output=True, text=True, timeout=10
            )
            pids = result.stdout.strip().split()
            if pids:
                return int(pids[0])
        else:
            result = subprocess.run(
                ['pgrep', '-f', f'main.py --agent-id {agent_id}'],
                capture_output=True, text=True, timeout=5
            )
            pids = result.stdout.strip().split()
            if pids:
                return int(pids[0])
    except Exception:
        pass
    return None

def _kill_process(pid: int) -> bool:
    """终止指定 PID 的进程树"""
    try:
        if sys.platform == 'win32':
            result = subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], capture_output=True, text=True)
            return result.returncode == 0
        else:
            os.kill(pid, signal.SIGTERM)
            return True
    except Exception:
        return False

@app.post("/agents/stop")
async def stop_agent_endpoint(data: LaunchAgentRequest):
    """终止指定智能体进程（支持 API 启动和 LLM 工具启动两种来源）"""
    agent_id = data.agent_id

    # 路径 1：从 API 启动的记录中查找
    process = _launched_agents.get(agent_id)
    if process is not None:
        # 清理已退出的僵尸进程
        if process.poll() is not None:
            del _launched_agents[agent_id]
        else:
            try:
                _kill_process(process.pid)
                del _launched_agents[agent_id]
                return {"status": "ok", "message": f"已终止智能体 '{agent_id}'"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

    # 路径 2：扫描系统进程（LLM 工具启动的智能体）
    pid = _find_agent_pid(agent_id)
    if pid is not None:
        if _kill_process(pid):
            return {"status": "ok", "message": f"已终止智能体 '{agent_id}' (PID: {pid})"}
        else:
            return {"status": "error", "message": f"无法终止进程 {pid}"}

    return {"status": "error", "message": f"未找到智能体 '{agent_id}' 的运行进程"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)