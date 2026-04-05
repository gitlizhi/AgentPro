import asyncio
import json
import re
from contextlib import asynccontextmanager
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# 全局变量
hub_ws = None
online_agents = set()
frontend_connections = set()
my_agent_id = 'super_user'

def clean_agent_response(text: str) -> str:
    """移除模型内部生成的摘要块"""
    # 移除各种摘要块
    text = re.sub(r'## SESSION INTENT.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'## SUMMARY.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'## ARTIFACTS.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'## NEXT STEPS.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
    # 移除可能残留的单独 ## 行
    text = re.sub(r'^\s*##\s*$', '', text, flags=re.MULTILINE)
    # 清理多余空行
    lines = [line for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(connect_to_hub())
    yield

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
        await websocket.send_json({"type": "agents", "agents": list(online_agents)})
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "send":
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
                    if data.get("type") == "agents_list":
                        online_agents = set(data.get("agents", []))
                        if my_agent_id in online_agents:
                            online_agents.remove(my_agent_id)
                        for conn in frontend_connections:
                            await conn.send_json({"type": "agents", "agents": list(online_agents)})
                    elif data.get("type") == "message":
                        sender = data.get("from")
                        payload = data.get("payload", {})
                        text = payload.get("text", "")
                        cleaned = clean_agent_response(text)
                        if not cleaned:
                            continue
                        for conn in frontend_connections:
                            await conn.send_json({
                                "type": "message",
                                "from": sender,
                                "text": cleaned
                            })
        except Exception as e:
            print(f"Hub 连接失败: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)