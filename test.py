import asyncio
import json
import websockets
import argparse
from datetime import datetime

async def receive_messages(ws):
    """持续接收并打印来自 Hub 的所有消息"""
    try:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            msg_from = data.get('from', 'unknown')
            msg_type = data.get('type', 'unknown')
            if msg_type == 'message':
                payload = data.get('payload', {})
                text = payload.get('text', '')
                print(f"\n📨 来自 {msg_from}: {text}")
            else:
                # 其他类型消息（如注册确认）
                print(f"\n[系统] {data}")
            # 重新显示输入提示，方便用户继续输入
            print("你: ", end='', flush=True)
    except websockets.exceptions.ConnectionClosed:
        print("连接已关闭")

async def send_messages(ws, local_agent_id, target_agent_id):
    """处理用户输入并发送消息"""
    while True:
        user_input = await asyncio.to_thread(input, "你: ")
        if user_input.lower() in ("exit", "quit"):
            break

        if user_input.startswith("/new"):
            text = user_input[4:].strip()
            await ws.send(json.dumps({
                "type": "message",
                "from": local_agent_id,
                "to": target_agent_id,
                "payload": {"text": text, "new_thread": True}
            }))
        else:
            await ws.send(json.dumps({
                "type": "message",
                "from": local_agent_id,
                "to": target_agent_id,
                "payload": {"text": user_input}
            }))
        print(f"[SENT] {user_input} at {datetime.now()}")
    # 退出时关闭连接
    await ws.close()

async def chat(target_agent_id: str):
    uri = "ws://localhost:8765"
    local_agent_id = "super_user"
    async with websockets.connect(uri) as ws:
        # 注册客户端
        await ws.send(json.dumps({"type": "register", "agent_id": local_agent_id}))
        resp = await ws.recv()
        print("✅ 已连接到 Hub，注册成功:", resp)

        print(f"🎯 开始与智能体 {target_agent_id} 对话，输入 'exit' 或 'quit' 退出。")
        # 同时运行接收和发送任务
        receiver = asyncio.create_task(receive_messages(ws))
        sender = asyncio.create_task(send_messages(ws, local_agent_id, target_agent_id))
        # 等待其中一个完成（通常是发送任务因用户退出而结束）
        done, pending = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
        # 取消另一个任务
        for task in pending:
            task.cancel()
        # 等待取消完成
        await asyncio.gather(*pending, return_exceptions=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="与智能体进行持续对话的测试客户端")
    parser.add_argument("--agent", required=True, help="目标智能体的ID（例如 agent_xxx）")
    args = parser.parse_args()
    asyncio.run(chat(args.agent))