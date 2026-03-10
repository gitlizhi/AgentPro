import asyncio
import json
import websockets
import argparse
import base64
import os
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
        if user_input.startswith("/broadcast "):
            user_input = user_input[11:].strip()  # 去掉 "/broadcast " 前缀
            target_agent_id = "broadcast"
        elif user_input.startswith("/target "):
            # 切换当前目标智能体
            new_target = user_input[8:].strip()
            if new_target:
                target_agent_id = new_target
                print(f"当前目标已切换为: {target_agent_id}")
            else:
                print("用法: /target <agent_id>")
        if user_input.startswith("/new"):
            text = user_input[4:].strip()
            await ws.send(json.dumps({
                "type": "message",
                "from": local_agent_id,
                "to": target_agent_id,
                "payload": {"text": text, "new_thread": True}
            }))
        elif user_input.startswith("/img "):
            parts = user_input.split(maxsplit=2)
            if len(parts) >= 2:
                image_path = parts[1]
                text = parts[2] if len(parts) > 2 else ""
                if os.path.exists(image_path):
                    image_b64 = encode_image_to_base64(image_path)
                    await ws.send(json.dumps({
                        "type": "message",
                        "from": local_agent_id,
                        "to": target_agent_id,
                        "payload": {"text": text, "image": image_b64, "new_thread": False}
                    }))
                    print(f"[SENT] 图片: {image_path}, 文字: {text} at {datetime.now()}")
                else:
                    print(f"图片文件不存在: {image_path}")
            else:
                print("用法: /img <图片路径> [文字说明]")
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


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="与智能体进行持续对话的测试客户端")
    parser.add_argument("--agent", required=True, help="目标智能体的ID（例如 agent_xxx）")
    args = parser.parse_args()
    asyncio.run(chat(args.agent))
    # 图片功能示例   /img C:\path\to\test.jpg 这张图里有什么？
    # 清空缓存，开启新话题   /new 我们聊点别的吧