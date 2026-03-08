import asyncio
import json
import websockets
import argparse

async def chat(agent_id: str):
    """与指定智能体进行持续对话"""
    uri = "ws://localhost:8765"
    local_agent_id = "super_user"
    async with websockets.connect(uri) as ws:
        # 注册客户端
        await ws.send(json.dumps({"type": "register", "agent_id": local_agent_id}))
        resp = await ws.recv()
        print("✅ 已连接到 Hub，注册成功:", resp)

        print(f"🎯 开始与智能体 {agent_id} 对话，输入 'exit' 或 'quit' 退出。")
        while True:
            # 使用 asyncio.to_thread 避免阻塞事件循环
            print()
            print()
            print("===" * 50)
            user_input = await asyncio.to_thread(input, "你: ")
            if user_input.lower() in ("exit", "quit"):
                break
            
            if user_input.startswith("/new"):
                # 发送特殊消息触发新对话，例如 payload 中包含标记
                text = user_input[4:].strip()  # 去掉 "/new" 前缀
                await ws.send(json.dumps({
                    "type": "message",
                    "from": local_agent_id,
                    "to": agent_id,
                    "payload": {"text": text, "new_thread": True}  # 携带标记
                }))

            else:
                # 正常发送
                await ws.send(json.dumps({
                    "type": "message",
                    "from": local_agent_id,
                    "to": agent_id,
                    "payload": {"text": user_input}
                }))

            # 等待智能体回复（单条回复）
            reply_raw = await ws.recv()
            reply = json.loads(reply_raw)
            print(f"🤖 智能体 {reply.get('from')}: {reply.get('payload', {}).get('text')}")
            print("===" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="与智能体进行持续对话的测试客户端")
    parser.add_argument("--agent", required=True, help="目标智能体的ID（例如 agent_xxx）")
    args = parser.parse_args()
    asyncio.run(chat(args.agent))