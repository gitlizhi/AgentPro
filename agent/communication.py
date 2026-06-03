"""
通讯模块 - 带自动重连的 WebSocket 客户端
"""
import asyncio
import json
import logging
import os
import time
import websockets
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

# 重连策略：指数退避
_RECONNECT_BASE_DELAY = 1.0   # 首次重连等待秒数
_RECONNECT_MAX_DELAY = 60.0    # 最大重连等待秒数
_RECONNECT_BACKOFF = 2.0       # 退避乘数
_RECONNECT_MAX_TOTAL = 60.0    # 累计断连超过此秒数则退出进程


class Communication:
    def __init__(self, agent_id: str, hub_url: str,
                 on_message: Callable[[dict], Awaitable[None]]):
        self.agent_id = agent_id
        self.hub_url = hub_url
        self.on_message = on_message
        self.websocket = None
        self._running = False

    async def connect(self):
        """连接 Hub 并持续监听消息，连接断开时自动重连。
        如果累计断连超过 _RECONNECT_MAX_TOTAL 秒，则直接退出进程。"""
        self._running = True
        attempt = 0
        disconnect_start = 0.0  # 首次断连的时间戳，0 表示已连接

        while self._running:
            attempt += 1
            try:
                async with websockets.connect(
                    self.hub_url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=20 * 1024 * 1024
                ) as ws:
                    self.websocket = ws
                    # 重置退避计数器和断连计时（成功连接后）
                    attempt = 0
                    disconnect_start = 0.0

                    # 发送注册信息
                    await self.send({
                        "type": "register",
                        "agent_id": self.agent_id
                    })
                    logger.info(f"Agent {self.agent_id} connected to hub")

                    # 接收循环：非致命错误不退出，继续监听
                    while self._running:
                        try:
                            message = await ws.recv()
                            data = json.loads(message)
                            if isinstance(data, dict):
                                await self.on_message(data)
                            else:
                                logger.warning(f"Received non-dict message, ignoring: {data}")
                        except websockets.exceptions.ConnectionClosed as e:
                            logger.warning(f"Connection closed: {e}")
                            break  # 退出接收循环，触发重连
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON received: {e}")
                            continue  # 单条消息损坏不退出
                        except Exception as e:
                            logger.error(f"Error in receive loop: {e}", exc_info=True)
                            # 非致命错误继续监听，避免因单条消息处理异常断开
                            continue

            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.InvalidURI,
                    OSError) as e:
                logger.warning(f"WebSocket connection failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected connection error: {e}", exc_info=True)
            finally:
                self.websocket = None

            # 准备重连
            if self._running:
                now = time.time()
                if disconnect_start == 0.0:
                    disconnect_start = now
                elif now - disconnect_start > _RECONNECT_MAX_TOTAL:
                    logger.critical(
                        f"Agent {self.agent_id} 累计断连超过 {_RECONNECT_MAX_TOTAL}s，退出进程"
                    )
                    os._exit(1)

                delay = min(_RECONNECT_BASE_DELAY * (_RECONNECT_BACKOFF ** (attempt - 1)),
                            _RECONNECT_MAX_DELAY)
                logger.info(f"Reconnecting in {delay:.1f}s (attempt {attempt}, disconnected for {now - disconnect_start:.0f}s)...")
                await asyncio.sleep(delay)

        logger.info(f"Agent {self.agent_id} communication loop exited")

    async def send(self, data: dict):
        """发送消息到 Hub"""
        if self.websocket:
            await self.websocket.send(json.dumps(data, ensure_ascii=False))

    async def send_to_agent(self, target_agent_id: str, payload: dict, ticket_id: str = None):
        """发送消息给指定智能体，可选关联到 TDP 工单。"""
        envelope = {
            "type": "message",
            "from": self.agent_id,
            "to": target_agent_id,
            "payload": payload
        }
        if ticket_id:
            envelope["ticket_id"] = ticket_id
        await self.send(envelope)

    async def send_to_room(self, room_id: str, payload: dict):
        """发送消息给指定房间"""
        await self.send({
            "type": "group_message",
            "from": self.agent_id,
            "room_id": room_id,
            "payload": payload
        })

    async def close(self):
        self._running = False
        if self.websocket:
            await self.websocket.close()
