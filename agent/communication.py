"""
通讯模块
"""
import asyncio
import json
import logging
import websockets
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class Communication:
	def __init__(self, agent_id: str, hub_url: str,
				 on_message: Callable[[dict], Awaitable[None]]):
		self.agent_id = agent_id
		self.hub_url = hub_url
		self.on_message = on_message
		self.websocket = None
		self._running = False
	
	async def connect(self):
		"""连接 Hub 并持续监听消息"""
		self._running = True
		try:
			async with websockets.connect(self.hub_url) as ws:
				self.websocket = ws
				# 发送注册信息
				await self.send({
					"type": "register",
					"agent_id": self.agent_id
				})
				logger.info(f"Agent {self.agent_id} connected to hub")
				
				# 监听消息
				while self._running:
					try:
						message = await ws.recv()
						data = json.loads(message)
						await self.on_message(data)
					except websockets.exceptions.ConnectionClosed:
						logger.warning("Connection closed")
						break
					except Exception as e:
						logger.error(f"Error receiving message: {e}")
		finally:
			self._running = False
			self.websocket = None
	
	async def send(self, data: dict):
		"""发送消息到 Hub"""
		if self.websocket:
			await self.websocket.send(json.dumps(data, ensure_ascii=False))
	
	async def send_to_agent(self, target_agent_id: str, payload: dict):
		"""发送消息给指定智能体"""
		await self.send({
			"type": "message",
			"from": self.agent_id,
			"to": target_agent_id,
			"payload": payload
		})
	
	async def close(self):
		self._running = False
		if self.websocket:
			await self.websocket.close()