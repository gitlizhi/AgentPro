"""
主程序入口
"""
import os
import asyncio
import sys
import logging
from agent.core import Agent
from agent.db import init_db_pool, close_db_pool


# 设置 Windows 事件循环
if sys.platform == 'win32':
	asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO)


async def main():
	# 初始化数据库连接池
	pool = await init_db_pool()
	
	# 创建 Agent 实例（技能会自动从注册中心加载）
	agent = Agent(
		# agent_id=f"agent_{uuid.uuid4()}",
		agent_id=f"agent_11",
		db_pool=pool,
		model_config_key="zhipu",
	)
	
	try:
		await agent.run()
	except KeyboardInterrupt:
		await agent.stop()
	finally:
		await close_db_pool()
		logging.info("Database pool closed")


if __name__ == "__main__":
	asyncio.run(main())