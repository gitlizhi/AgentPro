"""
Agent启动入口
"""
import asyncio
import sys
import logging
from agent.core import Agent
from agent.db import init_db_pool, close_db_pool
from agent.scheduler import init_scheduler
from agent.communication import Communication
from agent.tasks import set_reminder_comm
from agent.tasks import consolidate_all_users
import config

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO)

async def main():
    pool = await init_db_pool()

    # 初始化调度器（所有 Agent 共享）
    scheduler = init_scheduler()
    scheduler.start()
    
    scheduler.add_job(
        consolidate_all_users,
        trigger='cron',
        hour=3,
        minute=0,
        id='memory_consolidation_daily',
        replace_existing=True
    )
    
    # 创建提醒机器人的通讯实例（只需一个）
    async def dummy_handler(data):
        pass

    reminder_comm = Communication(
        agent_id="reminder_bot",
        hub_url=f"ws://{config.HUB_HOST}:{config.HUB_PORT}",
        on_message=dummy_handler
    )
    set_reminder_comm(reminder_comm)

    # 创建多个 Agent 实例
    num_agents = 3  # 你想要启动的 Agent 数量
    agents = []
    for i in range(1, num_agents + 1):
        agent = Agent(
            agent_id=f"agent_{i}",  # 确保每个 ID 唯一
            db_pool=pool,
            model_config_key="zhipu",
        )
        agents.append(agent)

    # 将所有 Agent 的 run 任务和提醒机器人的 connect 任务合并
    tasks = [agent.run() for agent in agents] + [reminder_comm.connect()]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        # 停止所有 Agent
        for agent in agents:
            await agent.stop()
        await reminder_comm.close()
    finally:
        scheduler.shutdown()
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())