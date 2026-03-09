"""
Agent启动入口
"""
import asyncio
import sys
import uuid
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

    # 初始化调度器
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
    # print(" 已调度每日记忆整理任务（每日3点）")
    
    # 创建提醒机器人的通讯实例
    async def dummy_handler(data):
        """提醒机器人不需要处理收到的消息"""
        pass

    reminder_comm = Communication(
        agent_id="reminder_bot",
        hub_url=f"ws://{config.HUB_HOST}:{config.HUB_PORT}",
        on_message=dummy_handler
    )

    # 将提醒机器人的通讯对象设置到 tasks 模块中，供 send_reminder 使用
    set_reminder_comm(reminder_comm)

    # 创建主 Agent
    agent = Agent(
        # agent_id=f"agent_{uuid.uuid4()}",
        agent_id=f"agent_17",
        db_pool=pool,
        model_config_key="zhipu",
    )

    try:
        # 并发运行主 Agent 和提醒机器人的连接
        await asyncio.gather(
            agent.run(),
            reminder_comm.connect()   # reminder_comm.connect() 会一直运行，直到关闭
        )
    except KeyboardInterrupt:
        await agent.stop()
        await reminder_comm.close()
    finally:
        scheduler.shutdown()
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())