"""
Agent启动入口
"""
import asyncio
import sys
import logging
import uuid
import argparse
from agent.reflection import reflection_worker
from agent.core import Agent
from agent.db import init_db_pool, close_db_pool
from agent.scheduler import init_scheduler
from agent.communication import Communication
from agent.tasks import set_reminder_comm
from agent.tasks import consolidate_all_users
from agent.skill_version_manager import consolidate_skills_job
from agent.conversation_memory_extractor import conversation_memory_worker
from config import config
from dotenv import load_dotenv
load_dotenv()

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", type=str, default=None, help="自定义 Agent ID（如果指定，则只启动一个 Agent，且不启动全局组件）")
    parser.add_argument("--system-prompt", type=str, default=None, help="自定义系统提示（仅子 Agent 模式有效）")
    args = parser.parse_args()

    pool = await init_db_pool()

    # ---------- 子 Agent 模式（由其他 Agent 启动） ----------
    if args.agent_id is not None:
        # 只启动单个 Agent，不启动任何全局组件（调度器、提醒机器人、记忆任务等）`
        agent = Agent(
            agent_id=args.agent_id,
            db_pool=pool,
            custom_system_prompt=args.system_prompt,
        )
        # 直接运行 Agent，不需要 asyncio.gather 其他任务
        await agent.run()
        return

    # ---------- 主程序模式（手动启动） ----------
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
    
    # 创建提醒机器人的通讯实例
    async def dummy_handler(data):
        pass
    
    async def skill_consolidation_loop():
        while True:
            await consolidate_skills_job()
            
    reminder_comm = Communication(
        agent_id="reminder_bot",
        hub_url=f"ws://{config.hub.hub_host}:{config.hub.hub_port}",
        on_message=dummy_handler
    )
    set_reminder_comm(reminder_comm)

    # 创建多个 Agent 实例（原有行为）
    agents = []
    for i in range(config.agent.num_agents):
        agent = Agent(
            # agent_id=f"{config.agent.agent_id_prefix}_{uuid.uuid4()}",
            agent_id=f"{config.agent.agent_id_prefix}_main",
            db_pool=pool,
        )
        agents.append(agent)
    # 启动对话记忆提取器（后台任务）
    memory_extractor_task = asyncio.create_task(conversation_memory_worker())
    # 启动后台反思 Worker
    reflection_task = asyncio.create_task(reflection_worker())
    # 启动遗忘巩固后台任务
    consolidation_task = asyncio.create_task(skill_consolidation_loop())
    # 并发运行所有任务
    tasks = [agent.run() for agent in agents] + [reminder_comm.connect(), reflection_task, consolidation_task, memory_extractor_task]
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        # 取消后台任务
        reflection_task.cancel()
        memory_extractor_task.cancel()
        try:
            await reflection_task
        except asyncio.CancelledError:
            pass
        try:
            await memory_extractor_task
        except asyncio.CancelledError:
            pass
        for agent in agents:
            await agent.stop()
        await reminder_comm.close()
    finally:
        scheduler.shutdown()
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())# 启动2个智能体，1个负责搜索资料，整理文件，1个负责针对资料生成可视化报告。