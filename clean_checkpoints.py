"""
清理短期记忆脚本，谨慎使用！

清除指定的线程id      python clean_checkpoints.py --thread "agent_17_super_user_e2746f03-5136-42b4-9982-0173d4957e87"
全部清楚            python clean_checkpoints.py --all

"""
import asyncio
import argparse
import sys
import os

# Windows 事件循环兼容性设置
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(__file__))

from agent.db import get_pool, init_db_pool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def clean_thread(thread_id: str):
    """删除指定线程的检查点"""
    await init_db_pool()
    pool = get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.adelete_thread(thread_id)
    print(f"✅ 线程 {thread_id} 清理完成")

async def clean_all():
    """删除所有线程的检查点"""
    await init_db_pool()
    pool = get_pool()
    async with pool.connection() as conn:
        # 使用游标执行查询
        async with conn.cursor() as cur:
            await cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = await cur.fetchall()
        for row in rows:
            thread_id = row[0]  # 假设返回的是元组，第一个元素是 thread_id
            print(f"正在清理 {thread_id}...")
            # 每个删除操作可以重用同一个 checkpointer 实例，但为了保险，我们每次都新建
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.adelete_thread(thread_id)
    print("✅ 所有线程清理完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", help="指定要删除的 thread_id")
    parser.add_argument("--all", action="store_true", help="删除所有线程的检查点")
    args = parser.parse_args()

    if args.thread:
        asyncio.run(clean_thread(args.thread))
    elif args.all:
        asyncio.run(clean_all())
    else:
        print("请指定 --thread 或 --all")