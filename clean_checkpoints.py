"""
清理短期记忆和群组数据脚本，谨慎使用！

用法：
    # 清除指定的短期记忆线程
    python clean_checkpoints.py --thread "agent_17_super_user_e2746f03-5136-42b4-9982-0173d4957e87"
    
    # 清除所有短期记忆线程
    python clean_checkpoints.py --all
    
    # 清除指定群聊房间及其成员、短期记忆
    python clean_checkpoints.py --room "room_id"
    
    # 清除所有群聊房间、成员及短期记忆
    python clean_checkpoints.py --clear-rooms
    
    # 清除指定智能体的聊天记录
    python clean_checkpoints.py --agent agent_main
"""
import asyncio
import argparse
import sys
import os

# Windows 事件循环兼容性设置
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(__file__))

from agent.db import get_pool, init_db_pool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def delete_room_and_checkpoints(room_id: str):
    """删除指定房间及其成员，并清理对应的短期记忆（group_{room_id}）"""
    await init_db_pool()
    pool = get_pool()
    # 1. 删除房间成员和房间记录
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM room_members WHERE room_id = %s", (room_id,))
            await cur.execute("DELETE FROM rooms WHERE room_id = %s", (room_id,))
    # 2. 删除对应的检查组 thread_id
    group_thread = f"group_{room_id}"
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.adelete_thread(group_thread)
    print(f"✅ 房间 {room_id} 及其成员、短期记忆已清理")

async def clear_all_rooms():
    """删除所有群聊房间、成员及对应的短期记忆"""
    await init_db_pool()
    pool = get_pool()
    # 1. 获取所有房间ID
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT room_id FROM rooms")
            rows = await cur.fetchall()
            room_ids = [row[0] for row in rows]
        # 删除成员和房间记录
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM room_members")
            await cur.execute("DELETE FROM rooms")
    # 2. 删除每个房间的短期记忆
    checkpointer = AsyncPostgresSaver(pool)
    for room_id in room_ids:
        group_thread = f"group_{room_id}"
        await checkpointer.adelete_thread(group_thread)
    print(f"✅ 共清理 {len(room_ids)} 个群聊房间及所有成员记录")

async def clean_thread(thread_id: str):
    """删除指定线程的检查点"""
    await init_db_pool()
    pool = get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.adelete_thread(thread_id)
    print(f"✅ 线程 {thread_id} 清理完成")

async def clean_all_short_memory():
    """删除所有线程的检查点（不清理群聊表）"""
    await init_db_pool()
    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = await cur.fetchall()
        for row in rows:
            thread_id = row[0]
            print(f"正在清理 {thread_id}...")
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.adelete_thread(thread_id)
    print("✅ 所有短期记忆线程清理完成")

async def clean_agent(agent_id: str):
    """删除指定智能体的所有私聊对话记录（持久化消息 + 短期记忆）"""
    await init_db_pool()
    pool = get_pool()
    pattern = f"private_{agent_id}_%"
    
    # 1. 删除 chat_messages 表中所有匹配的消息
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # 先查询匹配的线程数（用于日志）
            await cur.execute("SELECT COUNT(DISTINCT thread_id) FROM chat_messages WHERE thread_id LIKE %s", (pattern,))
            thread_count = (await cur.fetchone())[0]
            # 删除消息记录
            await cur.execute("DELETE FROM chat_messages WHERE thread_id LIKE %s", (pattern,))
            deleted_msg_count = cur.rowcount
            print(f"📋 删除 {deleted_msg_count} 条聊天消息，来自 {thread_count} 个线程")
    
    # 2. 删除短期记忆检查点（checkpoints）中对应的线程
    checkpointer = AsyncPostgresSaver(pool)
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE %s", (pattern,))
            rows = await cur.fetchall()
            thread_ids = [row[0] for row in rows]
    
    deleted_checkpoints = 0
    for tid in thread_ids:
        await checkpointer.adelete_thread(tid)
        deleted_checkpoints += 1
        print(f"🧠 清理短期记忆线程: {tid}")
    
    print(f"✅ 智能体 {agent_id} 的所有私聊记录已清理（消息数: {deleted_msg_count}, 检查点线程: {deleted_checkpoints}）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理短期记忆和群组数据")
    parser.add_argument("--thread", help="指定要删除的短期记忆线程 ID")
    parser.add_argument("--all", action="store_true", help="删除所有短期记忆线程")
    parser.add_argument("--room", help="指定要删除的群聊房间 ID（同时清理对应短期记忆）")
    parser.add_argument("--clear-rooms", action="store_true", help="删除所有群聊房间及成员（同时清理对应短期记忆）")
    parser.add_argument("--agent", help="指定要清理的智能体 ID（如 'reminder_bot'），将删除该智能体的全部私聊记录（包括持久化消息和短期记忆）")
    args = parser.parse_args()

    if args.thread:
        asyncio.run(clean_thread(args.thread))
    elif args.all:
        asyncio.run(clean_all_short_memory())
    elif args.room:
        asyncio.run(delete_room_and_checkpoints(args.room))
    elif args.clear_rooms:
        asyncio.run(clear_all_rooms())
    elif args.agent:
        asyncio.run(clean_agent(args.agent))
    else:
        print("请指定 --thread、--all、--room 或 --clear-rooms")