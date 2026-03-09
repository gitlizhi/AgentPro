"""
定时任务执行函数
"""
import logging
import os
import asyncio
from agent.memory import get_memory
from agent.memory_consolidation import consolidate_user_memory
logger = logging.getLogger(__name__)

_reminder_comm = None

def set_reminder_comm(comm):
    global _reminder_comm
    _reminder_comm = comm

async def send_reminder(user_id: str, message: str):
    if _reminder_comm is None:
        raise RuntimeError("Reminder comm not set")
    await _reminder_comm.send_to_agent(user_id, {"text": f"提醒：{message}"})

async def consolidate_all_users():
    """遍历所有用户的记忆文件，并行整理（异步）"""
    memory = get_memory()
    if not os.path.exists(memory.markdown_dir):
        return
    tasks = []
    for filename in os.listdir(memory.markdown_dir):
        if filename.endswith('.md'):
            user_id = filename[:-3]
            tasks.append(consolidate_user_memory(user_id))
    if tasks:
        await asyncio.gather(*tasks)
    print(" 所有用户记忆整理完成")