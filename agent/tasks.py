"""
定时任务执行函数
"""
import logging
import os
import time
import asyncio
from pathlib import Path
from agent.memory import get_memory
from agent.memory_consolidation import consolidate_user_memory
logger = logging.getLogger(__name__)

_reminder_comm = None

def set_reminder_comm(comm):
    global _reminder_comm
    _reminder_comm = comm

async def send_reminder(user_id: str, message: str):
    global _reminder_comm
    if _reminder_comm is None:
        raise RuntimeError("Reminder comm not set")
    # 通过 brain 发送并记录消息
    logger.info(f"提醒执行: user={user_id}, message={message}")
    await _reminder_comm.send_to_agent(user_id, {"text": f"⏰ 提醒：{message}"})

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


def cleanup_old_screenshots(days: int = 30):
    """清理超过指定天数的截图文件和图表文件（同步执行，由 APScheduler 调度）"""
    for keyword in ["screenshots", "diagrams"]:
        clean_dir = Path(__file__).parent.parent / keyword
        if not clean_dir.exists():
            continue
    
        cutoff = time.time() - days * 86400
        deleted = 0
        for filepath in clean_dir.iterdir():
            if not filepath.is_file():
                continue
            try:
                mtime = filepath.stat().st_mtime
                if mtime < cutoff:
                    filepath.unlink()
                    deleted += 1
            except OSError as e:
                logger.warning(f"清理截图文件失败: {filepath.name} - {e}")

    if deleted > 0:
        logger.info(f"截图清理完成: 删除了 {deleted} 个超过 {days} 天的文件")