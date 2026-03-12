"""
调度器模块
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from config import config

# 全局调度器实例
_scheduler = None

def init_scheduler():
    """初始化 APScheduler 实例，使用 PostgreSQL 存储"""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    # 配置作业存储
    jobstore = SQLAlchemyJobStore(url=config.db.postgres_uri_sync)  # 需要使用同步连接字符串

    # 配置执行器（异步执行器，支持异步作业函数）
    executors = {
        'default': AsyncIOExecutor()
    }

    # 创建调度器
    _scheduler = AsyncIOScheduler(
        jobstores={'default': jobstore},
        executors=executors,
        timezone = config.scheduler.scheduler_timezone
    )
    return _scheduler

def get_scheduler():
    """获取全局调度器实例（确保已初始化）"""
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized. Call init_scheduler() first.")
    return _scheduler