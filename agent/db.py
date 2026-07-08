"""
数据库连接池
"""
from typing import Optional
import logging
import psycopg
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from config import config

logger = logging.getLogger(__name__)

_pool: Optional[AsyncConnectionPool] = None
_sync_pool: Optional[ConnectionPool] = None

# ── 统一 DDL（所有表定义集中管理）──────────────────────────────
_SCHEMA_DDL = [
    # === reminders ===
    """
    CREATE TABLE IF NOT EXISTS reminders (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        reminder_time TIMESTAMP NOT NULL,
        message TEXT NOT NULL,
        triggered BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_reminders_user_time ON reminders (user_id, reminder_time) WHERE NOT triggered",

    # === rooms ===
    """
    CREATE TABLE IF NOT EXISTS rooms (
        room_id VARCHAR(255) PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS room_members (
        room_id VARCHAR(255) REFERENCES rooms(room_id) ON DELETE CASCADE,
        agent_id VARCHAR(255),
        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (room_id, agent_id)
    )
    """,

    # === chat ===
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        thread_id VARCHAR(255) NOT NULL,
        role VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        image TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id)",
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created ON chat_messages(thread_id, created_at DESC)",

    # === conversation_threads ===
    """
    CREATE TABLE IF NOT EXISTS conversation_threads (
        id SERIAL PRIMARY KEY,
        thread_id VARCHAR(255) UNIQUE NOT NULL,
        agent_id VARCHAR(255) NOT NULL,
        title VARCHAR(500) DEFAULT 'New Chat',
        user_id VARCHAR(255) DEFAULT 'super_user',
        is_archived BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_threads_agent ON conversation_threads(agent_id, user_id)",
    "CREATE INDEX IF NOT EXISTS idx_threads_updated ON conversation_threads(updated_at DESC)",

    # === 迁移：为已有表添加可能缺失的列 ===
    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS image TEXT",

    # === 编排系统 ===
    """
    CREATE TABLE IF NOT EXISTS orchestration_plans (
        plan_id VARCHAR(64) PRIMARY KEY,
        description TEXT NOT NULL,
        issuer VARCHAR(255) NOT NULL,
        state VARCHAR(32) NOT NULL DEFAULT 'planning',
        completed_count INTEGER DEFAULT 0,
        failed_count INTEGER DEFAULT 0,
        created_at DOUBLE PRECISION,
        updated_at DOUBLE PRECISION
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orchestration_subtasks (
        plan_id VARCHAR(64) REFERENCES orchestration_plans(plan_id) ON DELETE CASCADE,
        subtask_id VARCHAR(16) NOT NULL,
        description TEXT NOT NULL,
        assigned_to VARCHAR(255),
        status VARCHAR(32) NOT NULL DEFAULT 'pending',
        depends_on JSONB DEFAULT '[]',
        result TEXT,
        ticket_id VARCHAR(32),
        suggested_role VARCHAR(128),
        PRIMARY KEY (plan_id, subtask_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orch_subtasks_ticket
    ON orchestration_subtasks (ticket_id) WHERE ticket_id IS NOT NULL
    """,

    # === 委托系统 ===
    """
    CREATE TABLE IF NOT EXISTS delegation_tickets (
        ticket_id VARCHAR(32) PRIMARY KEY,
        issuer VARCHAR(255) NOT NULL,
        assignee VARCHAR(255) NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        expected_output TEXT NOT NULL DEFAULT '',
        max_rounds INTEGER DEFAULT 8,
        state VARCHAR(32) NOT NULL DEFAULT 'pending',
        round_count INTEGER DEFAULT 0,
        clarification_count INTEGER DEFAULT 0,
        result_summary TEXT,
        cancel_reason TEXT,
        cancelled_by VARCHAR(255),
        thread_id VARCHAR(255),
        created_at DOUBLE PRECISION,
        accepted_at DOUBLE PRECISION,
        completed_at DOUBLE PRECISION,
        last_activity DOUBLE PRECISION DEFAULT 0,
        orchestration_plan_id VARCHAR(64)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_delegation_tickets_pair
    ON delegation_tickets (issuer, assignee)
    WHERE state NOT IN ('closed','declined','timed_out','cancelled')
    """,
    # 为已存在的表补充 orchestration_plan_id 列（兼容旧数据）
    """
    DO $$ BEGIN
        ALTER TABLE delegation_tickets ADD COLUMN orchestration_plan_id VARCHAR(64);
    EXCEPTION WHEN duplicate_column THEN
        NULL;
    END $$
    """,

    # === Loop Engineering 迁移：orchestration_subtasks 新增列 ===
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS reviewer_agent VARCHAR(255)",
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS review_feedback TEXT",
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0",
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS max_retries INTEGER DEFAULT 3",
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS blocked_reason TEXT",
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS attempts JSONB DEFAULT '[]'",
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS worker_system_prompt TEXT",
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS reviewer_system_prompt TEXT",
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS escalated_at TIMESTAMP",
    "ALTER TABLE orchestration_subtasks ADD COLUMN IF NOT EXISTS skipped BOOLEAN DEFAULT FALSE",

    # === Loop Engineering 迁移：orchestration_plans 新增列 ===
    "ALTER TABLE orchestration_plans ADD COLUMN IF NOT EXISTS project_overview TEXT DEFAULT ''",
    "ALTER TABLE orchestration_plans ADD COLUMN IF NOT EXISTS critical_decisions JSONB DEFAULT '[]'",
    "ALTER TABLE orchestration_plans ADD COLUMN IF NOT EXISTS escalation_log JSONB DEFAULT '[]'",
    "ALTER TABLE orchestration_plans ADD COLUMN IF NOT EXISTS created_from_clarification BOOLEAN DEFAULT FALSE",
    "ALTER TABLE orchestration_plans ADD COLUMN IF NOT EXISTS agent_pool JSONB DEFAULT '{}'",
    "ALTER TABLE orchestration_plans ADD COLUMN IF NOT EXISTS degradation_level INTEGER DEFAULT 0",
]


async def _run_ddl_async(conn) -> None:
    """在异步连接上执行全部 DDL"""
    for sql in _SCHEMA_DDL:
        await conn.execute(sql)


def _run_ddl_sync(conn) -> None:
    """在同步连接上执行全部 DDL"""
    for sql in _SCHEMA_DDL:
        conn.execute(sql)


async def ensure_database_exists(uri: str):
    """检查并创建数据库（如果不存在）"""
    from urllib.parse import urlparse, urlunparse
    
    # 解析原始 URI
    parsed = urlparse(uri)
    # 获取路径部分（数据库名）
    path_parts = parsed.path.split('/')
    if len(path_parts) < 2:
        raise ValueError("URI 路径中必须包含数据库名")
    target_db = path_parts[1]  # 假设路径是 "/dbname"
    
    # 构建连接到默认数据库（postgres）的 URI
    # 将路径部分替换为 "/postgres"
    new_path = '/postgres'
    if len(path_parts) > 2:
        # 如果有额外路径段，保留（不太可能，但安全处理）
        new_path += '/' + '/'.join(path_parts[2:])
    
    # 创建新的解析组件元组，替换 path
    new_parsed = parsed._replace(path=new_path)
    default_uri = urlunparse(new_parsed)
    
    # 打印调试信息（注意隐藏密码）
    # safe_uri = default_uri.replace(parsed.netloc.split('@')[-1] if '@' in parsed.netloc else parsed.netloc, '***')
    # print(f"Connecting to default database with URI: {safe_uri}")
    
    conn = None
    try:
        conn = await psycopg.AsyncConnection.connect(default_uri)
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            exists = await cur.fetchone()
            if not exists:
                await cur.execute(f'CREATE DATABASE "{target_db}"')
            #     print(f"✅ Database '{target_db}' created.")
            # else:
            #     print(f"ℹ️ Database '{target_db}' already exists.")
    except Exception as e:
        logger.error(f"Error connecting to default database: {e}")
        raise
    finally:
        if conn:
            await conn.close()


async def init_db_pool():
    global _pool
    if _pool is None:
        await ensure_database_exists(config.db.postgres_uri)
        _pool = AsyncConnectionPool(
            config.db.postgres_uri,
            min_size=config.db.postgres_pool_min_size,
            max_size=config.db.postgres_pool_max_size,
            timeout=config.db.postgres_pool_timeout,
            open=False,
            kwargs={
                "autocommit": True,
                "connect_timeout": 10,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
            max_idle=300,
            max_lifetime=3600,
            num_workers=2,
            reconnect_timeout=10,
        )
        await _pool.open()

        # 测试连接
        try:
            async with _pool.connection() as conn:
                await conn.execute("SELECT 1")
            # print("✅ 连接池测试成功")
        except Exception as e:
            # print(f"❌ 连接池测试失败: {e}")
            await _pool.close()
            _pool = None
            raise
            
        # 执行统一 DDL
        async with _pool.connection() as conn:
            await _run_ddl_async(conn)

        # 初始化检查点表
        checkpointer = AsyncPostgresSaver(_pool)
        await checkpointer.setup()
        logger.info("数据库连接池已初始化，表已创建")
    return _pool


async def close_db_pool():
    """关闭数据库连接池"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed.")


def get_pool() -> AsyncConnectionPool:
    """获取全局连接池（确保已初始化）"""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db_pool() first.")
    return _pool


# ── 同步连接池（供 client.py FastAPI 使用）───────────────────


def init_sync_pool():
    """初始化同步连接池并创建表"""
    global _sync_pool
    if _sync_pool is None:
        _sync_pool = ConnectionPool(
            config.db.postgres_uri,
            min_size=2,
            max_size=10,
            timeout=30,
            max_idle=300,
            max_lifetime=3600,
            kwargs={
                "autocommit": True,
                "connect_timeout": 10,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )
        try:
            with _sync_pool.connection() as conn:
                conn.execute("SELECT 1")
        except Exception:
            _sync_pool.close()
            _sync_pool = None
            raise

        with _sync_pool.connection() as conn:
            _run_ddl_sync(conn)

        logger.info("同步连接池已初始化，表已创建")
    return _sync_pool


def get_sync_pool() -> ConnectionPool:
    """获取全局同步连接池（确保已初始化）"""
    if _sync_pool is None:
        raise RuntimeError("Sync pool not initialized. Call init_sync_pool() first.")
    return _sync_pool


def close_sync_pool():
    """关闭同步连接池"""
    global _sync_pool
    if _sync_pool:
        _sync_pool.close()
        _sync_pool = None
        logger.info("Sync pool closed.")