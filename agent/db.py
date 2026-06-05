"""
数据库连接池
"""
from typing import Optional
import logging
import psycopg
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from config import config

logger = logging.getLogger(__name__)

_pool: Optional[AsyncConnectionPool] = None


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
            
        # ---------- 新增：创建 reminders 表 ----------
        async with _pool.connection() as conn:
            await conn.execute("""
                    CREATE TABLE IF NOT EXISTS reminders (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        reminder_time TIMESTAMP NOT NULL,
                        message TEXT NOT NULL,
                        triggered BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            # 创建索引以加速查询
            await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_reminders_user_time
                    ON reminders (user_id, reminder_time) WHERE NOT triggered
                """)
            # print("✅ 已确保 reminders 表存在")
            await conn.execute("""
                    CREATE TABLE IF NOT EXISTS rooms (
                        room_id VARCHAR(255) PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            await conn.execute("""
                    CREATE TABLE IF NOT EXISTS room_members (
                    room_id VARCHAR(255) REFERENCES rooms(room_id) ON DELETE CASCADE,
                    agent_id VARCHAR(255),
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (room_id, agent_id)
                );
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    thread_id VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id)")

            # ── 编排系统表 ──
            await conn.execute("""
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
            """)
            await conn.execute("""
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
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orch_subtasks_ticket
                ON orchestration_subtasks (ticket_id) WHERE ticket_id IS NOT NULL
            """)
            # ── 委托系统表 ──
            await conn.execute("""
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
            """)
            # 为已有数据库添加列（如果不存在）
            await conn.execute("""
                DO $$ BEGIN
                    ALTER TABLE delegation_tickets ADD COLUMN orchestration_plan_id VARCHAR(64);
                EXCEPTION WHEN duplicate_column THEN
                    NULL;
                END $$;
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_delegation_tickets_pair
                ON delegation_tickets (issuer, assignee)
                WHERE state NOT IN ('closed','declined','timed_out','cancelled')
            """)

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