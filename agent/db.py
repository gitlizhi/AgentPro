# agent/db.py
import asyncio
from typing import Optional
import psycopg
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import config

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
    safe_uri = default_uri.replace(parsed.netloc.split('@')[-1] if '@' in parsed.netloc else parsed.netloc, '***')
    print(f"Connecting to default database with URI: {safe_uri}")
    
    conn = None
    try:
        conn = await psycopg.AsyncConnection.connect(default_uri)
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            exists = await cur.fetchone()
            if not exists:
                await cur.execute(f'CREATE DATABASE "{target_db}"')
                print(f"✅ Database '{target_db}' created.")
            else:
                print(f"ℹ️ Database '{target_db}' already exists.")
    except Exception as e:
        print(f"❌ Error connecting to default database: {e}")
        raise
    finally:
        if conn:
            await conn.close()


# agent/db.py
async def init_db_pool():
    global _pool
    if _pool is None:
        await ensure_database_exists(config.POSTGRES_URI)

        _pool = AsyncConnectionPool(
            config.POSTGRES_URI,
            min_size=0,
            max_size=10,
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
            print("✅ 连接池测试成功")
        except Exception as e:
            print(f"❌ 连接池测试失败: {e}")
            await _pool.close()
            _pool = None
            raise

        # 初始化检查点表
        checkpointer = AsyncPostgresSaver(_pool)
        await checkpointer.setup()
        print("✅ 数据库连接池已初始化，表已创建")
    return _pool


async def close_db_pool():
    """关闭数据库连接池"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✅ Database pool closed.")


def get_pool() -> AsyncConnectionPool:
    """获取全局连接池（确保已初始化）"""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db_pool() first.")
    return _pool