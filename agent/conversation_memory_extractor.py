"""
对话记忆自动提取器
功能：定期从短期对话历史中提取值得长期记忆的事实，自动写入 LongTermMemory。
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set

from agent.utils import call_big_model_chat
from agent.memory import get_memory
from agent.db import get_pool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from config import config

# ---------- 配置 ----------
EXTRACT_INTERVAL_SECONDS = 300       # 每5分钟扫描一次
MIN_MESSAGES_BEFORE_EXTRACT = 6      # 至少积累6条消息才触发提取（避免频繁调用LLM）
LOOKBACK_DAYS = 1                    # 只处理最近1天内有活动的线程

# 全局状态：记录每个 (user_id, thread_id) 上次提取的消息数量或时间
_last_extract_info = {}  # key: (user_id, thread_id) -> {"last_msg_count": int, "last_extract_time": datetime}

async def get_active_threads(since: datetime, pool) -> List[tuple]:
    """
    从 PostgreSQL checkpoint 中获取最近活跃的 thread_id 及其所属 user_id。
    返回列表: [(thread_id, user_id), ...]
    """
    # checkpoint 表结构通常为: thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
    # metadata 中可能存储用户信息，但这里简化：从你的业务表 rooms? 或者从 thread_id 格式解析
    # 假设 thread_id 格式为 f"{agent_id}_{user_id}_{uuid}"，那么我们可以解析出 user_id
    # 为简单起见，这里提供一个通用方法：直接查询 checkpoint 表，获取最近更新的 thread_id，
    # 然后通过你已有的 metadata 或直接返回 thread_id，后续再通过 thread_id 提取 user_id
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE checkpoint_id IN (
                    SELECT DISTINCT ON (thread_id) checkpoint_id
                    FROM checkpoints
                    WHERE thread_id IS NOT NULL
                    ORDER BY thread_id, checkpoint_id DESC
                )
            """)
            rows = await cur.fetchall()
    # 从 thread_id 解析 user_id（只记录超级用户的）
    result = []
    for (thread_id,) in rows:
        if 'super_user' in thread_id:
            result.append((thread_id, 'super_user'))
    return result


async def get_recent_messages(thread_id: str, since: datetime) -> List[Dict]:
    """
    从检查点中获取指定 thread_id 在 since 时间之后的消息（仅 human 和 ai）。
    返回格式: [{"role": "user" 或 "assistant", "content": "..."}, ...]
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from agent.db import get_pool
    
    pool = get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    # 注意：如果表未初始化，需要调用 await checkpointer.setup()，但应在应用启动时做一次，避免重复调用
    
    config_obj = {"configurable": {"thread_id": thread_id}}
    result = await checkpointer.aget(config_obj)
    if result is None:
        return []
    
    # 兼容两种情况：如果是字典，取 'checkpoint' 字段；否则取 .checkpoint 属性
    if isinstance(result, dict):
        checkpoint = result.get("checkpoint", result)
    else:
        checkpoint = getattr(result, "checkpoint", result)
    
    if not checkpoint:
        return []
    
    channel_values = checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])
    
    recent = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type in ("human", "ai"):
            # 尝试获取时间戳（消息可能没有）
            timestamp = getattr(msg, 'timestamp', None)
            if timestamp and isinstance(timestamp, datetime) and timestamp >= since:
                recent.append({
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                })
            elif not timestamp:
                # 如果没有时间戳，保守起见也加入（可能会重复，但后续去重会处理）
                recent.append({
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                })
    return recent

async def extract_facts_from_conversation(messages: List[Dict]) -> List[str]:
    """调用 LLM 从对话中提取值得长期记忆的事实"""
    if not messages:
        return []
    # 限制消息数量，避免超出上下文
    if len(messages) > 30:
        messages = messages[-30:]
    prompt = f"""
你是一个智能记忆提取器。请从以下对话中提取出**值得长期记住的事实**，例如用户的偏好、重要信息、重复出现的行为等。
只输出一个 JSON 数组，每个元素是一条简短的事实字符串。如果没有任何值得记忆的内容，输出空数组。

对话记录：
{json.dumps(messages, ensure_ascii=False, indent=2)}

输出格式：["事实1", "事实2"]
"""
    try:
        response = await call_big_model_chat(prompt, model=config.model.default_model,
                                             temperature=0.2, is_json=True)
        content = response["choices"][0]["message"]["content"]
        # 清理 JSON
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        facts = json.loads(content)
        if not isinstance(facts, list):
            return []
        return [str(f).strip() for f in facts if str(f).strip()]
    except Exception as e:
        print(f"[EXTRACT] Error extracting facts: {e}")
        return []

# 在 conversation_memory_extractor.py 中修改
async def process_user_conversation(user_id: str, thread_id: str, force: bool = False):
    """处理某个用户的一个对话线程，提取新事实并存储"""
    memory = get_memory()
    # 获取上次提取的消息数量
    last_msg_count_key = f"last_msg_count_{user_id}_{thread_id}"
    last_msg_count_str = memory.get_user_metadata("__system__", last_msg_count_key)
    last_msg_count = int(last_msg_count_str) if last_msg_count_str else 0

    # 获取当前消息列表（完整）
    messages = await get_recent_messages(thread_id, datetime.min)  # 获取全部消息
    if not messages:
        return

    current_msg_count = len(messages)
    # 如果没有新消息，跳过
    if not force and current_msg_count <= last_msg_count:
        return

    # 提取新增的消息（从 last_msg_count 到结尾）
    new_messages = messages[last_msg_count:]
    if not new_messages:
        return

    # 调用 LLM 提取事实
    new_facts = await extract_facts_from_conversation(new_messages)
    if new_facts:
        # 去重（与已有长期记忆比较）
        existing_facts = set()
        for fact in new_facts:
            similar = memory.query_relevant(fact, user_id, n_results=2)
            for sim in similar:
                if sim["content"].strip() == fact.strip():
                    existing_facts.add(fact)
                    break
        final_facts = [f for f in new_facts if f not in existing_facts]
        if final_facts:
            metadata = {
                "source": "auto_extracted",
                "thread_id": thread_id,
                "extracted_at": datetime.now().isoformat()
            }
            memory.add_facts_batch(final_facts, user_id, metadata)
            print(f"[MEM] Extracted {len(final_facts)} new facts for user {user_id} from thread {thread_id} (new messages: {len(new_messages)})")

    # 更新已处理的消息数量
    memory.set_user_metadata("__system__", last_msg_count_key, str(current_msg_count))

async def conversation_memory_worker():
    """后台工作器：定期扫描活跃线程并提取记忆"""
    print("[MEM] Starting conversation memory extractor worker...")
    pool = get_pool()
    while True:
        try:
            # 扫描最近活跃的线程
            since = datetime.now() - timedelta(days=LOOKBACK_DAYS)
            active_threads = await get_active_threads(since, pool)
            print(f"[MEM] Found {len(active_threads)} active threads to check")
            for thread_id, user_id in active_threads:
                await process_user_conversation(user_id, thread_id)
            # 等待下一次扫描
            await asyncio.sleep(EXTRACT_INTERVAL_SECONDS)
        except Exception as e:
            print(f"[MEM] Error in conversation memory worker: {e}")
            await asyncio.sleep(EXTRACT_INTERVAL_SECONDS)