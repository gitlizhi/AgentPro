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

async def get_active_threads(pool) -> List[tuple]:
    """获取所有包含 super_user 的 thread_id"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE thread_id LIKE '%super_user%'
            """)
            rows = await cur.fetchall()
    return [(tid, 'super_user') for (tid,) in rows]

async def get_all_messages(thread_id: str) -> List[Dict]:
    """获取 thread 中所有 human 和 ai 消息，按时间顺序返回"""
    pool = get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    config_obj = {"configurable": {"thread_id": thread_id}}
    result = await checkpointer.aget(config_obj)
    if not result:
        return []
    if isinstance(result, dict):
        checkpoint = result.get("checkpoint", result)
    else:
        checkpoint = getattr(result, "checkpoint", result)
    if not checkpoint:
        return []
    channel_values = checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])
    # 过滤并转换格式
    recent = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type in ("human", "ai"):
            recent.append({
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content
            })
    return recent

async def extract_memories_from_conversation(messages: List[Dict]) -> Dict[str, List]:
    """
    返回: {
        "facts": ["用户喜欢冰美式", ...],
        "events": [
            {"summary": "用户要求总结新闻，我搜索后返回结果，用户满意", "outcome": "success"},
            ...
        ]
    }
    """
    if not messages:
        return {"facts": [], "events": []}
    if len(messages) > 30:
        messages = messages[-30:]
    prompt = f"""
你是一个智能记忆提取器。请分析以下对话，提取两种信息：

1. **语义事实**：用户的长期偏好、个人信息、重要约定等。每条用简短句子描述。
2. **事件记忆**：智能体执行的重要任务、动作、结果以及用户的反馈。每条应包含：做了什么、结果如何（成功/失败）、用户是否满意。

输出格式为 JSON 对象：
{{
  "facts": ["事实1", "事实2"],
  "events": [
    {{"summary": "事件描述", "outcome": "success/failure/neutral"}},
    ...
  ]
}}

如果没有某类信息，对应数组为空。

对话：
{json.dumps(messages, ensure_ascii=False, indent=2)}
"""
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
    data = json.loads(content)
    return {
        "facts": data.get("facts", []),
        "events": data.get("events", [])
    }

async def process_user_conversation(user_id: str, thread_id: str, force: bool = False):
    memory = get_memory()
    key_count = f"last_msg_count_{user_id}_{thread_id}"
    last_count_str = memory.get_user_metadata("__system__", key_count)
    last_count = int(last_count_str) if last_count_str else 0
    
    all_msgs = await get_all_messages(thread_id)
    current_count = len(all_msgs)
    if not force and current_count <= last_count:
        return
        
    # 取增量消息（从 last_count 位置开始）
    new_messages = all_msgs[last_count:] if last_count < current_count else []
    if not force and len(new_messages) < MIN_MESSAGES_BEFORE_EXTRACT:
        return  # 增量不足，稍后再处理
    
    # 只取增量（可选：为了 LLM 上下文，取最近 30 条）
    context_msgs = all_msgs[-30:] if len(all_msgs) > 30 else all_msgs
    extracted = await extract_memories_from_conversation(context_msgs)
    new_facts = extracted["facts"]
    new_events = extracted["events"]

    # 处理 facts（去重逻辑保持不变）
    if new_facts:
        existing_facts = set()
        for fact in new_facts:
            similar = memory.query_relevant(fact, user_id, n_results=2)
            for sim in similar:
                if sim["content"].strip() == fact.strip():
                    existing_facts.add(fact)
                    break
        final_facts = [f for f in new_facts if f not in existing_facts]
        if final_facts:
            memory.add_facts_batch(final_facts, user_id, {
                "source": "auto_extracted",
                "thread_id": thread_id,
                "type": "fact"
            })
            print(f"[MEM] Extracted {len(final_facts)} new facts for {user_id}")

    # 处理 events：直接存储（可用相似度去重，但事件通常不重复）
    if new_events:
        event_summaries = [e["summary"] for e in new_events]
        event_outcomes = [e.get("outcome", "neutral") for e in new_events]
        # 批量存入，每条作为独立文档
        for summary, outcome in zip(event_summaries, event_outcomes):
            # 可选：检查是否已存在相似事件（按内容向量相似度>0.9可跳过）
            memory.add_fact(  # 复用 add_fact 方法，但修改 metadata
                content=summary,
                user_id=user_id,
                metadata={
                    "source": "auto_extracted",
                    "thread_id": thread_id,
                    "type": "event",
                    "outcome": outcome,
                    "extracted_at": datetime.now().isoformat()
                }
            )
        print(f"[MEM] Extracted {len(new_events)} new events for {user_id}")
        
    # 更新记录：保存新的消息总数
    memory.set_user_metadata("__system__", f"last_msg_count_{user_id}_{thread_id}", str(current_count))
    memory.set_user_metadata("__system__", f"last_extract_{user_id}_{thread_id}", datetime.now().isoformat())

async def conversation_memory_worker():
    """后台工作器：定期扫描活跃线程并提取记忆"""
    print("[MEM] Starting conversation memory extractor worker...")
    pool = get_pool()
    while True:
        try:
            active_threads = await get_active_threads(pool)
            print(f"[MEM] Found {len(active_threads)} active threads to check")
            for thread_id, user_id in active_threads:
                await process_user_conversation(user_id, thread_id)
            # 等待下一次扫描
            await asyncio.sleep(EXTRACT_INTERVAL_SECONDS)
        except Exception as e:
            print(f"[MEM] Error in conversation memory worker: {e}")
            await asyncio.sleep(EXTRACT_INTERVAL_SECONDS)