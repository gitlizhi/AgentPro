"""
对话记忆自动提取器（事件驱动版）
触发时机：
  1. 任务完成并反思后（agent 调用 log_memory(task_complete=True) 后）
  2. 对话因长时间无活动而被终止时（Brain.close() 内）
不再使用轮询，避免空转消耗 token。
"""
import json
import logging
from datetime import datetime
from typing import List, Dict

from agent.utils import call_big_model_chat
from agent.prompts import build_memory_extraction_prompt
from agent.memory import get_memory
from agent.db import get_pool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from config import config

logger = logging.getLogger(__name__)

MIN_MESSAGES_BEFORE_EXTRACT = 20  # 至少积累 20 条消息才触发提取


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
    使用小模型从对话中提取 facts 和 events。
    返回: {"facts": [...], "events": [{"summary": "...", "outcome": "..."}]}
    """
    if not messages:
        return {"facts": [], "events": []}
    if len(messages) > 30:
        messages = messages[-30:]

    prompt = build_memory_extraction_prompt(messages)
    response = await call_big_model_chat(
        prompt,
        model=config.model.memory_extraction_model,
        temperature=0.2,
        is_json=True,
    )
    content = response["choices"][0]["message"]["content"]
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


async def trigger_memory_extraction(thread_id: str, user_id: str = "super_user",
                                     agent_id: str = "unknown") -> bool:
    """供外部调用的入口：对指定 thread 做一次记忆提取。
    通过 ChromaDB 记录上次提取的消息数，实现增量提取。

    Returns:
        True 如果实际执行了提取，False 如果增量不足被跳过。
    """
    memory = get_memory()
    key_count = f"last_msg_count_{user_id}_{thread_id}"
    last_count_str = memory.get_user_metadata("__system__", key_count)
    last_count = int(last_count_str) if last_count_str else 0

    all_msgs = await get_all_messages(thread_id)
    current_count = len(all_msgs)
    new_messages = all_msgs[last_count:] if last_count < current_count else []

    if len(new_messages) < MIN_MESSAGES_BEFORE_EXTRACT:
        logger.debug(
            f"Skip extraction for {thread_id}: {len(new_messages)} new msgs < {MIN_MESSAGES_BEFORE_EXTRACT}"
        )
        return False

    context_msgs = all_msgs[-30:] if len(all_msgs) > 30 else all_msgs
    extracted = await extract_memories_from_conversation(context_msgs)
    new_facts = extracted["facts"]
    new_events = extracted["events"]

    # --- 处理 facts（语义去重后写入） ---
    SIMILARITY_THRESHOLD = 0.15
    final_facts = []
    for fact in new_facts:
        similar = memory.query_relevant(fact, user_id, n_results=3)
        is_dup = False
        for sim in similar:
            if (sim["content"].strip() == fact.strip()
                    or (sim.get("distance") is not None and sim["distance"] < SIMILARITY_THRESHOLD)):
                is_dup = True
                break
        if not is_dup:
            final_facts.append(fact)

    if final_facts:
        memory.add_facts_batch(final_facts, user_id, {
            "source": "auto_extracted",
            "thread_id": thread_id,
            "agent_id": agent_id,
            "type": "fact",
        })
        logger.info(
            f"Memory extraction [{agent_id}]: {len(final_facts)} facts "
            f"({len(new_facts) - len(final_facts)} dupes filtered)"
        )

    # --- 处理 events（按 agent 写入） ---
    if new_events:
        for ev in new_events:
            memory.add_fact(
                content=ev["summary"],
                user_id=user_id,
                metadata={
                    "source": "auto_extracted",
                    "thread_id": thread_id,
                    "agent_id": agent_id,
                    "type": "event",
                    "outcome": ev.get("outcome", "neutral"),
                    "extracted_at": datetime.now().isoformat(),
                }
            )
        logger.info(f"Memory extraction [{agent_id}]: {len(new_events)} events")

    # 更新进度
    memory.set_user_metadata("__system__", key_count, str(current_count))
    memory.set_user_metadata(
        "__system__", f"last_extract_{user_id}_{thread_id}", datetime.now().isoformat()
    )
    return True
