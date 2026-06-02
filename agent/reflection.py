"""
反思与技能提炼模块
功能：
- 接收一个任务的完整步骤和结果
- 调用 LLM 分析成败原因，生成反思总结
- 如果成功且步骤数 >= 2，进一步生成可复用的技能文档
- 将反思/技能存入向量库和 Markdown 文件
"""
import asyncio
import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from agent.skill_version_manager import create_new_skill_version, update_skill_usage

logger = logging.getLogger(__name__)

# 假设你的项目已有这些配置
from config import config
from agent.utils import call_big_model_chat  # 异步大模型调用函数
from agent.prompts import build_task_reflection_prompt, build_skill_document_prompt

# ---------- 目录配置 ----------
BASE_DIR = Path(__file__).parent.absolute()
PENDING_TASKS_DIR = BASE_DIR / "data" / "pending_tasks"
SKILLS_DIR = BASE_DIR / "data" / "skills"             # 自我进化skill
REFLECTIONS_DIR = BASE_DIR / "data" / "reflections"   # 纯反思记录

# 确保目录存在
PENDING_TASKS_DIR.mkdir(parents=True, exist_ok=True)
SKILLS_DIR.mkdir(parents=True, exist_ok=True)
REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 向量库集成 ----------
# 注意：这里需要你提供实际的 Chroma 客户端实例或 get_chroma_collection 函数
# 下面是一个占位符，你需要根据你的实际代码修改
_chroma_client = None
_skill_collection = None

def init_chroma(client):
    """初始化向量库（在应用启动时调用一次），并将内置技能索引到向量库。"""
    global _chroma_client, _skill_collection
    if not _chroma_client or not _skill_collection:
        _chroma_client = client
        # 技能集合：用于存储反思文本和技能文档
        _skill_collection = _chroma_client.get_or_create_collection(
            name="agent_skills",
            metadata={"hnsw:space": "cosine"}
        )
        _index_builtin_skills()


def _index_builtin_skills():
    """将内置技能（agent/skills/）索引到 ChromaDB，使 search_skills 能搜到所有技能。"""
    import json as _json
    builtin_dir = Path(__file__).parent / "skills"
    if not builtin_dir.exists():
        return
    for skill_dir in builtin_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        content = skill_md.read_text(encoding='utf-8')
        if not content.startswith("---"):
            continue
        parts = content.split("---", 2)
        if len(parts) < 2:
            continue
        frontmatter = parts[1]
        name = ""
        description = ""
        triggers = []
        lines = frontmatter.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip()
            elif line.startswith("triggers:"):
                triggers_str = line.split(":", 1)[1].strip()
                if triggers_str:
                    try:
                        triggers = _json.loads(triggers_str)
                    except (_json.JSONDecodeError, TypeError):
                        triggers = [triggers_str.strip("[]'\"")]
                else:
                    i += 1
                    while i < len(lines):
                        item_line = lines[i].strip()
                        if item_line.startswith("- "):
                            triggers.append(item_line[2:].strip().strip("'\""))
                            i += 1
                        elif item_line == "" or item_line.startswith(("#", "version:", "license:")):
                            break
                        else:
                            break
                    continue
            i += 1

        if not name:
            continue

        doc_id = f"builtin_{name}"
        # 将技能正文的关键部分也纳入向量化，提升检索相关性
        body = parts[2].strip() if len(parts) > 2 else ""
        body_snippet = body[:800] if body else ""
        embed_text = f"技能名称：{name}\n触发词：{', '.join(triggers)}\n描述：{description}\n内容概要：{body_snippet}"
        _skill_collection.upsert(
            documents=[embed_text],
            metadatas=[{
                "type": "skill",
                "skill_name": name,
                "source": "builtin",
                "trigger_phrases": _json.dumps(triggers),
            }],
            ids=[doc_id]
        )

def get_skill_collection():
    return _skill_collection

# ---------- 反思生成 ----------
async def reflect_on_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    对已完成的任务进行反思，返回反思结果。
    task_data 格式:
    {
        "task_id": "thread_id 或唯一标识",
        "task_description": "用户原始请求",
        "steps": [{"step_description": "...", "result": "...", "tool_calls": [...]}, ...],
        "final_result": "success/failure 或 最终输出",
        "user_feedback": "可选的用户反馈"
    }
    """
    prompt = build_task_reflection_prompt(task_data)
    response = await call_big_model_chat(prompt, model=config.model.default_model,
                                         temperature=0.3, is_json=True)
    content = response["choices"][0]["message"]["content"]
    # 清理 Markdown 代码块
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    result = json.loads(content)
    return result

# ---------- 技能提炼 ----------
async def create_skill_from_reflection(task_data: Dict, reflection: Dict) -> Optional[str]:
    """
    根据反思结果生成技能文档（Markdown），存入 SKILLS_DIR，同时向量化。
    返回技能文件名（或 None）。
    """
    if not reflection.get("should_create_skill", False):
        return None

    skill_name = reflection.get("skill_name", f"skill_{uuid.uuid4().hex[:8]}")
    trigger_phrases = reflection.get("skill_trigger_phrases", [])
    key_lessons = reflection.get("key_lessons", "")
    reflection_text = reflection.get("reflection", "")

    # 调用 LLM 生成完整的技能文档（三级内容）
    skill_doc = await _generate_skill_document(
        task_description=task_data.get("task_description", ""),
        steps=task_data["steps"],
        reflection=reflection_text,
        key_lessons=key_lessons,
        trigger_phrases=trigger_phrases
    )

    # 保存 Markdown 文件
    filename = f"{skill_name}.md"
    filepath = SKILLS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(skill_doc)

    # 向量化：纳入技能名称、触发词、关键教训和文档概要，提升检索相关性
    embed_text = f"技能名称：{skill_name}\n触发词：{', '.join(trigger_phrases)}\n关键教训：{key_lessons}\n内容概要：{skill_doc[:600]}"
    collection = get_skill_collection()
    if collection:
        doc_id = f"skill_{skill_name}"
        collection.upsert(
            documents=[embed_text],
            metadatas=[{
                "type": "skill",
                "skill_name": skill_name,
                "trigger_phrases": json.dumps(trigger_phrases),
                "created_at": datetime.now().isoformat()
            }],
            ids=[doc_id]
        )

    logger.info(f"Created skill: {skill_name}")
    # 使用版本管理创建新版本
    await create_new_skill_version(skill_name, skill_doc, reflection.get("reflection", ""))
    return filename

async def _generate_skill_document(task_description: str, steps: List[Dict],
                                   reflection: str, key_lessons: str,
                                   trigger_phrases: List[str]) -> str:
    """调用 LLM 生成格式化的技能 Markdown 文档"""
    prompt = build_skill_document_prompt(
        task_description, steps, reflection, key_lessons, trigger_phrases
    )
    response = await call_big_model_chat(prompt, model=config.model.default_model,
                                         temperature=0.4, is_json=False)
    return response["choices"][0]["message"]["content"]

# ---------- 保存纯反思记录 ----------
def save_reflection(task_id: str, reflection_data: Dict):
    """将反思结果保存到 reflections 目录（JSON）"""
    filepath = REFLECTIONS_DIR / f"{task_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(reflection_data, f, indent=2, ensure_ascii=False)

# ---------- 异步处理任务（类似你现有的 memory_processor） ----------
async def process_pending_task(filepath: Path):
    """处理一个 pending 任务文件：反思 -> 提炼技能 -> 入库"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        # 1. 反思
        reflection = await reflect_on_task(task_data)

        # 2. 保存反思记录
        save_reflection(task_data["task_id"], reflection)

        # 3. 如果成功且建议创建技能，则生成技能
        if reflection.get("outcome") == "success":
            await create_skill_from_reflection(task_data, reflection)

        # 删除 pending 文件
        filepath.unlink()
        logger.info(f"Processed task {task_data['task_id']} -> {reflection.get('outcome')}")
    except Exception as e:
        logger.error(f"Failed to process task file {filepath.name}: {e}", exc_info=True)

async def reflection_worker():
    """后台任务：轮询 pending_tasks 目录并处理"""
    logger.info("Starting reflection worker...")
    while True:
        for filepath in PENDING_TASKS_DIR.glob("*.json"):
            await process_pending_task(filepath)
        await asyncio.sleep(5)  # 每5秒检查一次

# ---------- 外部调用接口 ----------
def submit_task_for_reflection(task_data: Dict):
    """
    提交一个已完成的任务到待处理队列（由你的 Agent 在任务结束时调用）
    task_data 需包含 task_id, task_description, steps, final_result 等
    """
    task_id = task_data.get("task_id")
    if not task_id:
        task_id = f"task_{uuid.uuid4().hex}"
        task_data["task_id"] = task_id
    filepath = PENDING_TASKS_DIR / f"{task_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)