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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from agent.skill_version_manager import create_new_skill_version, update_skill_usage

# 假设你的项目已有这些配置
from config import config
from agent.utils import call_big_model_chat  # 异步大模型调用函数

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
    """初始化向量库（在应用启动时调用一次）"""
    global _chroma_client, _skill_collection
    if not _chroma_client or not _skill_collection:
        _chroma_client = client
        # 技能集合：用于存储反思文本和技能文档
        _skill_collection = _chroma_client.get_or_create_collection(
            name="agent_skills",
            metadata={"hnsw:space": "cosine"}
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
    prompt = f"""
你是一个经验学习智能体。根据以下任务执行过程，分析成败原因，并输出 JSON。

任务描述：{task_data.get('task_description', '无描述')}

步骤轨迹：
{json.dumps(task_data['steps'], indent=2, ensure_ascii=False)}

最终结果：{task_data.get('final_result', 'unknown')}
用户反馈：{task_data.get('user_feedback', '无')}

请输出如下 JSON 格式：
{{
  "outcome": "success" 或 "failure",
  "reflection": "总结成功的关键动作或失败的根本原因（不超过 150 字）",
  "should_create_skill": true/false,   // 建议：成功 && 步骤数>=2 且不重复已有技能
  "skill_name": "建议的技能名称（snake_case）",
  "skill_trigger_phrases": ["触发短语1", "触发短语2"],
  "key_lessons": "可复用的经验教训（一句话）"
}}

只输出 JSON，不要额外解释。
"""
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

    # 向量化：使用技能文档的前两部分作为检索内容
    # 提取名称、描述、触发短语用于检索
    embed_text = f"技能名称：{skill_name}\n触发词：{', '.join(trigger_phrases)}\n描述：{skill_doc[:500]}"
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

    print(f"[SKILL] Created skill: {skill_name}")
    # 使用版本管理创建新版本
    await create_new_skill_version(skill_name, skill_doc, reflection.get("reflection", ""))
    return filename

async def _generate_skill_document(task_description: str, steps: List[Dict],
                                   reflection: str, key_lessons: str,
                                   trigger_phrases: List[str]) -> str:
    """调用 LLM 生成格式化的技能 Markdown 文档"""
    prompt = f"""
你是一个技能文档编写器。根据以下任务执行过程和反思，创建一个可复用的技能文档。

任务描述：{task_description}

执行步骤：
{json.dumps(steps, indent=2, ensure_ascii=False)}

反思总结：{reflection}
经验教训：{key_lessons}
建议触发短语：{', '.join(trigger_phrases)}

请输出 Markdown 格式的技能文档，结构如下：

---
name: <技能名称（英文下划线）>
description: <一句话描述技能作用>
triggers: [<触发短语列表>]
version: 1.0
---

## 技能描述
详细说明技能适用的场景和解决的问题。

## 执行步骤
1. ...
2. ...

## 注意事项
- ...

## 反思与优化
{reflection}
"""
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
        print(f"[REFLECTION] Processed task {task_data['task_id']} -> {reflection.get('outcome')}")
    except Exception as e:
        print(f"[ERROR] Failed to process task file {filepath.name}: {e}")
        import traceback
        traceback.print_exc()

async def reflection_worker():
    """后台任务：轮询 pending_tasks 目录并处理"""
    print("Starting reflection worker...")
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