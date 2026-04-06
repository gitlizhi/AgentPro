"""
记忆（过往经验） 处理模块
"""
import asyncio
import json
import re
import glob
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from config import config
from agent.utils import call_zhipu_chat
# 配置
BASE_DIR = Path(__file__).parent.absolute()   # memory_processor.py 所在的目录（agent目录）
PENDING_DIR = BASE_DIR / "data" / "pending"
MEMORIES_DIR = BASE_DIR / "data" / "memories"
INDEX_PATH = MEMORIES_DIR / "index.json"
POLL_INTERVAL = 2  # 秒


def safe_filename(s: str) -> str:
    """替换 Windows 文件名中的非法字符为下划线"""
    # Windows 非法字符: \ / : * ? " < > |
    return re.sub(r'[\\/*?:"<>|]', '_', s)

def ensure_dirs():
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)


def load_index() -> Dict:
    if INDEX_PATH.exists():
        with open(INDEX_PATH, 'r') as f:
            return json.load(f)
    return {
        "last_updated": datetime.now().isoformat(),
        "top_tags": [],
        "recent_tasks": [],
        "tag_cooccurrence": {}
    }


def save_index(index: Dict):
    index["last_updated"] = datetime.now().isoformat()
    with open(INDEX_PATH, 'w') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


async def call_llm_to_summarize(raw_log: Dict) -> Dict:
    """
    调用 LLM 从原始日志中提取：
    - task_id (如果新任务则生成新ID)
    - step_summary (步骤级摘要)
    - task_summary (如果该步骤标志任务结束则生成)
    - tags (标签列表)
    - confidence (初始置信度，默认0.5)
    """
    prompt = f"""
    你是一个记忆摘要生成器。请从以下智能体操作日志中提取关键信息，输出 JSON 格式。
    
    原始日志：
    {json.dumps(raw_log, indent=2, ensure_ascii=False)}
    
    输出格式：
    {{
      "task_id": "任务ID（如果是新任务，使用 'task_'+时间戳；如果可关联现有任务，使用已有ID）",
      "step_summary": "该步骤的简短摘要（不超过100字），包含做了什么、结果如何",
      "task_summary": "如果该步骤完成了整个任务，提供任务级摘要；否则为 null",
      "tags": ["标签1", "标签2"],
      "confidence": 0.5
    }}
    
    要求：
    - tags 由你根据内容来总结，可以是一个或者多个
    - 如果日志包含错误，tags 中必须包含 'error' 或具体错误类型
    - step_summary 要突出操作和结果，比如“执行了数据库迁移，成功”或“尝试连接Redis，超时失败”
    
    请直接输出 JSON 对象，不要使用 Markdown 代码块，不要添加任何额外解释。
    """
    content = await call_zhipu_chat(prompt, model=config.model.default_model,
                                    temperature=0.3)
    # 提取模型输出的文本内容
    content_str = content["choices"][0]["message"]["content"]
    
    # 清理 Markdown 代码块标记
    cleaned = content_str.strip()
    # 匹配 ```json ... ``` 或 ``` ... ```
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = cleaned  # 如果没有代码块，直接使用原始字符串

    # 尝试解析 JSON
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON from LLM: {e}\nRaw output: {content_str}")
        # 返回一个默认结构
        result = {}
    
    # 确保必需字段存在（兜底）
    defaults = {
        "task_id": raw_log.get("step_id", f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        "step_summary": raw_log.get("step_description", "No summary")[:100],
        "task_summary": "",
        "tags": ["unclassified"],
        "confidence": 0.5
    }
    for key, default_value in defaults.items():
        if key not in result or result[key] is None:
            result[key] = default_value
            
    # 确保 step_summary 不是 None
    if result["step_summary"] is None:
        result["step_summary"] = "No summary"
    if result["task_summary"] is None:
        result["task_summary"] = ""
        
    # 特别处理 tags 必须是列表
    if not isinstance(result.get("tags"), list):
        result["tags"] = ["unclassified"]
    
    return result


def update_index_with_memory(index: Dict, memory_meta: Dict):
    """更新索引：标签计数、最近任务、共现矩阵"""
    # 获取摘要，确保不是 None
    step_summary = memory_meta.get("step_summary") or ""
    task_summary = memory_meta.get("task_summary") or ""
    task_title = (task_summary if task_summary else step_summary)[:50]
    
    # 更新 top_tags
    tags = memory_meta.get("tags", [])
    for tag in tags:
        found = False
        for t in index["top_tags"]:
            if t["tag"] == tag:
                t["count"] += 1
                t["avg_confidence"] = (t["avg_confidence"] * (t["count"] - 1) + memory_meta["confidence"]) / t["count"]
                found = True
                break
        if not found:
            index["top_tags"].append({"tag": tag, "count": 1, "avg_confidence": memory_meta["confidence"]})
    
    # 更新 recent_tasks
    task_id = memory_meta["task_id"]
    # 避免重复添加同一任务
    existing = [t for t in index["recent_tasks"] if t["task_id"] == task_id]
    if not existing:
        index["recent_tasks"].insert(0, {
            "task_id": task_id,
            "title": task_title,
            "timestamp": datetime.now().isoformat()
        })
    # 保持最近20条
    index["recent_tasks"] = index["recent_tasks"][:20]
    
    # 更新 tag_cooccurrence
    if len(tags) > 1:
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1:]:
                if t1 not in index["tag_cooccurrence"]:
                    index["tag_cooccurrence"][t1] = []
                if t2 not in index["tag_cooccurrence"][t1]:
                    index["tag_cooccurrence"][t1].append(t2)
                if t2 not in index["tag_cooccurrence"]:
                    index["tag_cooccurrence"][t2] = []
                if t1 not in index["tag_cooccurrence"][t2]:
                    index["tag_cooccurrence"][t2].append(t1)
    
    # 按 count 排序 top_tags
    index["top_tags"].sort(key=lambda x: x["count"], reverse=True)
    index["top_tags"] = index["top_tags"][:20]


def save_memory_markdown(memory_meta: Dict, raw_log: Dict):
    """将记忆保存为 MD 文件"""
    task_id = safe_filename(memory_meta["task_id"])
    step_id = safe_filename(raw_log.get("step_id", datetime.now().strftime("%Y%m%d_%H%M%S")))
    filename = f"{task_id}_{step_id}.md"
    filepath = MEMORIES_DIR / filename
    
    # Frontmatter (YAML 风格)
    frontmatter = f"""---
    task_id: {task_id}
    step_summary: {memory_meta["step_summary"]}
    task_summary: {memory_meta.get("task_summary", "")}
    tags: {json.dumps(memory_meta["tags"])}
    confidence: {memory_meta["confidence"]}
    last_used: {datetime.now().isoformat()}
    created_at: {datetime.now().isoformat()}
    ---
    """
    # 内容
    content = f"""# {memory_meta["step_summary"]}

    ## 原始日志
    ```json
    {json.dumps(raw_log, indent=2, ensure_ascii=False)}
    {", ".join(memory_meta["tags"])}
    ```
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(frontmatter + "\n" + content)
    
    return filepath.name


async def process_pending_file(filepath: Path):
    """处理单个 pending 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_log = json.load(f)
        # 调用 LLM 生成摘要
        memory_meta = await call_llm_to_summarize(raw_log)
        # 清理 task_id 中的非法字符
        memory_meta["task_id"] = safe_filename(memory_meta["task_id"])
        # 确保 step_id 也安全
        raw_log["step_id"] = safe_filename(raw_log.get("step_id", datetime.now().strftime("%Y%m%d_%H%M%S")))
        # 保存为 MD 文件
        memory_filename = save_memory_markdown(memory_meta, raw_log)
        # 更新索引
        index = load_index()
        update_index_with_memory(index, memory_meta)
        save_index(index)
            
        print(f"[OK] Processed {filepath.name} -> {memory_filename}")
        # 删除pending文件
        filepath.unlink()
    except Exception as e:
        print(f"[ERROR] Failed to process {filepath.name}: {e}")
        traceback.print_exc()  # 打印完整调用栈


async def memory_task():
    ensure_dirs()
    print(f"Starting memory processor, watching {PENDING_DIR} ...")
    while True:
        pending_files = glob.glob(str(PENDING_DIR / "*.json"))
        for pf in pending_files:
            await process_pending_file(Path(pf))
        await asyncio.sleep(POLL_INTERVAL)
