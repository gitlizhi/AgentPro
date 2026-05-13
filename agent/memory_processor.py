"""
记忆（过往经验） 处理模块
"""

"""
核心架构：三级渐进式记忆检索
设计了一个分层记忆系统，共三层，按需加载：

第一级	索引文件 (index.json)   提供全局统计（高频标签、最近任务），帮助决定查哪些标签
第二级	摘要列表（每个记忆的 step_summary + task_summary）  快速浏览相关经验的概要，决定哪一条值得细看
第三级	完整记忆（Markdown 文件）  获取详细步骤、原始日志、错误信息等


标签只存在于第一级和第二级之间——用来索引和过滤摘要，而不是直接用于语义匹配。
摘要才是用来判断相关性的主要文本。Agent 读摘要后，再决定是否加载完整内容。

标签用于快速过滤（粗粒度）。
摘要用于语义判断（细粒度）。

1.标签（Tags）
作用：粗粒度过滤器，用于快速缩小候选记忆范围。

2.摘要（Summaries）
作用：细粒度的相关性判断。Agent 阅读摘要文本（自然语言），判断是否与当前任务相关。

3.完整记忆（Full Memory）
作用：提供可执行的步骤、错误栈、配置参数等具体细节，供 Agent 模仿或调整。
存储格式：Markdown 文件，包含 frontmatter（元数据）和原始日志。


检索流程（以“爬取百度图片”为例）
1.用户提问：“帮我爬取百度图片，关键词‘刘亦菲’。”

2.Agent 调用 retrieve_memory(query)：

 - 读取 index.json，发现高频标签有 web_crawler, download, delete 等。
 - 从 query 中提取可能的标签：web_crawler（因为“爬取”）、download（因为“图片”）。
 - 扫描所有记忆文件，找到包含这些标签的文件（最多 10 个）。
 - 提取每个文件的 step_summary，返回给 Agent。

3.Agent 阅读摘要：

 - 看到“创建百度图片爬虫脚本，成功下载3张图片到桌面” → 高度相关。
 - 看到“删除桌面图片文件” → 不相关。

4.Agent 调用 load_full_memory 加载相关记忆的完整 Markdown 文件。

5.Agent 基于记忆中的步骤，调整后执行新任务。

标签在这里的作用：只是第一轮筛选器，避免扫描所有记忆。真正的相关性判断完全依赖摘要文本。



记忆转经验思维导图

用户请求
   │
   ▼
Agent.run() ──调用──► retrieve_memory(query)
                           │
                           ├─ 1. 读取 index.json（高频标签、最近任务）
                           │
                           ├─ 2. 从 query 中提取候选标签（例如通过关键词映射）
                           │
                           ├─ 3. 在 /memories/*.md 中 grep 包含这些标签的文件
                           │
                           ├─ 4. 对每个文件解析 frontmatter，取出 step_summary 和 task_summary
                           │
                           ├─ 5. 按（置信度 × 时间衰减）排序，返回前 N 条摘要
                           │
                           └─ 6. Agent 阅读摘要，决定是否调用 load_full_memory()
                                    │
                                    ▼
                              执行任务
                                    │
                                    ▼
                        关键步骤后调用 log_memory()
                                    │
                                    ▼
                        原始日志写入 /pending/*.json
                                    │
                                    ▼
              异步 memory_processor 处理：
                    - 调用 LLM 生成摘要、标签、置信度
                    - 保存为 /memories/*.md
                    - 更新 index.json

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
from agent.utils import call_big_model_chat
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


async def call_llm_to_summarize(raw_log: Dict, retry: int = 0) -> Dict:
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
    - 如果日志包含错误，tags 中必须包含 'error' 或具体错误类型
    - step_summary 要突出操作和结果，比如“执行了数据库迁移，成功”或“尝试连接Redis，超时失败”
    - tags 由你根据内容来总结，可以是一个或者多个
        标签生成规则（非常重要）：
        1. 每个标签是一个简短的关键词短语，使用下划线连接单词，例如：create_crawler_script、delete_image_files、connect_database。
        2. 标签应该体现“动作 + 对象”或“领域 + 动作”，长度不超过30字符。
        3. 禁止包含结果词（如 success、failed、error）、具体的人名/地名/文件名、时间信息。
        4. 最多生成3个标签，如果任务单一可以只给1个。
        5. 不要使用过于宽泛的词如 "task"、"operation"，尽量具体。
        
        示例：
        - 日志：使用 requests 和 beautifulsoup 爬取百度图片，保存到桌面。 → 标签：["web_crawler", "download", "beautifulsoup"]
        - 日志：删除桌面上的 3 张图片文件。 → 标签：["file_management", "delete"]
        - 日志：配置 Flask 应用的数据库连接池。 → 标签：["deployment", "configure", "flask"]
    
    请直接输出 JSON 对象，不要使用 Markdown 代码块，不要添加任何额外解释。
    """
    content = await call_big_model_chat(prompt, model=config.model.default_model,
                                    temperature=0.3, is_json=True)
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
        # 出错后重试
        if retry <= 2:
            retry += 1
            result = await call_llm_to_summarize(raw_log, retry)
        # 返回一个默认结构
        else:
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
