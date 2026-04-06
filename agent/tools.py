"""
自定义工具
"""
import json
import uuid
from pathlib import Path
from datetime import datetime
from langchain.tools import tool

BASE_DIR = Path(__file__).parent.absolute()   # memory_processor.py 所在的目录（agent目录）
PENDING_DIR = BASE_DIR / "data" / "pending"
MEMORIES_DIR = BASE_DIR / "data" / "memories"
INDEX_PATH = MEMORIES_DIR / "index.json"


# 辅助函数：确保目录存在
def ensure_dirs():
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

@tool
def log_memory(step_description: str, tool_name: str, input_args: str, output: str, error: str = ""):
    """
    记录一个关键步骤的原始日志，供异步记忆处理器生成摘要。
    应该在每个重要操作（工具调用、错误、用户反馈）后调用。
    """
    ensure_dirs()
    log_entry = {
        "step_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "step_description": step_description,
        "tool_name": tool_name,
        "input": input_args,
        "output": output,
        "error": error,
        "success": error == ""
    }
    filename = f"{log_entry['step_id']}.json"
    filepath = PENDING_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
    return f"Logged memory step {log_entry['step_id']}"


@tool
def retrieve_memory(query: str, max_summaries: int = 10) -> str:
    """
    根据任务描述检索相关记忆的摘要列表（二级渐进的第一、二级）。
    返回摘要文本，Agent 可根据需要再调用 load_full_memory 加载完整内容。
    排序依据：置信度 × 时间衰减（越近使用权重越高）
    """
    ensure_dirs()
    if not INDEX_PATH.exists():
        return "No memory index found. No past experiences available."
    
    with open(INDEX_PATH, 'r') as f:
        index = json.load(f)
    
    # 第一级：从索引中选出最相关的标签（简单文本匹配）
    query_lower = query.lower()
    candidate_tags = []
    for tag_info in index.get("top_tags", []):
        tag = tag_info["tag"]
        if tag.lower() in query_lower or query_lower in tag.lower():
            candidate_tags.append(tag)
    if not candidate_tags:
        candidate_tags = [t["tag"] for t in index.get("top_tags", [])[:3]]
    
    # 辅助函数：解析 frontmatter 中的字段
    def parse_frontmatter(content: str) -> dict:
        """从 Markdown 文件内容中提取 frontmatter 字段"""
        result = {"confidence": 0.5, "last_used": None, "step_summary": "", "task_summary": ""}
        lines = content.split("\n")
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                if not in_frontmatter:
                    in_frontmatter = True
                else:
                    break
                continue
            if in_frontmatter and ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key == "confidence":
                    try:
                        result["confidence"] = float(val)
                    except:
                        pass
                elif key == "last_used":
                    result["last_used"] = val
                elif key == "step_summary":
                    result["step_summary"] = val
                elif key == "task_summary":
                    result["task_summary"] = val
        return result
    
    def calculate_score(confidence: float, last_used_str: str, now=None) -> float:
        """计算综合得分：置信度 × 时间衰减因子（每天衰减5%）"""
        if now is None:
            now = datetime.now()
        if not last_used_str:
            return confidence  # 无时间信息，仅用置信度
        try:
            last_used = datetime.fromisoformat(last_used_str)
            days_diff = (now - last_used).days
            decay = 0.95 ** days_diff  # 每天衰减5%
            return confidence * decay
        except:
            return confidence
    
    # 第二级：按标签搜索摘要文件，收集候选记忆
    memories = []  # 每个元素为 (score, file_name, step_summary, task_summary, confidence)
    for tag in candidate_tags[:3]:  # 最多3个标签
        for md_file in MEMORIES_DIR.glob("*.md"):
            # 简单检查文件中是否包含该标签（避免重复添加同一个文件）
            content = md_file.read_text(encoding='utf-8')
            if f"tags: {tag}" not in content and f"tags:[\"{tag}\"]" not in content:
                continue
            # 解析 frontmatter
            meta = parse_frontmatter(content)
            # 计算得分
            score = calculate_score(meta["confidence"], meta.get("last_used"))
            # 去重：同一个文件只保留最高分（理论上只会出现一次）
            existing = next((m for m in memories if m[1] == md_file.name), None)
            if existing:
                if score > existing[0]:
                    existing[0] = score
            else:
                memories.append([
                    score,
                    md_file.name,
                    meta["step_summary"],
                    meta["task_summary"],
                    meta["confidence"]
                ])
    
    # 按得分降序排序，取前 max_summaries
    memories.sort(key=lambda x: x[0], reverse=True)
    memories = memories[:max_summaries]
    
    if not memories:
        return "No relevant memories found."
    
    # 构建返回的摘要文本
    result = "Found these relevant memories (sorted by relevance, higher score = more useful):\n\n"
    for idx, (score, fname, step_summary, task_summary, conf) in enumerate(memories):
        result += f"[{idx + 1}] File: {fname}\n"
        result += f"    Step summary: {step_summary}\n"
        if task_summary:
            result += f"    Task summary: {task_summary}\n"
        result += f"    Confidence: {conf:.2f}, Relevance score: {score:.3f}\n\n"
    result += "To view full details of a memory, use load_full_memory with the file name."
    return result


@tool
def load_full_memory(filename: str) -> str:
    """
    加载完整记忆内容（三级渐进的第三级）。
    filename 应该是 retrieve_memory 返回结果中的文件名。
    """
    filepath = MEMORIES_DIR / filename
    if not filepath.exists():
        return f"Memory file {filename} not found."
    content = filepath.read_text(encoding='utf-8')
    # 限制长度，避免超出上下文
    if len(content) > 8000:
        content = content[:8000] + "\n... (truncated)"
    return content


@tool
def update_memory_confidence(filename: str, success: bool):
    """
    在根据记忆成功解决问题后调用，增加置信度；失败则降低。
    """
    filepath = MEMORIES_DIR / filename
    if not filepath.exists():
        return f"Memory file {filename} not found."
    
    content = filepath.read_text(encoding='utf-8')
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("confidence:"):
            old_conf = float(line.split(":", 1)[1].strip())
            if success:
                new_conf = min(1.0, old_conf + 0.1)
            else:
                new_conf = max(0.0, old_conf - 0.1)
            new_lines.append(f"confidence: {new_conf:.2f}")
        else:
            new_lines.append(line)
    # 更新 last_used
    for i, line in enumerate(new_lines):
        if line.startswith("last_used:"):
            new_lines[i] = f"last_used: {datetime.now().isoformat()}"
            break
    filepath.write_text("\n".join(new_lines), encoding='utf-8')
    return f"Updated confidence for {filename} to {new_conf:.2f}"