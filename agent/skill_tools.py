"""
技能检索工具
- list_skills：列出所有可用技能（名称+触发词）
- load_skill：按需加载技能详情（三级渐进）
"""
import json
from pathlib import Path
from typing import List, Dict

from langchain.tools import tool

# 假设你的 reflection 模块中已有技能目录和向量集合
from agent.reflection import SKILLS_DIR, get_skill_collection

@tool
async def list_skills() -> str:
    """
    列出所有已沉淀的技能（仅返回技能名称和触发短语，不加载详细步骤）。
    用于快速浏览可用技能。
    """
    skills = []
    for md_file in SKILLS_DIR.glob("*.md"):
        # 简单解析 frontmatter
        content = md_file.read_text(encoding="utf-8")
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 2:
                frontmatter = parts[1]
                # 提取 name 和 triggers
                name = ""
                triggers = []
                for line in frontmatter.splitlines():
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip()
                    if line.startswith("triggers:"):
                        triggers_str = line.split(":", 1)[1].strip()
                        try:
                            triggers = json.loads(triggers_str)
                        except:
                            triggers = [triggers_str]
                if name:
                    skills.append(f"• {name} (触发: {', '.join(triggers)})")
    if not skills:
        return "当前没有可用技能。"
    return "可用的技能列表：\n" + "\n".join(skills)

@tool
async def load_skill(skill_name: str, detail_level: str = "basic") -> str:
    """
    加载指定技能的详细信息。detail_level 可选：
    - basic: 仅名称和描述（约 50 tokens）
    - intermediate: 描述 + 触发条件 + 注意事项（约 200 tokens）
    - full: 完整步骤 + 示例代码（约 1000+ tokens）
    """
    skill_file = SKILLS_DIR / f"{skill_name}.md"
    if not skill_file.exists():
        # 尝试模糊匹配
        candidates = list(SKILLS_DIR.glob(f"*{skill_name}*.md"))
        if not candidates:
            return f"未找到技能 '{skill_name}'。"
        skill_file = candidates[0]

    content = skill_file.read_text(encoding="utf-8")

    if detail_level == "basic":
        # 提取 name 和 description
        lines = content.splitlines()
        desc = ""
        name = ""
        for i, line in enumerate(lines):
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            if line.startswith("description:"):
                desc = line.split(":", 1)[1].strip()
                break
        return f"技能 {name}: {desc}"
    elif detail_level == "intermediate":
        # 返回 description + 注意事项 + 触发词
        # 简单截取前 600 字符
        return content[:600] + ("..." if len(content) > 600 else "")
    else:
        return content

# 可选：使用向量检索直接搜索相关技能
@tool
async def search_skills(query: str, n: int = 3) -> str:
    """
    根据自然语言查询检索最相关的技能（利用向量库）。
    """
    collection = get_skill_collection()
    if not collection:
        return "技能检索功能未初始化。"
    results = collection.query(query_texts=[query], n_results=n)
    if not results['documents'] or not results['documents'][0]:
        return "未找到相关技能。"
    output = "相关技能：\n"
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        skill_name = meta.get('skill_name', '未知')
        output += f"- {skill_name}: {doc[:200]}...\n"
    return output

