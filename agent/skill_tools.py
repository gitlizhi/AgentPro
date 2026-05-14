"""
技能检索工具
- list_skills：列出所有可用技能（名称+触发词）
- load_skill：按需加载技能详情（三级渐进）
"""
import json
from pathlib import Path
from typing import List, Dict
from agent.skill_version_manager import (get_skill_latest_version, get_skill_file_path, create_new_skill_version,
                                   update_skill_usage, load_manifest, compute_skill_value)
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
async def load_skill(skill_name: str, version: str = "latest", detail_level: str = "basic") -> str:
    """
    加载指定技能的详细信息。
    version 可以是 'latest' 或具体版本号如 '2'
    detail_level 可选：
        - basic: 仅名称和描述（约 50 tokens）
        - intermediate: 描述 + 触发条件 + 注意事项（约 200 tokens）
        - full: 完整步骤 + 示例代码（约 1000+ tokens）
    """
    if version == "latest":
        ver = get_skill_latest_version(skill_name)
    else:
        ver = int(version)
    if not ver:
        return f"技能 {skill_name} 未找到"
    filepath = get_skill_file_path(skill_name, ver)
    if not filepath.exists():
        return f"技能 {skill_name} 版本 {ver} 不存在"
        
    # 更新使用统计（假设加载成功）
    update_skill_usage(skill_name, success=True)  # 这里简单认为加载即成功，更精确可在工具调用后由 Agent 反馈
    content = filepath.read_text(encoding='utf-8')

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


@tool
async def skill_stats(skill_name: str) -> str:
    """查看技能的使用统计和版本历史"""
    manifest = load_manifest()
    if skill_name not in manifest["skills"]:
        return f"未找到技能 {skill_name}"
    data = manifest["skills"][skill_name]
    return f"""
        技能: {skill_name}
        最新版本: v{data['latest_version']}
        总使用次数: {data.get('total_uses', 0)}
        成功次数: {data.get('success_count', 0)}
        失败次数: {data.get('fail_count', 0)}
        最后使用: {data.get('last_used', '从未')}
        价值分数: {compute_skill_value(data)}
        版本历史: {data.get('version_history', [])}
        """

@tool
async def upgrade_skill(skill_name: str, new_content: str, changelog: str) -> str:
    """手动创建新版本技能（由 Agent 或管理员调用）"""
    new_ver = await create_new_skill_version(skill_name, new_content, changelog)
    return f"技能 {skill_name} 已升级到 v{new_ver}"

@tool
async def report_skill_result(skill_name: str, success: bool) -> str:
    """报告技能执行结果，用于更新统计"""
    update_skill_usage(skill_name, success)
    return "已记录"