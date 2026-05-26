"""
技能检索工具（统一入口）
- list_skills：列出所有技能（内置 + 自我进化）
- load_skill：按需加载技能详情（优先自我进化版本）
- search_skills：向量检索所有技能
"""
import json
from pathlib import Path
from typing import List, Dict
from agent.skill_version_manager import (get_skill_latest_version, get_skill_file_path, create_new_skill_version,
                                   update_skill_usage, load_manifest, compute_skill_value)
from langchain.tools import tool

from agent.reflection import SKILLS_DIR, get_skill_collection

# 内置（预定义）技能目录 — agent/skills/，子目录 + SKILL.md 格式
PREDEFINED_SKILLS_DIR = Path(__file__).parent / "skills"


def _parse_frontmatter(content: str) -> dict:
    """从 Markdown 内容提取 frontmatter 中的 name, description, triggers。
    支持两种 triggers 格式：
      - JSON 数组: triggers: ["词1", "词2"]
      - YAML 列表: triggers:\n  - 词1\n  - 词2
    """
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 2:
        return {}
    frontmatter = parts[1]
    info = {"name": "", "description": "", "triggers": []}
    lines = frontmatter.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("name:"):
            info["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("description:"):
            info["description"] = line.split(":", 1)[1].strip()
        elif line.startswith("triggers:"):
            triggers_str = line.split(":", 1)[1].strip()
            if triggers_str:
                # 单行 JSON 格式: triggers: ["a", "b"]
                try:
                    info["triggers"] = json.loads(triggers_str)
                except (json.JSONDecodeError, TypeError):
                    info["triggers"] = [triggers_str.strip("[]'\"")]
            else:
                # YAML 多行格式: 后续行以 "  - " 开头
                i += 1
                while i < len(lines):
                    item_line = lines[i].strip()
                    if item_line.startswith("- "):
                        info["triggers"].append(item_line[2:].strip().strip("'\""))
                        i += 1
                    elif item_line == "" or item_line.startswith(("#", "version:", "license:")):
                        break  # 空行或其他字段，结束收集
                    else:
                        break
                continue  # 已经推进了 i，跳过末尾的 i+=1
        i += 1
    return info


def _scan_builtin_skills() -> list:
    """扫描内置技能目录（agent/skills/），子目录+SKILL.md 格式"""
    skills = []
    if not PREDEFINED_SKILLS_DIR.exists():
        return skills
    for skill_dir in sorted(PREDEFINED_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        info = _parse_frontmatter(skill_md.read_text(encoding='utf-8'))
        if info.get("name"):
            info["source"] = "builtin"
            skills.append(info)
    return skills


def _scan_learned_skills() -> list:
    """扫描自我进化技能目录（agent/data/skills/），平铺 .md 文件"""
    skills = []
    for md_file in sorted(SKILLS_DIR.glob("*.md")):
        info = _parse_frontmatter(md_file.read_text(encoding='utf-8'))
        if info.get("name"):
            info["source"] = "learned"
            skills.append(info)
    return skills


def _format_content(content: str, detail_level: str) -> str:
    """按 detail_level 截取技能内容"""
    if detail_level == "basic":
        info = _parse_frontmatter(content)
        name = info.get("name", "未知")
        desc = info.get("description", "")
        return f"技能 {name}: {desc}" if desc else f"技能 {name}"
    elif detail_level == "intermediate":
        return content[:600] + ("..." if len(content) > 600 else "")
    else:  # full
        return content


@tool
async def list_skills() -> str:
    """
    列出所有可用技能（内置 + 自我进化），仅返回技能名称和触发短语。
    用于快速浏览可用技能，不加载详细步骤。
    """
    lines = []

    builtin = _scan_builtin_skills()
    if builtin:
        lines.append("【内置技能】")
        for s in builtin:
            triggers = ', '.join(s.get('triggers', []))
            lines.append(f"  {s['name']}（触发: {triggers}）")

    learned = _scan_learned_skills()
    if learned:
        lines.append("【学习技能】")
        for s in learned:
            triggers = ', '.join(s.get('triggers', []))
            lines.append(f"  {s['name']}（触发: {triggers}）")

    if not lines:
        return "当前没有可用技能。"
    return "\n".join(lines)


@tool
async def load_skill(skill_name: str, version: str = "latest", detail_level: str = "basic") -> str:
    """
    加载指定技能的详细信息。优先返回自我进化版本，其次内置版本。
    version 可以是 'latest' 或具体版本号如 '2'
    detail_level 可选：
        - basic: 仅名称和描述（约 50 tokens）
        - intermediate: 描述 + 触发条件 + 注意事项（约 200 tokens）
        - full: 完整步骤 + 示例代码（约 1000+ tokens）
    """
    # 1. 优先查找自我进化技能（agent/data/skills/）
    if version == "latest":
        ver = get_skill_latest_version(skill_name)
    else:
        ver = int(version)

    if ver:
        filepath = get_skill_file_path(skill_name, ver)
        if filepath.exists():
            update_skill_usage(skill_name, success=True)
            content = filepath.read_text(encoding='utf-8')
            return _format_content(content, detail_level)

    # 2. 回退到内置技能（agent/skills/{name}/SKILL.md）
    builtin_path = PREDEFINED_SKILLS_DIR / skill_name / "SKILL.md"
    if builtin_path.exists():
        content = builtin_path.read_text(encoding='utf-8')
        return _format_content(content, detail_level)

    return f"技能 {skill_name} 未找到。请用 list_skills 查看可用技能列表。"


@tool
async def search_skills(query: str, n: int = 3) -> str:
    """
    根据自然语言查询检索最相关的技能（向量检索 + 触发词匹配，覆盖所有技能）。
    """
    collection = get_skill_collection()
    results = []

    # 1. ChromaDB 向量检索（内置 + 自我进化都已索引）
    if collection:
        try:
            chroma_results = collection.query(query_texts=[query], n_results=n)
            if chroma_results.get('documents') and chroma_results['documents'][0]:
                for doc, meta in zip(chroma_results['documents'][0], chroma_results['metadatas'][0]):
                    skill_name = meta.get('skill_name', '未知')
                    source = meta.get('source', 'learned')
                    source_label = "内置" if source == "builtin" else "学习"
                    results.append(f"[{source_label}] {skill_name}: {doc[:200]}...")
        except Exception:
            pass

    # 2. 触发词回退匹配（向量搜索结果不足时补充）
    if len(results) < n:
        query_lower = query.lower()
        for s in _scan_builtin_skills() + _scan_learned_skills():
            if len(results) >= n:
                break
            if any(s['name'] in r for r in results):
                continue
            triggers = s.get('triggers', [])
            if any(t.lower() in query_lower or query_lower in t.lower() for t in triggers):
                source_label = "内置" if s.get('source') == "builtin" else "学习"
                desc = s.get('description', '')[:200]
                results.append(f"[{source_label}] {s['name']}: {desc}")

    if not results:
        return "未找到相关技能。请用 list_skills 查看完整技能列表。"
    return "相关技能：\n" + "\n".join(results)


@tool
async def skill_stats(skill_name: str) -> str:
    """查看技能的使用统计和版本历史"""
    manifest = load_manifest()
    if skill_name not in manifest["skills"]:
        # 检查是否为内置技能
        builtin_path = PREDEFINED_SKILLS_DIR / skill_name / "SKILL.md"
        if builtin_path.exists():
            return f"技能 {skill_name}（内置）\n暂无使用统计（内置技能不跟踪统计）。"
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
