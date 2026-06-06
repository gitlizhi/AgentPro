"""
技能版本管理与遗忘巩固
功能：
- 技能版本升级/回滚
- 使用统计更新
- 价值评分与自动归档
"""

import json
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
import math

logger = logging.getLogger(__name__)
from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).parent.absolute()
SKILLS_DIR = BASE_DIR / "data" / "skills"
SKILLS_ARCHIVE_DIR = BASE_DIR / "data" / "skills_archive"
MANIFEST_PATH = SKILLS_DIR.parent / "skills_manifest.json"

SKILLS_DIR.mkdir(parents=True, exist_ok=True)
SKILLS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# 获取 chroma collection（复制 reflection.py 中的实现）
_chroma_client = None
_skill_collection = None

def get_skill_collection():
    global _chroma_client, _skill_collection
    if _skill_collection is None:
        chroma_path = BASE_DIR / "chroma_db"
        _chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        _skill_collection = _chroma_client.get_or_create_collection(
            name="agent_skills",
            metadata={"hnsw:space": "cosine"}
        )
    return _skill_collection

def load_manifest() -> Dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return {"skills": {}}

def save_manifest(manifest: Dict):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

def get_skill_latest_version(skill_name: str) -> Optional[int]:
    manifest = load_manifest()
    skill_info = manifest["skills"].get(skill_name)
    if skill_info:
        return skill_info["latest_version"]
    return None

def get_skill_file_path(skill_name: str, version: int) -> Path:
    return SKILLS_DIR / f"{skill_name}_v{version}.md"

def update_skill_usage(skill_name: str, success: bool):
    """更新技能的使用统计（每次调用后调用）"""
    manifest = load_manifest()
    if skill_name not in manifest["skills"]:
        return
    now = datetime.now().isoformat()
    skill_data = manifest["skills"][skill_name]
    skill_data["last_used"] = now
    if success:
        skill_data["success_count"] = skill_data.get("success_count", 0) + 1
    else:
        skill_data["fail_count"] = skill_data.get("fail_count", 0) + 1
    skill_data["total_uses"] = skill_data.get("total_uses", 0) + 1
    save_manifest(manifest)

async def create_new_skill_version(skill_name: str, content: str, changelog: str) -> int:
    """
    创建新版本技能。
    返回新版本号。
    """
    manifest = load_manifest()
    if skill_name in manifest["skills"]:
        old_version = manifest["skills"][skill_name]["latest_version"]
        new_version = old_version + 1
        # 将旧版本文件保留（不删除）
    else:
        old_version = None
        new_version = 1

    # 解析 frontmatter 获取描述和触发词
    # 简单提取名称和触发词（可调用 LLM 生成，这里简化为从 content 开头提取）
    import re
    name_match = re.search(r'name:\s*(.+)', content)
    desc_match = re.search(r'description:\s*(.+)', content)
    triggers_match = re.search(r'triggers:\s*\[(.*?)\]', content)
    skill_desc = desc_match.group(1) if desc_match else ""
    triggers = []
    if triggers_match:
        triggers = [t.strip().strip('"\'') for t in triggers_match.group(1).split(',')]

    # 保存新版本文件
    filepath = get_skill_file_path(skill_name, new_version)
    filepath.write_text(content, encoding='utf-8')

    # 更新 manifest
    now = datetime.now().isoformat()
    manifest["skills"][skill_name] = {
        "latest_version": new_version,
        "versions": manifest.get(skill_name, {}).get("versions", []) + [new_version],
        "description": skill_desc,
        "triggers": triggers,
        "created_at": now,
        "last_used": now,
        "success_count": 0,
        "fail_count": 0,
        "total_uses": 0,
        "changelog": changelog
    }
    # 保留版本历史
    if "version_history" not in manifest["skills"][skill_name]:
        manifest["skills"][skill_name]["version_history"] = []
    manifest["skills"][skill_name]["version_history"].append({
        "version": new_version,
        "created_at": now,
        "changelog": changelog
    })
    save_manifest(manifest)

    # 更新向量库（用新版本覆盖旧版本嵌入）
    collection = get_skill_collection()
    if collection:
        embed_text = f"技能名称：{skill_name}\n触发词：{', '.join(triggers)}\n描述：{skill_desc}"
        doc_id = f"skill_{skill_name}"
        collection.upsert(
            documents=[embed_text],
            metadatas=[{
                "type": "skill",
                "skill_name": skill_name,
                "version": new_version,
                "trigger_phrases": json.dumps(triggers),
                "updated_at": now
            }],
            ids=[doc_id]
        )
    return new_version

async def rollback_skill(skill_name: str, target_version: int) -> bool:
    """回滚到指定版本"""
    manifest = load_manifest()
    if skill_name not in manifest["skills"]:
        return False
    current = manifest["skills"][skill_name]["latest_version"]
    if target_version >= current:
        return False
    # 检查版本文件是否存在
    target_path = get_skill_file_path(skill_name, target_version)
    if not target_path.exists():
        return False
    # 读取旧版本内容，作为新版本创建（版本号 = current+1，内容为旧版本）
    content = target_path.read_text(encoding='utf-8')
    changelog = f"Rollback from v{current} to v{target_version}"
    await create_new_skill_version(skill_name, content, changelog)
    return True

def archive_old_skill(skill_name: str, version: int, reason: str = "low_value"):
    """将技能移至归档目录"""
    src = get_skill_file_path(skill_name, version)
    if not src.exists():
        return
    dst = SKILLS_ARCHIVE_DIR / f"{skill_name}_v{version}_archived_{datetime.now().strftime('%Y%m%d')}.md"
    shutil.move(str(src), str(dst))
    # 从 manifest 中移除该版本（但保留记录在 archived_versions 中）
    manifest = load_manifest()
    if skill_name in manifest["skills"]:
        if version in manifest["skills"][skill_name].get("versions", []):
            manifest["skills"][skill_name]["versions"].remove(version)
        # 如果是最新版本被归档，则更新 latest_version 为之前的版本
        if version == manifest["skills"][skill_name]["latest_version"]:
            remaining = [v for v in manifest["skills"][skill_name]["versions"] if v != version]
            if remaining:
                manifest["skills"][skill_name]["latest_version"] = max(remaining)
            else:
                del manifest["skills"][skill_name]
        save_manifest(manifest)

def compute_skill_value(skill_data: Dict) -> float:
    """计算技能价值分数（基于使用次数、成功率、时间衰减）"""
    total_uses = skill_data.get("total_uses", 0)
    if total_uses == 0:
        return 0.0
    success_count = skill_data.get("success_count", 0)
    success_rate = success_count / total_uses
    # 使用次数对数奖励（避免高频技能无限增长）
    usage_bonus = math.log(total_uses + 1) * 0.2
    # 时间衰减：最后使用距今天数，超过30天开始衰减
    last_used = datetime.fromisoformat(skill_data.get("last_used", datetime.now().isoformat()))
    days_since = (datetime.now() - last_used).days
    time_decay = math.exp(-days_since / 30.0)  # 30天半衰期
    # 综合分数
    value = (success_rate * 0.6 + usage_bonus * 0.2) * time_decay
    return round(value, 4)

async def consolidate_skills_job():
    """后台任务：定期评估技能价值，低价值技能归档"""
    manifest = load_manifest()
    for skill_name, data in manifest["skills"].items():
        value = compute_skill_value(data)
        if value < 0.1 and data["total_uses"] > 3:  # 低价值且使用过
            archive_old_skill(skill_name, data["latest_version"], reason="low_value")
            # 从向量库移除该技能
            collection = get_skill_collection()
            if collection:
                doc_id = f"skill_{skill_name}"
                try:
                    collection.delete([doc_id])
                except Exception:
                    logger.debug(f"删除 skill {skill_name} 向量文档失败", exc_info=True)
            logger.info(f"Archived low-value skill: {skill_name} (value={value})")
        else:
            # 高价值技能可增加检索权重（通过更新 metadata 中的 boost）
            collection = get_skill_collection()
            if collection:
                doc_id = f"skill_{skill_name}"
                # 更新 metadata 中的 value 字段用于检索时排序（需要查询时手动排序，或者使用 chroma 的 metadata 过滤）
                # 这里仅记录，实际检索时可以根据 value 重新排序
                pass
    # 每24小时运行一次
    await asyncio.sleep(86400)