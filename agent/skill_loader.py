# agent/skill_loader.py
import os
import asyncio
import yaml
import importlib.util
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain.tools import tool


@dataclass
class SkillMetadata:
    """技能元数据（从SKILL.md的YAML头加载）"""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "unknown"
    tags: List[str] = None


class Skill:
    """代表一个完整的技能（元数据 + 内容 + 脚本）"""
    
    def __init__(self, skill_dir: Path):
        self.skill_dir = skill_dir
        self.metadata = self._load_metadata()
        self.full_content = self._load_full_content()
        self.scripts = self._discover_scripts()
    
    def _load_metadata(self) -> SkillMetadata:
        """只加载SKILL.md的YAML头（元数据层）"""
        skill_path = self.skill_dir / "SKILL.md"
        with open(skill_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析YAML frontmatter（在 --- 之间）
        if content.startswith('---'):
            _, yaml_part, _ = content.split('---', 2)
            data = yaml.safe_load(yaml_part)
            return SkillMetadata(
                name=data.get('name', self.skill_dir.name),
                description=data.get('description', ''),
                version=data.get('version', '1.0.0'),
                author=data.get('author', 'unknown'),
                tags=data.get('tags', [])
            )
        else:
            # 兼容没有YAML头的旧格式
            return SkillMetadata(
                name=self.skill_dir.name,
                description="No description"
            )
    
    def _load_full_content(self) -> str:
        """加载完整的SKILL.md内容（技能主体层）"""
        skill_path = self.skill_dir / "SKILL.md"
        with open(skill_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _discover_scripts(self) -> Dict[str, Path]:
        """发现scripts目录下的所有可执行脚本"""
        scripts_dir = self.skill_dir / "scripts"
        if not scripts_dir.exists():
            return {}
        
        scripts = {}
        for script_file in scripts_dir.glob("*.py"):
            scripts[script_file.stem] = script_file
        return scripts
    
    async def execute_script(self, script_name: str, **kwargs):
        """执行指定的脚本（资源层）"""
        if script_name not in self.scripts:
            raise ValueError(f"Script {script_name} not found")
        
        script_path = self.scripts[script_name]
        
        # 动态导入并执行脚本
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # if hasattr(module, 'execute'):
        #     return await module.execute(**kwargs)
        # else:
        #     raise ValueError(f"Script {script_name} has no execute() function")
        # 定义要尝试的入口函数名称列表（按优先级排序）
        entry_points = ['execute', 'main', 'run', 'handler', script_name, 'call']
        
        for func_name in entry_points:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if callable(func):
                    # 如果函数是异步的，直接 await；否则在线程池中运行
                    if asyncio.iscoroutinefunction(func):
                        return await func(**kwargs)
                    else:
                        # 同步函数放到线程池中执行，避免阻塞事件循环
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, lambda: func(**kwargs))
        
        # 如果没有找到任何入口函数，抛出明确的错误
        raise ValueError(
            f"Script {script_name} has no known entry function. "
            f"Tried: {', '.join(entry_points)}. "
            f"Please ensure the script defines one of these functions, "
            f"or specify a custom entry point in SKILL.md using 'entry_point: function_name'."
        )
    
    def get_instructions(self) -> str:
        """返回技能的完整指令（例如 SKILL.md 的全部内容）"""
        return self.full_content  # 假设你已经在 Skill 中存储了 full_content
        
    def get_tools(self):
        tools = []
        for script_name, script_path in self.scripts.items():
            # 使用 @tool 装饰器，并通过 description 参数提供描述
            @tool(description=f"执行 {self.metadata.name} 技能的 {script_name} 操作")
            async def execute_skill(path: str) -> str:
                result = await self.execute_script(script_name, path=path)
                return result.get("result", str(result))
            
            # 设置工具名称
            execute_skill.name = self.metadata.name  # 例如 'read_file'
            tools.append(execute_skill)
        return tools
        

  
class SkillRegistry:
    """技能注册中心，负责发现和管理所有技能"""
    
    def __init__(self, skills_dir: str = "./skills"):
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Skill] = {}
        self._loaded = False
        
    async def load(self):
        """异步加载所有技能（将同步扫描放入线程池）"""
        if self._loaded:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._discover_skills_sync)
        self._loaded = True
        
    def _discover_skills_sync(self):
        """同步扫描技能目录（在子线程中执行）"""
        if not self.skills_dir.exists():
            print(f"⚠️ 技能目录不存在: {self.skills_dir}")
            return
        for skill_folder in self.skills_dir.iterdir():
            if skill_folder.is_dir() and (skill_folder / "SKILL.md").exists():
                skill = Skill(skill_folder)  # Skill 初始化同步
                self.skills[skill.metadata.name] = skill
                print(f"✅ 发现技能: {skill.metadata.name} - {skill.metadata.description}")

    def get_skill_metadata(self) -> List[Dict[str, Any]]:
        """获取所有技能的元数据（用于加载到系统提示）"""
        return [
            {
                "name": skill.metadata.name,
                "description": skill.metadata.description,
                "tags": skill.metadata.tags
            }
            for skill in self.skills.values()
        ]
    
    def find_relevant_skills(self, query: str, top_k: int = 3) -> List[Skill]:
        """根据查询找到最相关的技能（简单关键词匹配，后续可升级为向量检索）"""
        # 简单实现：在描述中搜索关键词
        relevant = []
        query_lower = query.lower()
        for skill in self.skills.values():
            if any(tag.lower() in query_lower for tag in skill.metadata.tags or []):
                relevant.append(skill)
            elif query_lower in skill.metadata.description.lower():
                relevant.append(skill)
        
        return relevant[:top_k]
    
    def get_skill(self, name: str) -> Skill:
        """根据名称获取技能"""
        return self.skills.get(name)
        
    def list_skills(self):
        return list(self.skills.values())
        
    