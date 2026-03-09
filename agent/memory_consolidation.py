"""
记忆提炼总结
"""
import os
import json
import asyncio
from datetime import datetime
from agent.memory import get_memory
from agent.utils import call_zhipu_chat  # 注意：使用异步版本

async def extract_facts_from_markdown(file_path: str) -> list:
    """从 Markdown 文件中提取所有事实内容（异步版，但这里只是读取文件，可以保持同步，但为了统一，可以用同步）"""
    facts = []
    if not os.path.exists(file_path):
        return facts
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('- 事实：'):
                fact = line[5:].strip()
                facts.append(fact)
    return facts

def write_facts_to_markdown(file_path: str, facts: list):
    """将事实列表写入 Markdown 文件（同步）"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for fact in facts:
            f.write(f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 事实：{fact}\n")
            f.write("- source：consolidated\n")
            f.write("- type：fact\n\n")

async def deduplicate_facts_with_llm(facts: list) -> list:
    """使用大模型对事实列表进行语义去重和合并（异步）"""
    if not facts:
        return facts

    prompt = f"""你是一个智能的记忆整理助手。我将给你一系列用户提供的事实，这些事实可能重复、相似或互相包含。请你去除重复，合并相似的事实，返回一个简洁、无冗余的事实列表。

            要求：
            - 完全相同的文本只保留一个。
            - 语义相似的事实，例如“我喜欢吃苹果”和“我喜欢苹果”，可以合并成更通用的表述，或者保留其中一个。
            - 如果事实之间存在包含关系，保留更完整的那条。
            - 输出格式：一个 JSON 数组，每个元素是一条事实字符串。
            
            事实列表：
            {json.dumps(facts, ensure_ascii=False, indent=2)}
            
            只输出 JSON，不要任何额外文字。"""
    try:
        response = await call_zhipu_chat(prompt, model="GLM-4.7", temperature=0.0)
        content = response["choices"][0]["message"]["content"]
        print(content)
        # 清理可能的 Markdown 代码块
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        new_facts = json.loads(content)
        if isinstance(new_facts, list) and all(isinstance(f, str) for f in new_facts):
            return new_facts
        else:
            print("LLM返回格式错误，使用简单去重")
            return list(set(facts))
    except Exception as e:
        print(f" LLM去重失败: {e}，使用简单去重")
        return list(set(facts))

async def consolidate_user_memory(user_id: str):
    """异步整理单个用户的记忆"""
    memory = get_memory()
    markdown_path = os.path.join(memory.markdown_dir, f"{user_id}.md")

    facts = await extract_facts_from_markdown(markdown_path)  # 注意 await
    if not facts:
        return

    unique_facts = await deduplicate_facts_with_llm(facts)  # await 异步去重

    memory.clear_user_memory(user_id)

    if unique_facts:
        memory.add_facts_batch(unique_facts, user_id, {"source": "consolidated"})

    write_facts_to_markdown(markdown_path, unique_facts)  # 同步写入

    print(f" 用户 {user_id} 记忆整理完成：{len(facts)} -> {len(unique_facts)} 条")