"""
记忆模块
"""
import os
os.environ['CHROMA_CACHE_DIR'] = os.path.join(os.path.dirname(__file__), '..', 'chroma_cache')
import chromadb
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LongTermMemory:
    def __init__(self, persist_directory="./chroma_db", markdown_dir="./agent_memory"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {}  # 按 user_id 缓存
        self.markdown_dir = markdown_dir
        os.makedirs(self.markdown_dir, exist_ok=True)

    def _get_collection(self, user_id: str):
        """获取或创建对应用户的记忆集合"""
        if user_id not in self.collections:
            collection_name = f"user_memories_{user_id}"
            self.collections[user_id] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collections[user_id]

    def add_fact(self, content: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加一条事实到指定用户的记忆库"""
        coll = self._get_collection(user_id)
        doc_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
        metadata["timestamp"] = datetime.now().isoformat()
        metadata.setdefault("type", "fact")
        metadata["user_id"] = user_id  # 冗余存储，便于调试
        coll.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        # 只有用户的事实信息才同步到 Markdown，event事件存入向量库，但不存markdown文件
        if metadata.get("type") == "fact":
            self._sync_to_markdown(user_id, content, metadata)
        return doc_id

    def query_relevant(self, query: str, user_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """从指定用户的记忆库中检索相关事实"""
        try:
            coll = self._get_collection(user_id)
            results = coll.query(query_texts=[query], n_results=n_results)
        except Exception as e:
            if "Error creating hnsw segment reader" in str(e):
                logger.warning(f"记忆索引损坏，尝试重建 collection for user {user_id}")
                # 删除并重建 collection
                self.client.delete_collection(f"user_memories_{user_id}")
                coll = self.client.create_collection(f"user_memories_{user_id}")
                # 临时从markdown加载
                return self.query_from_markdown(user_id)
            else:
                raise e
        memories = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                memories.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        return memories

    def _get_metadata_collection(self):
        """获取或创建用于存储用户元数据的 collection"""
        return self.client.get_or_create_collection("user_metadata")

    def set_user_metadata(self, user_id: str, key: str, value: str):
        """存储用户特定的元数据（如最近 thread_id）"""
        coll = self._get_metadata_collection()
        doc_id = f"{user_id}:{key}"
        coll.upsert(
            documents=[value],
            metadatas=[{"user_id": user_id, "key": key}],
            ids=[doc_id]
        )

    def get_user_metadata(self, user_id: str, key: str) -> Optional[str]:
        """获取用户特定的元数据，若不存在返回 None"""
        coll = self._get_metadata_collection()
        doc_id = f"{user_id}:{key}"
        result = coll.get(ids=[doc_id])
        if result['documents'] and result['documents'][0]:
            return result['documents'][0]
        return None
    
    def _sync_to_markdown(self, user_id: str, fact: str, metadata: dict):
        """将事实追加到用户的 Markdown 记忆文件中（仅写入事实本身）"""
        file_path = os.path.join(self.markdown_dir, f"{user_id}.md")
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            time_str = timestamp

        entry = f"\n## {time_str}\n- 事实：{fact}\n"
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(entry)
    
    def query_from_markdown(self, user_id: str) -> List[Dict]:
        file_path = os.path.join(self.markdown_dir, f"{user_id}.md")
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 简单解析（可根据需要优化）
        lines = content.split('\n')
        memories = []
        for line in lines:
            if line.startswith('- 事实：'):
                fact = line[4:].strip()
                memories.append({"content": fact, "metadata": {}})
        return memories
    
    def clear_user_memory(self, user_id: str):
        """删除用户的所有记忆（删除 collection）"""
        try:
            self.client.delete_collection(f"user_memories_{user_id}")
            if user_id in self.collections:
                del self.collections[user_id]
        except Exception:
            logger.debug(f"删除 collection user_memories_{user_id} 失败（可能不存在）", exc_info=True)
    
    def add_facts_batch(self, facts: List[str], user_id: str, metadata: dict = None):
        """批量添加事实，提高效率"""
        coll = self._get_collection(user_id)
        ids = [str(uuid.uuid4()) for _ in facts]
        # 构建基础元数据
        base_meta = metadata.copy() if metadata else {}
        base_meta["timestamp"] = datetime.now().isoformat()
        base_meta.setdefault("type", "fact")
        base_meta["user_id"] = user_id
        # 为每条事实生成独立的元数据副本
        metadatas = [base_meta.copy() for _ in facts]
        coll.add(
            documents=facts,
            metadatas=metadatas,
            ids=ids
        )
        if base_meta.get("type") == "fact":
            self._append_to_markdown(user_id, facts, metadatas)
    
    def _append_to_markdown(self, user_id: str, facts: List[str], metadatas: List[dict]):
        """将批量事实追加到用户的 Markdown 记忆文件中（仅写入事实本身）"""
        file_path = os.path.join(self.markdown_dir, f"{user_id}.md")
        with open(file_path, 'a', encoding='utf-8') as f:
            for fact, meta in zip(facts, metadatas):
                timestamp = meta.get('timestamp', datetime.now().isoformat())
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    time_str = timestamp
                f.write(f"\n## {time_str}\n- 事实：{fact}\n")
    
    def get_user_profile(self, user_id: str) -> str:
        """
        获取用户画像 - 从 Markdown 文件中读取所有事实（仅 facts，不包含 events）。
        返回去重后的用户画像文本，供 load_user_profile 工具使用。
        """
        file_path = os.path.join(self.markdown_dir, f"{user_id}.md")
        if not os.path.exists(file_path):
            return "暂无该用户的画像信息。"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析出所有事实行
        facts = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('- 事实：') or line.startswith('- fact：'):
                fact = line.split('：', 1)[1].strip() if '：' in line else line[4:].strip()
                if fact:
                    facts.append(fact)

        if not facts:
            return "暂无该用户的画像信息。"

        # 去重（保持顺序）
        seen = set()
        unique_facts = []
        for f in facts:
            if f not in seen:
                seen.add(f)
                unique_facts.append(f)

        return "## 用户画像\n\n" + "\n".join(f"- {f}" for f in unique_facts)

    def get_random_facts(self, user_id: str, n: int = 3) -> List[str]:
        """随机获取用户记忆中的 n 条事实"""
        coll = self._get_collection(user_id)
        try:
            # 先获取总记录数（如果有元数据可以计数，但直接获取可能更好）
            # 简单起见，获取前 n 条作为随机（虽然不是真随机，但可行）
            result = coll.get(limit=n)
            return result.get('documents', [])
        except Exception as e:
            logger.warning(f"获取随机记忆失败: {e}")
            return []

# 全局单例
_memory_instance = None

def get_memory():
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = LongTermMemory()
    return _memory_instance