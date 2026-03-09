"""
配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

HUB_HOST = os.getenv("HUB_HOST", "localhost")
HUB_PORT = int(os.getenv("HUB_PORT", 8765))
AGENT_ID = os.getenv("AGENT_ID", "agent_1")  # 每个实例应有唯一 ID

# LLM 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# 记忆文件夹
MEMORY_MARKDOWN_DIR = os.getenv("MEMORY_MARKDOWN_DIR", "./agent_memory")

# pg数据库
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable")
POSTGRES_URI_ASYNC = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable")

POSTGRES_URI_SYNC = POSTGRES_URI.replace("postgresql://", "postgresql+psycopg://")