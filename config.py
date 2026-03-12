"""
配置文件
"""
import os
from dotenv import load_dotenv
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

load_dotenv()
# 记忆文件夹
MEMORY_MARKDOWN_DIR = os.getenv("MEMORY_MARKDOWN_DIR", "./agent_memory")
# pg数据库
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://root:pwd@localhost:5442/postgres?sslmode=disable")
POSTGRES_URI_ASYNC = os.getenv("POSTGRES_URI", "postgresql://root:pwd@localhost:5442/postgres?sslmode=disable")
POSTGRES_URI_SYNC = POSTGRES_URI.replace("postgresql://", "postgresql+psycopg://")


class DatabaseConfig(BaseSettings):
    """数据库相关配置"""
    postgres_uri: str = Field(..., alias="POSTGRES_URI", description="PostgreSQL连接字符串")
    postgres_pool_min_size: int = Field(1, alias="POSTGRES_POOL_MIN_SIZE")
    postgres_pool_max_size: int = Field(10, alias="POSTGRES_POOL_MAX_SIZE")
    postgres_pool_timeout: int = Field(30, alias="POSTGRES_POOL_TIMEOUT")
    chroma_persist_dir: str = Field("./chroma_db", alias="CHROMA_PERSIST_DIR")
    memory_markdown_dir: str = Field("./agent_memory", alias="MEMORY_MARKDOWN_DIR")
    postgres_uri_sync: Optional[str] = Field(None, alias="POSTGRES_URI_SYNC", description="PostgreSQL连接")
    
    @model_validator(mode="after")
    def set_sync_uri(self) -> "DatabaseConfig":
        # 如果没有提供同步 URI，则从异步 URI 生成
        if self.postgres_uri_sync is None:
            # 将 postgresql:// 替换为 postgresql+psycopg2://
            self.postgres_uri_sync = self.postgres_uri.replace("postgresql://", "postgresql+psycopg://")
        return self


class HubConfig(BaseSettings):
    """Hub通信配置"""
    hub_host: str = Field("localhost", alias="HUB_HOST")
    hub_port: int = Field(8765, alias="HUB_PORT")

    @property
    def hub_url(self) -> str:
        return f"ws://{self.hub_host}:{self.hub_port}"

class ModelConfig(BaseSettings):
    """模型相关配置"""
    zhipu_api_key: Optional[str] = Field(None, alias="ZHIPU_API_KEY")
    default_provider: Optional[str] = Field(None, alias="DEFAULT_PROVIDER")
    default_model: str = Field("GLM-4.7", alias="DEFAULT_MODEL")
    vision_model: str = Field("glm-4.6v", alias="VISION_MODEL")
    intent_model: str = Field("glm-4-flash", alias="INTENT_MODEL")
    model_temperature: float = Field(0.0, alias="MODEL_TEMPERATURE")
    # 可以扩展其他模型提供商的配置

class AgentConfig(BaseSettings):
    """Agent实例配置"""
    agent_id_prefix: str = Field("agent", alias="AGENT_ID_PREFIX")
    num_agents: int = Field(1, alias="NUM_AGENTS", description="启动的Agent数量")
    use_long_term_memory: bool = Field(True, alias="USE_LONG_TERM_MEMORY")

class BackendConfig(BaseSettings):
    """deepagents后端配置"""
    backend_root_dir: str = Field(os.getcwd(), alias="BACKEND_ROOT_DIR")  # 相对于项目根目录
    backend_virtual_mode: bool = Field(True, alias="BACKEND_VIRTUAL_MODE")
    backend_timeout: int = Field(30, alias="BACKEND_TIMEOUT")
    backend_max_output_bytes: int = Field(10000, alias="BACKEND_MAX_OUTPUT_BYTES")

class SchedulerConfig(BaseSettings):
    """调度器配置"""
    scheduler_timezone: str = Field("Asia/Shanghai", alias="SCHEDULER_TIMEZONE")
    reminder_check_interval: int = Field(10, alias="REMINDER_CHECK_INTERVAL")  # 秒

class AppConfig(BaseSettings):
    """总配置，组合所有子配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # 数据库
    db: DatabaseConfig = DatabaseConfig()

    # Hub
    hub: HubConfig = HubConfig()

    # 模型
    model: ModelConfig = ModelConfig()

    # Agent
    agent: AgentConfig = AgentConfig()

    # Backend
    backend: BackendConfig = BackendConfig()

    # Scheduler
    scheduler: SchedulerConfig = SchedulerConfig()

# 创建全局配置实例，供整个应用使用
config = AppConfig()