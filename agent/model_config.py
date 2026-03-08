"""
模型配置类
"""
import os
from typing import Optional, Dict, Any, Union
from enum import Enum
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
import config


class ModelProvider(str, Enum):
	"""支持的模型提供商枚举"""
	OPENAI = "openai"
	ANTHROPIC = "anthropic"
	DEEPSEEK = "deepseek"
	GOOGLE = "google_vertexai"  # Gemini
	MISTRAL = "mistralai"
	COHERE = "cohere"
	GROQ = "groq"
	OLLAMA = "ollama"
	PERPLEXITY = "perplexity"
	XAI = "xai"  # Grok
	OPENAI_COMPATIBLE = "openai_compatible"  # 其他兼容OpenAI格式的服务

# 默认的 API 基础地址配置
DEFAULT_BASE_URLS = {
	ModelProvider.OPENAI: "https://api.openai.com/v1",
	ModelProvider.DEEPSEEK: "https://api.deepseek.com/v1",
	ModelProvider.ANTHROPIC: "https://api.anthropic.com/v1",
	ModelProvider.GOOGLE: "https://generativelanguage.googleapis.com",
	ModelProvider.OPENAI_COMPATIBLE: "https://open.bigmodel.cn/api/paas/v4",  # 需要用户指定
}

class ModelConfig:
	"""模型配置管理类"""
	
	def __init__(self):
		self._model_instances: Dict[str, BaseChatModel] = {}
	
	def create_model(
			self,
			provider: Union[ModelProvider, str],
			model_name: str,
			temperature: float = 0,
			max_tokens: Optional[int] = None,
			base_url: Optional[str] = None,
			api_key: Optional[str] = None,
			**kwargs
	) -> BaseChatModel:
		"""
		创建指定提供商的模型实例

		参数：
			provider: 模型提供商
			model_name: 模型名称
			temperature: 温度参数
			max_tokens: 最大输出token数
			base_url: 自定义API地址（可选）
			api_key: API密钥（可选，默认从环境变量读取）
			**kwargs: 其他模型参数
		"""
		
		# 处理 provider 为字符串的情况
		if isinstance(provider, str):
			provider = ModelProvider(provider)
		
		# 获取 API key
		api_key = api_key or self._get_api_key(provider)
		
		# 根据提供商类型创建模型
		if provider == ModelProvider.OPENAI_COMPATIBLE:
			# 使用 ChatOpenAI 接入兼容格式的服务
			if not base_url:
				raise ValueError("使用 OpenAI 兼容格式时必须提供 base_url")
			
			return ChatOpenAI(
				model=model_name,
				api_key=api_key,
				base_url=base_url,
				temperature=temperature,
				max_tokens=max_tokens,
				**kwargs
			)
		else:
			# 使用官方的 init_chat_model
			# 构造模型标识符（格式：provider:model_name）
			model_identifier = f"{provider.value}:{model_name}"
			
			# 准备初始化参数
			init_kwargs = {
				"model": model_identifier,
				"temperature": temperature,
				"api_key": api_key,
				**kwargs
			}
			
			if max_tokens:
				init_kwargs["max_tokens"] = max_tokens
			
			if base_url:
				init_kwargs["base_url"] = base_url
			
			return init_chat_model(**init_kwargs)
	
	def _get_api_key(self, provider: ModelProvider) -> Optional[str]:
		"""从环境变量获取对应提供商的 API key"""
		env_var_map = {
			ModelProvider.OPENAI: "OPENAI_API_KEY",
			ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
			ModelProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
			ModelProvider.GOOGLE: "GOOGLE_API_KEY",
			ModelProvider.MISTRAL: "MISTRAL_API_KEY",
			ModelProvider.COHERE: "COHERE_API_KEY",
			ModelProvider.GROQ: "GROQ_API_KEY",
			ModelProvider.OLLAMA: None,  # Ollama 本地无需 key
			ModelProvider.PERPLEXITY: "PERPLEXITY_API_KEY",
			ModelProvider.XAI: "XAI_API_KEY",
			ModelProvider.OPENAI_COMPATIBLE: "OPENAI_COMPATIBLE_API_KEY",
		}
		
		env_var = env_var_map.get(provider)
		if env_var:
			return os.getenv(env_var)
		return None
	
	def get_model(self, config_key: str = "default") -> BaseChatModel:
		"""
		获取缓存的模型实例，避免重复创建
		支持通过配置键切换不同模型
		"""
		if config_key not in self._model_instances:
			# 从配置加载模型参数
			model_config = self._load_model_config(config_key)
			self._model_instances[config_key] = self.create_model(**model_config)
		
		return self._model_instances[config_key]
	
	def _load_model_config(self, config_key: str) -> Dict[str, Any]:
		"""从配置文件加载模型参数"""
		# 这里可以从 YAML/JSON 文件或环境变量读取
		# 示例：硬编码一些常用配置
		configs = {
			"default": {
				"provider": ModelProvider.OPENAI,
				"model_name": "gpt-4o-mini",
				"temperature": 0,
			},
			"deepseek": {
				"provider": ModelProvider.DEEPSEEK,
				"model_name": "deepseek-chat",
				"temperature": 0.1,
			},
			"claude": {
				"provider": ModelProvider.ANTHROPIC,
				"model_name": "claude-3-sonnet",
				"temperature": 0,
			},
			"gemini": {
				"provider": ModelProvider.GOOGLE,
				"model_name": "gemini-1.5-pro",
				"temperature": 0,
			},
			"tongyi": {  # 通义千问（通过 OpenAI 兼容格式）
				"provider": ModelProvider.OPENAI_COMPATIBLE,
				"model_name": "qwen-plus",
				"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
				"temperature": 0,
			},
			"zhipu": {  # 智谱 AI
				"provider": ModelProvider.OPENAI_COMPATIBLE,
				"model_name": "GLM-4.7",
				"base_url": "https://open.bigmodel.cn/api/paas/v4/",
				"temperature": 0,
			},
			"ollama": {  # 本地模型
				"provider": ModelProvider.OLLAMA,
				"model_name": "llama3.1",
				"base_url": "http://localhost:11434",
				"temperature": 0,
			}
		}
		
		return configs.get(config_key, configs["default"])


# 创建全局配置实例
model_config = ModelConfig()

# 使用示例
if __name__ == "__main__":
	# 测试不同模型
	import asyncio
	async def test_models():
		# # 使用默认 OpenAI 模型
		# default_model = model_config.get_model("default")
		#
		# # 使用 DeepSeek
		# deepseek_model = model_config.get_model("deepseek")
		#
		# # 使用通义千问
		# tongyi_model = model_config.get_model("tongyi")
		
		# 直接创建临时模型
		custom_model = model_config.create_model(
			provider=ModelProvider.OPENAI_COMPATIBLE,
			base_url=DEFAULT_BASE_URLS.get(ModelProvider.OPENAI_COMPATIBLE, None),
			model_name="GLM-4.7-Flash",
			temperature=0.5,
			max_tokens=1000
		)
		result = await custom_model.ainvoke("你是谁？")
		print(result)
		return result
	
	asyncio.run(test_models())

