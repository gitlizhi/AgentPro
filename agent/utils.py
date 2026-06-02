"""
其他工具
"""
import json
from dotenv import load_dotenv
load_dotenv()
import os
import httpx
from typing import List, Dict, Any, Optional
from config import config

async def call_big_model_chat(
    user_input: str,
    model: str = "deepseek-chat",
    temperature: float = 1.0,
    stream: bool = False,
    max_tokens: Optional[int] = None,
    is_json: bool = False,
) -> Dict[str, Any]:
    """
    调用大模型的API的接口。

    :param user_input: 用户输入字符串，会自动包装为 messages 格式
    :param model: 模型名称，如 "deepseek-chat", "deepseek-reasoner" 等
    :param temperature: 温度参数，控制随机性
    :param stream: 是否使用流式输出（当前不支持，请使用 False）
    :param max_tokens: 最大生成token数
    :param is_json: 是否启用 JSON 模式输出
    :return: API 返回的 JSON 数据（字典）
    """
    api_key = config.model.api_key
    if not api_key:
        raise ValueError("环境变量 api_key 未设置")

    # url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [{"role": "system", "content": "你是一个聪明的人工智能助手"}, {"role": "user", "content": user_input}]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
        "response_format": {"type": "json_object"} if is_json else {"type": "text"},
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()  # 抛出4xx/5xx错误
            return response.json()
        except httpx.HTTPStatusError as e:
            # 打印详细信息以便调试
            print(f"HTTP错误 {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            print(f"请求异常: {e}")
            raise