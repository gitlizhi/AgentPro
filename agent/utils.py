# agent/utils.py
import json
from dotenv import load_dotenv
load_dotenv()
import os
import httpx
from typing import List, Dict, Any, Optional

async def call_zhipu_chat(
    user_input: str,
    model: str = "glm-5",
    temperature: float = 1.0,
    stream: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    调用智谱AI的聊天完成接口。

    :param messages: 消息列表，例如 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    :param model: 模型名称，如 "glm-5", "glm-4-flash" 等
    :param temperature: 温度参数，控制随机性
    :param stream: 是否使用流式输出
    :param max_tokens: 最大生成token数
    :return: API 返回的 JSON 数据（字典）
    """
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("环境变量 ZHIPU_API_KEY 未设置")

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [{"role": "system", "content": "你是一个意图识别的人工智能助手"}, {"role": "user", "content": user_input}]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
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