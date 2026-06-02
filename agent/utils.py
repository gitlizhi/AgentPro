"""
其他工具
"""
import asyncio
import json
import logging
from dotenv import load_dotenv
load_dotenv()
import os
import httpx
from typing import List, Dict, Any, Optional
from config import config

logger = logging.getLogger(__name__)

# LLM API 重试配置
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0  # 基础等待秒数 (2^attempt: 2s, 4s, 8s)
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}  # 可重试的 HTTP 状态码


def _is_retryable_error(exception: Exception) -> bool:
    """判断异常是否可重试（瞬时性错误）。"""
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in _RETRYABLE_STATUSES
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, (httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    return False


async def call_big_model_chat(
    user_input: str,
    model: str = "deepseek-chat",
    temperature: float = 1.0,
    stream: bool = False,
    max_tokens: Optional[int] = None,
    is_json: bool = False,
) -> Dict[str, Any]:
    """
    调用大模型的API的接口，带重试机制。

    :param user_input: 用户输入字符串，会自动包装为 messages 格式
    :param model: 模型名称
    :param temperature: 温度参数，控制随机性
    :param stream: 是否使用流式输出（当前不支持，请使用 False）
    :param max_tokens: 最大生成token数
    :param is_json: 是否启用 JSON 模式输出
    :return: API 返回的 JSON 数据（字典）
    """
    api_key = config.model.api_key
    if not api_key:
        raise ValueError("环境变量 API_KEY 未设置")

    base_url = config.model.base_url or "https://open.bigmodel.cn/api/paas/v4"
    url = base_url.rstrip("/") + "/chat/completions"
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

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES and _is_retryable_error(e):
                delay = _RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    f"LLM API 调用失败 (尝试 {attempt + 1}/{_MAX_RETRIES + 1})，"
                    f"{'HTTP ' + str(e.response.status_code) if isinstance(e, httpx.HTTPStatusError) else type(e).__name__}，"
                    f"{delay:.0f}s 后重试: {e}"
                )
                await asyncio.sleep(delay)
                continue
            # 不可重试的错误或已达重试上限
            if isinstance(e, httpx.HTTPStatusError):
                logger.error(f"HTTP错误 {e.response.status_code}: {e.response.text}")
            else:
                logger.error(f"请求异常: {e}")
            raise last_error