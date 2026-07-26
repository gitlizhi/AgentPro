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
    model: Optional[str] = None,
    temperature: float = 1.0,
    stream: bool = False,
    max_tokens: Optional[int] = None,
    is_json: bool = False,
) -> Dict[str, Any]:
    """
    调用大模型的API的接口，带重试机制。

    :param user_input: 用户输入字符串，会自动包装为 messages 格式
    :param model: 模型名称，默认使用 config.model.default_model
    :param temperature: 温度参数，控制随机性
    :param stream: 是否使用流式输出（当前不支持，请使用 False）
    :param max_tokens: 最大生成token数
    :param is_json: 是否启用 JSON 模式输出。先尝试 response_format，若 400 则降级重试
    :return: API 返回的 JSON 数据（字典）
    """
    if model is None:
        model = config.model.default_model

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
    }
    if is_json:
        payload["response_format"] = {"type": "json_object"}
    if max_tokens:
        payload["max_tokens"] = max_tokens

    last_error = None
    json_mode_fallback_tried = False

    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            last_error = e
            status = e.response.status_code
            # 如果在 JSON 模式下收到 400，降级重试：去掉 response_format，靠 prompt 指引输出 JSON
            if status == 400 and is_json and not json_mode_fallback_tried:
                logger.warning(
                    f"JSON 模式请求返回 400（{e.response.text[:200]}），"
                    f"降级重试：移除 response_format 依赖 prompt 指引输出 JSON"
                )
                payload.pop("response_format", None)
                json_mode_fallback_tried = True
                continue
            if attempt < _MAX_RETRIES and status in _RETRYABLE_STATUSES:
                delay = _RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    f"LLM API 调用失败 (尝试 {attempt + 1}/{_MAX_RETRIES + 1})，"
                    f"HTTP {status}，{delay:.0f}s 后重试: {e}"
                )
                await asyncio.sleep(delay)
                continue
            logger.error(f"HTTP错误 {status}: {e.response.text[:500]}")
            raise
        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES and _is_retryable_error(e):
                delay = _RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    f"LLM API 调用失败 (尝试 {attempt + 1}/{_MAX_RETRIES + 1})，"
                    f"{type(e).__name__}，{delay:.0f}s 后重试: {e}"
                )
                await asyncio.sleep(delay)
                continue
            logger.error(f"请求异常: {e}")
            raise