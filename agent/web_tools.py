"""网页抓取工具。

提供 web_fetch 工具，拉取 URL 内容并将 HTML 转换为纯文本。
"""

import re
import urllib.parse
from html.parser import HTMLParser

import requests
from langchain.tools import tool

# 伪装成浏览器，避免被拒
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_FETCH_TIMEOUT = 15  # 秒
_MAX_CONTENT_LENGTH = 100_000  # 字符，超出截断
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2MB，超出不下载

# 禁止访问的地址
_BLOCKED_HOSTS = frozenset({"127.0.0.1", "localhost", "0.0.0.0", "::1"})


class _TextExtractor(HTMLParser):
    """从 HTML 中提取纯文本。跳过 script / style / head 标签内容。"""

    def __init__(self):
        super().__init__()
        self._text: list[str] = []
        self._skip_tags: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"script", "style", "head", "noscript", "iframe", "svg"}:
            self._skip_tags.add(tag_lower)
        # 块级元素前加换行
        if tag_lower in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                          "tr", "blockquote", "section", "article", "header", "footer"}:
            self._text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        self._skip_tags.discard(tag_lower)
        if tag_lower in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                          "tr", "blockquote", "section", "article"}:
            self._text.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_tags:
            return
        text = data.strip()
        if text:
            self._text.append(text + " ")

    def get_text(self) -> str:
        raw = "".join(self._text)
        # 合并多余空白行
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        return raw.strip()


def _extract_text(html: str) -> str:
    """将 HTML 转为可读纯文本。"""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    text = parser.get_text()
    if len(text) < 50:
        # 回退：简单去标签
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


@tool
def web_fetch(url: str) -> str:
    """抓取网页 URL 的正文内容，返回纯文本。

    用于阅读文章、文档、新闻等网页的完整内容（TavilySearch 只返回摘要）。
    返回结果限制在 100000 字符内，超长内容会被截断。

    Args:
        url: 要抓取的网页地址（必须包含 http:// 或 https://）。
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return "Error: 无效的 URL 格式"

    if parsed.scheme not in ("http", "https"):
        return "Error: 仅支持 http/https 协议"

    if parsed.hostname in _BLOCKED_HOSTS:
        return "Error: 禁止访问本地地址"

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html, text/plain"},
            timeout=_FETCH_TIMEOUT,
            allow_redirects=True,
            stream=True,
        )
    except requests.Timeout:
        return f"Error: 请求超时（{_FETCH_TIMEOUT}s）"
    except requests.ConnectionError:
        return "Error: 无法连接到该地址"
    except requests.RequestException as e:
        return f"Error: 请求失败 - {e}"

    if resp.status_code != 200:
        return f"Error: HTTP {resp.status_code}"

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return f"Error: 不支持的内容类型 '{content_type}'（仅支持 HTML/纯文本）"

    try:
        raw = b""
        for chunk in resp.iter_content(chunk_size=65536):
            raw += chunk
            if len(raw) > _MAX_RESPONSE_BYTES:
                return f"Error: 页面过大（> {_MAX_RESPONSE_BYTES // 1024 // 1024}MB），不予下载"
        resp.close()
    except Exception as e:
        return f"Error: 下载失败 - {e}"

    # 检测编码
    html = ""
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            html = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if not html:
        html = raw.decode("utf-8", errors="replace")

    text = _extract_text(html)

    if not text:
        return "页面无文本内容（可能是纯 JS 渲染页面或空白页）"

    if len(text) > _MAX_CONTENT_LENGTH:
        text = text[:_MAX_CONTENT_LENGTH] + (
            f"\n\n[... 内容已截断，原文共约 {len(text)} 字符。"
            "如需完整内容，请尝试更具体的 URL ...]"
        )

    return f"[来源: {url}]\n\n{text}"
