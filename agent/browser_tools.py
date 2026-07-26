"""
内置浏览器操作工具
基于 Playwright 提供持久化浏览器会话，支持网页导航、交互、截图等操作。
使用同步 API + 单线程执行器，避免 Windows 下 asyncio/greenlet 线程切换问题。
"""
import asyncio
import base64
import concurrent.futures
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Literal
from langchain.tools import tool
from playwright_stealth import Stealth

logger = logging.getLogger(__name__)

# Playwright 专用单线程执行器（sync API 的 greenlet 要求所有操作在同一线程）
_playwright_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# 默认显示浏览器窗口（设 BROWSER_HEADLESS=1 可切换为无头模式）
_BROWSER_HEADLESS = os.environ.get("BROWSER_HEADLESS", "0") == "1"

# 浏览器通道：设为 "chromium" 使用 Playwright 自带 Chromium，默认 "chrome" 使用系统 Chrome
_BROWSER_CHANNEL = os.environ.get("BROWSER_CHANNEL", "chrome")

# Chrome 可执行路径（channel="chrome" 时自动从注册表查找，也可手动指定）
_CHROME_EXECUTABLE_PATH = os.environ.get("CHROME_PATH", "") or None

# CDP 远程调试端口：设为 9222 则连接到已有 Chrome 而非启动新实例（终极反检测方案）
# 使用方式：先手动启动 Chrome 并添加参数 --remote-debugging-port=9222
_BROWSER_CDP_PORT = os.environ.get("BROWSER_CDP_PORT", "")

# 浏览器会话单例
_browser_session = None
_lock = threading.Lock()
_chromium_checked = False

# 浏览器数据目录（持久化 cookie、localStorage 等）
BROWSER_DATA_DIR = Path(__file__).parent.parent / "browser_data"

# 截图保存目录
SCREENSHOT_DIR = Path(__file__).parent.parent / "screenshots"

# 反自动化检测：Chrome 启动参数
_STEALTH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-infobars",
    "--disable-background-networking",
    "--disable-sync",
    "--disable-translate",
    "--disable-default-apps",
    "--disable-component-extensions-with-background-pages",
    "--disable-client-side-phishing-detection",
    "--disable-component-update",
    "--password-store=basic",
    "--use-mock-keychain",
]

# 真实 Chrome 131 Windows User-Agent
_STEALTH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)



def _install_chromium():
    """自动安装 Playwright Chromium 浏览器（已有则跳过）。返回 True 表示安装成功。"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("Chromium 浏览器已就绪")
            return True
        else:
            logger.warning(
                f"Chromium 安装可能失败 (code={result.returncode}): {result.stderr.strip()}"
            )
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Chromium 安装超时，将尝试继续启动")
        return False
    except Exception as e:
        logger.warning(f"自动安装 Chromium 时出错: {e}，将尝试继续启动")
        return False


def _cleanup_old_screenshots(days: int = 30):
    """删除超过指定天数的截图文件。"""
    import time
    if not SCREENSHOT_DIR.exists():
        return
    cutoff = time.time() - days * 86400
    deleted = 0
    for f in SCREENSHOT_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass
    if deleted > 0:
        logger.info(f"已清理 {deleted} 个过期截图（>{days}天）")


class BrowserSession:
    """管理 Playwright 持久化浏览器会话（同步 API，在独立线程中运行）"""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False

    def _ensure_initialized(self):
        """延迟初始化浏览器（同步）"""
        if self._initialized and self._page:
            return

        with _lock:
            if self._initialized and self._page:
                return

            # 首次使用时自动安装 Chromium
            global _chromium_checked
            if not _chromium_checked:
                if _install_chromium():
                    _chromium_checked = True

            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                raise ImportError(
                    "playwright 未安装。请运行: pip install playwright && playwright install chromium"
                )

            # Windows: Playwright 内部需要 ProactorEventLoop 来创建子进程
            # 临时切换策略，初始化完成后恢复（项目其他组件可能需要 SelectorEventLoop）
            if sys.platform == "win32":
                _old_policy = asyncio.get_event_loop_policy()
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            try:
                BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)
                SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
                _cleanup_old_screenshots()

                self._playwright = sync_playwright().start()

                # ========== CDP 模式：连接到已有 Chrome 实例（终极反检测） ==========
                if _BROWSER_CDP_PORT:
                    cdp_url = f"http://127.0.0.1:{_BROWSER_CDP_PORT}"
                    logger.info(f"通过 CDP 连接到已有 Chrome: {cdp_url}")
                    self._browser = self._playwright.chromium.connect_over_cdp(cdp_url)
                    contexts = self._browser.contexts
                    if contexts:
                        self._context = contexts[0]
                    else:
                        self._context = self._browser.new_context()
                    pages = self._context.pages
                    if pages:
                        self._page = pages[-1]  # 使用最近打开的标签页
                    else:
                        self._page = self._context.new_page()
                    self._initialized = True
                    logger.info(f"已连接到已有 Chrome 会话（{len(pages)} 个标签页）")
                    return

                launch_error = None
                try:
                    launch_kwargs = dict(
                        user_data_dir=str(BROWSER_DATA_DIR),
                        headless=self.headless,
                        channel=_BROWSER_CHANNEL,
                        args=[
                            "--no-sandbox",
                            "--disable-setuid-sandbox",
                            "--disable-dev-shm-usage",
                            "--force-device-scale-factor=1",
                        ] + _STEALTH_ARGS + (["--disable-gpu"] if self.headless else []),
                        viewport={"width": 1280, "height": 800},
                        locale="zh-CN",
                        user_agent=_STEALTH_USER_AGENT,
                        ignore_https_errors=True,
                    )
                    if _CHROME_EXECUTABLE_PATH:
                        launch_kwargs["executable_path"] = _CHROME_EXECUTABLE_PATH
                    self._context = self._playwright.chromium.launch_persistent_context(**launch_kwargs)
                except Exception as e:
                    launch_error = e
                    logger.warning(f"持久化上下文启动失败: {type(e).__name__}: {e}，尝试普通启动...")

                if launch_error is not None:
                    try:
                        fallback_kwargs = dict(
                            headless=self.headless,
                            channel=_BROWSER_CHANNEL,
                            args=[
                                "--no-sandbox",
                                "--disable-setuid-sandbox",
                                "--disable-dev-shm-usage",
                                "--force-device-scale-factor=1",
                            ] + _STEALTH_ARGS + (["--disable-gpu"] if self.headless else []),
                        )
                        if _CHROME_EXECUTABLE_PATH:
                            fallback_kwargs["executable_path"] = _CHROME_EXECUTABLE_PATH
                        self._browser = self._playwright.chromium.launch(**fallback_kwargs)
                        self._context = self._browser.new_context(
                            viewport={"width": 1280, "height": 800},
                            locale="zh-CN",
                            user_agent=_STEALTH_USER_AGENT,
                            ignore_https_errors=True,
                        )
                    except Exception as e2:
                        logger.error(f"浏览器启动完全失败: {type(e2).__name__}: {e2}")
                        raise RuntimeError(
                            f"无法启动浏览器。请确保已安装 Chromium: playwright install chromium\n"
                            f"持久化上下文错误: {type(launch_error).__name__}: {launch_error}\n"
                            f"普通启动错误: {type(e2).__name__}: {e2}"
                        ) from e2

                pages = self._context.pages
                if pages:
                    self._page = pages[0]
                else:
                    self._page = self._context.new_page()

                # playwright-stealth：对所有页面（现有+新建）应用反检测补丁
                _stealth = Stealth()
                for p in self._context.pages:
                    try:
                        _stealth.apply_stealth_sync(p)
                    except Exception as e:
                        logger.warning(f"stealth 注入失败: {e}")

                # 注册新页面事件：后续打开的任何新页面也自动注入
                self._context.on("page", lambda p: _stealth.apply_stealth_sync(p))

                self._initialized = True
                logger.info("浏览器会话已启动（playwright-stealth 已注入）")
            finally:
                if sys.platform == "win32":
                    asyncio.set_event_loop_policy(_old_policy)

    # ========== 浏览器操作方法 ==========

    def navigate(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 30000) -> str:
        self._ensure_initialized()
        if not url.startswith("http"):
            url = "https://" + url
        try:
            self._page.goto(url, wait_until=wait_until, timeout=timeout)
            return f"已导航到: {self._page.url}\n页面标题: {self._page.title()}"
        except Exception as e:
            return f"导航失败: {e}"

    def click(self, selector: str, timeout: int = 10000) -> str:
        self._ensure_initialized()
        try:
            locator = self._resolve_locator(selector)
            locator.first.click(timeout=timeout)
            return f"已点击: {selector}"
        except Exception as e:
            return f"点击失败 ({selector}): {e}"

    def type(self, selector: str, text: str, clear_first: bool = True, timeout: int = 10000) -> str:
        self._ensure_initialized()
        try:
            locator = self._resolve_locator(selector)
            locator.first.wait_for(timeout=timeout)
            if clear_first:
                locator.first.fill(text)
            else:
                locator.first.type(text)
            return f"已在 {selector} 中输入: {text}"
        except Exception as e:
            return f"输入失败 ({selector}): {e}"

    def screenshot(self, full_page: bool = False) -> str:
        self._ensure_initialized()
        try:
            import time
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.png"
            filepath = SCREENSHOT_DIR / filename
            self._page.screenshot(path=str(filepath), full_page=full_page)
            with open(filepath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return (
                f"截图已保存: {filepath}\n"
                f"页面: {self._page.url}\n"
                f"要在聊天中展示此截图，请在回复中包含: ![截图](/screenshots/{filename})"
            )
        except Exception as e:
            return f"截图失败: {e}"

    def get_content(self) -> str:
        self._ensure_initialized()
        try:
            content = self._page.content()
            if len(content) > 10000:
                content = content[:10000] + "\n... (已截断，完整内容过长)"
            return content
        except Exception as e:
            return f"获取页面内容失败: {e}"

    def get_text(self, selector: Optional[str] = None) -> str:
        self._ensure_initialized()
        try:
            if selector:
                locator = self._resolve_locator(selector)
                text = locator.first.inner_text()
            else:
                text = self._page.inner_text("body")
            if len(text) > 8000:
                text = text[:8000] + "\n... (已截断)"
            return text
        except Exception as e:
            return f"获取文本失败: {e}"

    def execute_js(self, code: str) -> str:
        self._ensure_initialized()
        try:
            result = self._page.evaluate(code)
            return str(result)
        except Exception as e:
            return f"执行 JavaScript 失败: {e}"

    def scroll(self, direction: str = "down", amount: int = 500) -> str:
        self._ensure_initialized()
        try:
            if direction == "down":
                self._page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "up":
                self._page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "bottom":
                self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif direction == "top":
                self._page.evaluate("window.scrollTo(0, 0)")
            else:
                return f"未知滚动方向: {direction}"
            return f"已向{direction}滚动 {amount}px"
        except Exception as e:
            return f"滚动失败: {e}"

    def go_back(self) -> str:
        self._ensure_initialized()
        try:
            self._page.go_back()
            return f"已返回上一页: {self._page.url}"
        except Exception as e:
            return f"返回失败: {e}"

    def go_forward(self) -> str:
        self._ensure_initialized()
        try:
            self._page.go_forward()
            return f"已前进到: {self._page.url}"
        except Exception as e:
            return f"前进失败: {e}"

    def refresh(self) -> str:
        self._ensure_initialized()
        try:
            self._page.reload()
            return f"已刷新页面: {self._page.url}"
        except Exception as e:
            return f"刷新失败: {e}"

    def wait(self, selector: Optional[str] = None, ms: int = 3000) -> str:
        self._ensure_initialized()
        try:
            if selector:
                locator = self._resolve_locator(selector)
                locator.first.wait_for(timeout=ms)
                return f"元素 {selector} 已出现"
            else:
                self._page.wait_for_timeout(ms)
                return f"已等待 {ms}ms"
        except Exception as e:
            return f"等待超时: {e}"

    def select_option(self, selector: str, value: str) -> str:
        self._ensure_initialized()
        try:
            locator = self._resolve_locator(selector)
            locator.first.select_option(value)
            return f"已在 {selector} 中选择: {value}"
        except Exception as e:
            return f"选择失败 ({selector}): {e}"

    def press_key(self, key: str) -> str:
        self._ensure_initialized()
        try:
            self._page.keyboard.press(key)
            return f"已按下: {key}"
        except Exception as e:
            return f"按键失败: {e}"

    def hover(self, selector: str) -> str:
        self._ensure_initialized()
        try:
            locator = self._resolve_locator(selector)
            locator.first.hover()
            return f"已悬停在: {selector}"
        except Exception as e:
            return f"悬停失败 ({selector}): {e}"

    def get_url(self) -> str:
        self._ensure_initialized()
        return f"当前 URL: {self._page.url}"

    def get_title(self) -> str:
        self._ensure_initialized()
        return f"页面标题: {self._page.title()}"

    def get_elements(self, selector: str, limit: int = 20) -> str:
        self._ensure_initialized()
        try:
            locator = self._resolve_locator(selector)
            count = locator.count()
            items = []
            for i in range(min(count, limit)):
                el = locator.nth(i)
                tag = el.evaluate("el => el.tagName.toLowerCase()")
                text = el.inner_text()[:80] if el.inner_text() else ""
                visible = el.is_visible()
                items.append(f"  [{i}] <{tag}> visible={visible} text=\"{text}\"")
            result = f"找到 {count} 个匹配 '{selector}' 的元素"
            if count > limit:
                result += f"（仅显示前 {limit} 个）"
            result += ":\n" + "\n".join(items)
            return result
        except Exception as e:
            return f"查找元素失败 ({selector}): {e}"

    def _resolve_locator(self, selector: str):
        if selector.startswith("text=") or selector.startswith("role=") or selector.startswith("xpath="):
            return self._page.locator(selector)
        if selector.startswith("//") or selector.startswith("(//"):
            return self._page.locator(f"xpath={selector}")
        return self._page.locator(selector)

    def close(self):
        """关闭浏览器会话（CDP 模式下仅断开连接，不关浏览器）"""
        try:
            if not _BROWSER_CDP_PORT and self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            logger.debug("关闭浏览器会话失败", exc_info=True)
        finally:
            self._initialized = False
            logger.info("浏览器会话已关闭")


def _get_browser_session() -> BrowserSession:
    """获取全局浏览器会话（懒初始化）"""
    global _browser_session
    if _browser_session is None:
        _browser_session = BrowserSession(headless=_BROWSER_HEADLESS)
        logger.info(f"浏览器模式: {'无头' if _BROWSER_HEADLESS else '可见'}")
    return _browser_session


def _destroy_browser_session():
    """异常恢复：重置浏览器会话（下次调用自动重建）。"""
    global _browser_session
    _browser_session = None


async def close_browser_session():
    """关闭全局浏览器会话"""
    global _browser_session
    if _browser_session:
        await asyncio.to_thread(_browser_session.close)
        _browser_session = None


@tool
async def browser(
    action: Literal[
        "navigate", "click", "type", "screenshot", "get_content",
        "get_text", "execute_js", "scroll", "go_back", "go_forward",
        "refresh", "wait", "select_option", "press_key", "hover",
        "get_url", "get_title", "get_elements", "close"
    ],
    url: Optional[str] = None,
    selector: Optional[str] = None,
    text: Optional[str] = None,
    code: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[str] = None,
    direction: Optional[str] = "down",
    amount: Optional[int] = 500,
    timeout: Optional[int] = None,
    full_page: Optional[bool] = False,
    clear_first: Optional[bool] = True,
    limit: Optional[int] = 20,
    wait_until: str = "domcontentloaded",
) -> str:
    """
    内置浏览器工具，基于 Chromium 持久化会话，支持 18 种网页操作。
    首次使用前请调用 load_skill("browser-automation") 获取完整说明和选择器指南。

    操作(action): navigate/click/type/screenshot/get_content/get_text/execute_js/
                  scroll/go_back/go_forward/refresh/wait/select_option/press_key/
                  hover/get_url/get_title/get_elements/close
    """
    session = _get_browser_session()

    # 所有 Playwright 操作通过独立线程执行，避免 Windows asyncio 子进程问题
    def _run():
        if action == "navigate":
            if not url:
                return "错误: navigate 需要 url 参数"
            return session.navigate(url, wait_until=wait_until, timeout=timeout or 30000)

        elif action == "click":
            if not selector:
                return "错误: click 需要 selector 参数"
            return session.click(selector, timeout=timeout or 10000)

        elif action == "type":
            if not selector or not text:
                return "错误: type 需要 selector 和 text 参数"
            return session.type(selector, text,
                              clear_first=clear_first if clear_first is not None else True,
                              timeout=timeout or 10000)

        elif action == "screenshot":
            return session.screenshot(full_page=full_page or False)

        elif action == "get_content":
            return session.get_content()

        elif action == "get_text":
            return session.get_text(selector=selector)

        elif action == "execute_js":
            if not code:
                return "错误: execute_js 需要 code 参数"
            return session.execute_js(code)

        elif action == "scroll":
            return session.scroll(direction=direction or "down", amount=amount or 500)

        elif action == "go_back":
            return session.go_back()

        elif action == "go_forward":
            return session.go_forward()

        elif action == "refresh":
            return session.refresh()

        elif action == "wait":
            return session.wait(selector=selector, ms=timeout or 3000)

        elif action == "select_option":
            if not selector or not value:
                return "错误: select_option 需要 selector 和 value 参数"
            return session.select_option(selector, value)

        elif action == "press_key":
            if not key:
                return "错误: press_key 需要 key 参数"
            return session.press_key(key)

        elif action == "hover":
            if not selector:
                return "错误: hover 需要 selector 参数"
            return session.hover(selector)

        elif action == "get_url":
            return session.get_url()

        elif action == "get_title":
            return session.get_title()

        elif action == "get_elements":
            if not selector:
                return "错误: get_elements 需要 selector 参数"
            return session.get_elements(selector, limit=limit or 20)

        elif action == "close":
            session.close()
            global _browser_session
            _browser_session = None
            return "浏览器已关闭。如需继续使用，再次调用 browser 工具即可自动打开。"

        else:
            return f"未知操作: {action}"

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_playwright_executor, _run)
    except ImportError as e:
        return str(e)
    except Exception as e:
        import traceback
        err_detail = f"{type(e).__name__}: {e}"
        logger.error(f"浏览器操作 {action} 失败:\n{traceback.format_exc()}")
        # greenlet 线程错误 → 会话已损坏，销毁后下次调用会自动重建
        if "Cannot switch to a different thread" in str(e) or "greenlet" in type(e).__name__.lower():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_playwright_executor, _destroy_browser_session)
            return f"浏览器操作 {action} 失败: {err_detail}\n\n浏览器会话已重置，请重试操作。"
        return f"浏览器操作 {action} 失败: {err_detail}"
