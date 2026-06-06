"""
电脑操作工具 - 基于 Open Interpreter 架构 + GLM-4V-FLASH 视觉模型
提供完整的电脑控制能力：鼠标、键盘、截图、视觉识别、命令执行

底层实现：pyautogui（鼠标/键盘）+ Pillow（截图）+ GLM-4V-FLASH（视觉理解）
架构理念：采用 Open Interpreter 的计算机控制范式，Agent 通过视觉模型理解屏幕、
         再执行精确的鼠标/键盘操作来控制电脑。
可选增强：安装 open-interpreter 包（pip install open-interpreter==0.3.14）可启用其 computer 模块。
"""
import os
import io
import json
import base64
import logging
import subprocess
import traceback
import warnings
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

from PIL import Image, ImageGrab, ImageDraw, ImageFont
from langchain.tools import tool

from agent.model_config import model_config
from agent.tools import windows_automation
from config import config

# pygetwindow：查找和操作窗口
_pygw_available = False
try:
    import pygetwindow as _pygw
    _pygw_available = True
except ImportError:
    pass

# pyperclip：剪贴板操作
_pyperclip_available = False
try:
    import pyperclip as _pyperclip
    _pyperclip_available = True
except ImportError:
    pass

# EasyOCR：纯 Python OCR，支持中文，无需系统安装
_easyocr_available = False
_easyocr_reader = None
_easyocr_init_error = None
_easyocr_init_failed = False  # 缓存初始化失败，避免反复重试
try:
    import easyocr as _easyocr
    _easyocr_available = True
except ImportError:
    _easyocr_init_error = "easyocr 未安装，请运行: pip install easyocr"

# 尝试使用 Open Interpreter 的 computer 模块（可选，未安装时使用 pyautogui 后备）
_computer = None
_oi_available = False
try:
    from interpreter import interpreter as _oi
    _oi.llm.api_key = os.environ.get("ZHIPU_API_KEY", config.model.api_key)
    _oi.llm.api_base = "https://open.bigmodel.cn/api/paas/v4/"
    _oi.llm.model = "GLM-4V-FLASH"
    _oi.offline = True
    _computer = _oi.computer
    _oi_available = True
except ImportError:
    pass  # 使用 pyautogui 后备方案，功能完全一致

# pyautogui：鼠标/键盘控制（Open Interpreter 底层同样使用此库）
_pyautogui_available = False
try:
    import pyautogui as _pyautogui
    _pyautogui.FAILSAFE = True  # 鼠标移到角落时触发异常，安全机制
    _pyautogui_available = True
except ImportError:
    logger.warning("pyautogui 未安装，部分功能不可用")

# 临时文件目录
_TEMP_DIR = Path(os.getcwd()) / "screenshots"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 网格定位参数
# 注意：视觉模型在处理图片时会压缩，网格过密字号过小会导致数字不可读。
# 当前 20×14 配合 18px 字体，即使压缩到 1024 宽，数字仍有约 10px，可清晰辨认。
_GRID_COLS = 20   # 列数（≈96px/格 @1920宽）
_GRID_ROWS = 14   # 行数（≈77px/格 @1080高）


def _add_grid_to_image(img: Image.Image) -> tuple:
    """
    在截图上绘制粗网格 + 大号编号徽章，用于视觉模型精确定位。
    设计要点：
    - 20×14 粗网格，白色细线 + 黄色粗线每 5 格，确保在任何桌面背景下可见
    - 列号（顶部）+ 行号（左侧）白底黑字大号徽章，即使模型压缩图片也能辨认
    - 返回 (带网格的图片, 单元格宽度, 单元格高度)
    """
    w, h = img.size
    cols, rows = _GRID_COLS, _GRID_ROWS
    cell_w = w // cols
    cell_h = h // rows

    draw = ImageDraw.Draw(img, "RGBA")

    # 大号字体，确保经模型压缩后仍可读
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        logger.debug("加载 arial.ttf 字体失败，使用默认字体")
        font = ImageFont.load_default()

    # ---- 网格线 ----
    # 细白线（每格）
    for i in range(1, cols):
        x = i * cell_w
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 70), width=1)
    for i in range(1, rows):
        y = i * cell_h
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 70), width=1)
    # 粗黄线（每 5 格）
    for i in range(5, cols, 5):
        x = i * cell_w
        draw.line([(x, 0), (x, h)], fill=(0, 255, 200, 160), width=3)
    for i in range(5, rows, 5):
        y = i * cell_h
        draw.line([(0, y), (w, y)], fill=(0, 255, 200, 160), width=3)

    # ---- 列号徽章（顶部，每格一个） ----
    for i in range(cols):
        x = i * cell_w + 4
        label = str(i)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0] + 6
        th = bbox[3] - bbox[1] + 4
        # 黑色底色 + 白色文字 = 终极对比度
        draw.rectangle([(x, 2), (x + tw, th + 2)], fill=(0, 0, 0, 220))
        draw.text((x + 3, 3), label, fill=(255, 255, 255, 255), font=font)

    # ---- 行号徽章（左侧，每格一个） ----
    for i in range(rows):
        y = i * cell_h + 4
        label = str(i)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0] + 6
        th = bbox[3] - bbox[1] + 4
        draw.rectangle([(2, y), (tw + 2, y + th)], fill=(0, 0, 0, 220))
        draw.text((5, y + 1), label, fill=(255, 255, 255, 255), font=font)

    return img, cell_w, cell_h


def _take_screenshot_raw() -> Image.Image:
    """截取全屏，返回 PIL Image"""
    if _oi_available:
        try:
            # Open Interpreter 方式：使用 display.view()
            # 它返回 base64 编码的截图
            screenshot_b64 = _computer.display.view(show=False)
            if screenshot_b64:
                img_data = base64.b64decode(screenshot_b64)
                return Image.open(io.BytesIO(img_data))
        except Exception:
            logger.debug("Open Interpreter 截图失败，回退到 PIL ImageGrab", exc_info=True)
    # 后备：PIL ImageGrab
    return ImageGrab.grab()


def _screenshot_to_base64(img: Image.Image, quality: int = 75) -> str:
    """将 PIL Image 转为 base64 data URL"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ==================== 工具函数 ====================

@tool
def computer_screenshot(save: bool = False) -> str:
    """
    截取当前电脑屏幕，返回截图信息。
    如果 save=True，会保存截图文件并返回路径；否则只返回屏幕尺寸等元信息。
    配合 computer_see_and_describe 使用来理解屏幕内容。

    :param save: 是否保存截图到文件
    :return: 截图信息或文件路径
    """
    try:
        img = _take_screenshot_raw()
        info = f"截图成功。屏幕尺寸：{img.size[0]}x{img.size[1]}"
        if save:
            filename = f"screenshot_{os.getpid()}.png"
            filepath = _TEMP_DIR / filename
            img.save(str(filepath))
            info += f"，已保存到 {filepath}\n要在聊天中展示此截图，请在回复中包含: ![截图](/screenshots/{filename})"
        return info
    except Exception as e:
        return f"截图失败: {e}"


@tool
def computer_see_and_describe(task_hint: str = "请详细描述屏幕上显示的所有内容，包括窗口、文字、图标、按钮、菜单等。") -> str:
    """
    截取当前屏幕并使用视觉模型（GLM-4V-FLASH）理解屏幕内容。
    这是电脑操作的核心工具——先用它看清屏幕上有什么，再决定如何操作。

    :param task_hint: 告诉视觉模型你需要关注什么，如"屏幕上有哪些窗口和按钮？"
    :return: 视觉模型对屏幕内容的描述
    """
    try:
        img = _take_screenshot_raw()
        # 压缩图片以减少 token 消耗（限制长边不超过 1920）
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > 1920:
            scale = 1920 / max_dim
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        b64_url = _screenshot_to_base64(img)

        vision_model = model_config.get_model("computer_vision")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": task_hint},
                {"type": "image_url", "image_url": {"url": b64_url}}
            ]
        }]
        response = vision_model.invoke(messages)
        return f"屏幕描述 ({img.size[0]}x{img.size[1]}):\n{response.content}"
    except Exception as e:
        return f"视觉识别失败: {e}\n{traceback.format_exc()[:500]}"


@tool
def computer_move(x: int, y: int, smooth: bool = False) -> str:
    """
    移动鼠标光标到指定坐标。

    :param x: X 坐标（像素）
    :param y: Y 坐标（像素）
    :param smooth: 是否平滑移动（模拟人类移动轨迹）
    :return: 操作结果
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装，无法控制鼠标"
    try:
        if smooth:
            _pyautogui.moveTo(x, y, duration=0.5)
        else:
            _pyautogui.moveTo(x, y)
        return f"鼠标已移动到 ({x}, {y})"
    except Exception as e:
        return f"移动鼠标失败: {e}"


@tool
def computer_click(x: Optional[int] = None, y: Optional[int] = None, button: str = "left") -> str:
    """
    在指定坐标点击鼠标。如果不指定坐标，则在当前位置点击。

    :param x: X 坐标（可选，不指定则在当前位置点击）
    :param y: Y 坐标（可选）
    :param button: 鼠标按钮 ("left", "right", "middle")
    :return: 操作结果
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装，无法控制鼠标"
    try:
        if x is not None and y is not None:
            _pyautogui.click(x, y, button=button)
        else:
            _pyautogui.click(button=button)
        pos = _pyautogui.position()
        return f"已{button}键点击，当前位置 ({pos.x}, {pos.y})"
    except Exception as e:
        return f"点击失败: {e}"


@tool
def computer_double_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    在指定坐标双击鼠标。

    :param x: X 坐标（可选）
    :param y: Y 坐标（可选）
    :return: 操作结果
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装，无法控制鼠标"
    try:
        if x is not None and y is not None:
            _pyautogui.doubleClick(x, y)
        else:
            _pyautogui.doubleClick()
        return "已双击"
    except Exception as e:
        return f"双击失败: {e}"


@tool
def computer_right_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    在指定坐标右键点击。

    :param x: X 坐标（可选）
    :param y: Y 坐标（可选）
    :return: 操作结果
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装，无法控制鼠标"
    try:
        if x is not None and y is not None:
            _pyautogui.rightClick(x, y)
        else:
            _pyautogui.rightClick()
        return "已右键点击"
    except Exception as e:
        return f"右键点击失败: {e}"


@tool
def computer_type(text: str, interval: float = 0.05) -> str:
    """
    模拟键盘输入文本（支持中文）。注意：需要当前有激活的输入窗口。

    :param text: 要输入的文本内容
    :param interval: 每个字符之间的间隔时间（秒），中文输入建议 0.05~0.1
    :return: 操作结果
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装，无法控制键盘"
    try:
        _pyautogui.typewrite(text, interval=interval)
        return f"已输入文本: {text[:50]}{'...' if len(text) > 50 else ''}"
    except Exception as e:
        return f"输入文本失败: {e}"


@tool
def computer_key_press(keys: str) -> str:
    """
    按下键盘按键或组合键。如 'enter', 'ctrl+c', 'alt+tab', 'win+d'。

    :param keys: 按键名称，多个按键用 + 连接。常见按键: enter, space, tab, escape, backspace, delete,
                 up/down/left/right, home, end, pageup, pagedown,
                 f1-f12, ctrl, alt, shift, win
    :return: 操作结果
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装，无法控制键盘"
    try:
        _pyautogui.hotkey(*keys.split("+"))
        return f"已按下组合键: {keys}"
    except Exception as e:
        return f"按键失败: {e}"


@tool
def computer_scroll(clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    滚动鼠标滚轮。

    :param clicks: 滚动量，正数向上、负数向下（如 3 向上滚3格，-5 向下滚5格）
    :param x: 先移动鼠标到此 X 坐标再滚动（可选）
    :param y: 先移动鼠标到此 Y 坐标再滚动（可选）
    :return: 操作结果
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装，无法控制鼠标"
    try:
        if x is not None and y is not None:
            _pyautogui.moveTo(x, y)
        _pyautogui.scroll(clicks)
        return f"已滚动 {clicks} 格"
    except Exception as e:
        return f"滚动失败: {e}"


@tool
def computer_drag(x1: int, y1: int, x2: int, y2: int, button: str = "left", duration: float = 0.5) -> str:
    """
    从起点拖拽到终点（模拟鼠标按住并拖动）。

    :param x1: 起点 X 坐标
    :param y1: 起点 Y 坐标
    :param x2: 终点 X 坐标
    :param y2: 终点 Y 坐标
    :param button: 鼠标按钮
    :param duration: 拖拽持续时间（秒）
    :return: 操作结果
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装，无法控制鼠标"
    try:
        _pyautogui.moveTo(x1, y1)
        _pyautogui.drag(x2 - x1, y2 - y1, duration=duration, button=button)
        return f"已从 ({x1}, {y1}) 拖拽到 ({x2}, {y2})"
    except Exception as e:
        return f"拖拽失败: {e}"


@tool
def computer_get_screen_size() -> str:
    """
    获取当前屏幕分辨率。

    :return: 屏幕宽度和高度
    """
    try:
        if _pyautogui_available:
            w, h = _pyautogui.size()
            return f"屏幕分辨率: {w}x{h}"
        else:
            img = ImageGrab.grab()
            return f"屏幕分辨率: {img.size[0]}x{img.size[1]}"
    except Exception as e:
        return f"获取屏幕尺寸失败: {e}"


@tool
def computer_execute(command: str, timeout: int = 30) -> str:
    """
    在电脑上执行 Shell 命令（Windows cmd）。

    :param command: 要执行的命令
    :param timeout: 超时时间（秒）
    :return: 命令输出
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.home())
        )
        output = result.stdout.strip() or result.stderr.strip()
        if not output:
            output = f"命令执行完成，返回码: {result.returncode}"
        if len(output) > 4000:
            output = output[:4000] + "\n...(输出已截断)"
        return output
    except subprocess.TimeoutExpired:
        return f"命令执行超时（{timeout}秒）: {command}"
    except Exception as e:
        return f"执行命令失败: {e}"


@tool
def computer_get_cursor_position() -> str:
    """
    获取当前鼠标光标位置。

    :return: 当前光标坐标
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装"
    try:
        pos = _pyautogui.position()
        return f"当前光标位置: ({pos.x}, {pos.y})"
    except Exception as e:
        return f"获取光标位置失败: {e}"


@tool
def computer_locate(target: str) -> str:
    """
    在屏幕上查找指定目标（图标、按钮、窗口、文字等），返回点击用的精确像素坐标。
    **这是点击前必用的工具**——先用它拿到精确坐标，再用 computer_click 点击。

    工作原理：在截图上叠加 24×16 编号网格，视觉模型只需识别目标在哪个格子（远比
    估算像素坐标准确），然后由程序计算格子中心坐标作为点击目标。

    :param target: 要定位的目标描述，如"微信图标"、"Chrome 图标"、"开始菜单"、"保存按钮"
    :return: 目标中心坐标 (x, y)，可直接用于 computer_click
    """
    try:
        img = _take_screenshot_raw()
        w, h = img.size

        # 限制尺寸
        max_dim = max(w, h)
        if max_dim > 1920:
            scale = 1920 / max_dim
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = img.size

        # 画网格
        gridded, cell_w, cell_h = _add_grid_to_image(img)

        # 保存调试图片
        debug_path = _TEMP_DIR / "locate_grid.jpg"
        gridded.save(str(debug_path), quality=85)

        b64_url = _screenshot_to_base64(gridded, quality=80)

        prompt = (
            f"这张截图分辨率 {w}x{h}，已叠加 {_GRID_COLS}×{_GRID_ROWS} 的定位网格。\n\n"
            f"网格说明：\n"
            f"- 顶部白底黑字数字 = 列号（0~{_GRID_COLS - 1}，从左到右）\n"
            f"- 左侧白底黑字数字 = 行号（0~{_GRID_ROWS - 1}，从上到下）\n"
            f"- 黄色粗线每 5 格一条，方便快速定位\n\n"
            f"任务：找到「{target}」中心所在的网格单元格。\n"
            f"即使目标是任务栏小图标，也请仔细观察它最接近哪个格子。\n\n"
            f"输出纯 JSON（不要额外文字）：\n"
            f'{{"found": true/false, "col": 整数列号, "row": 整数行号, '
            f'"description": "简要描述目标外观和位置"}}\n\n'
            f"示例：如果 Chrome 图标在顶部第 10 列附近、第 3 行附近 → "
            f'{{"found": true, "col": 10, "row": 3, "description": "Chrome彩色圆形图标，任务栏左起第3个"}}'
        )

        vision_model = model_config.get_model("computer_vision")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_url}}
            ]
        }]
        response = vision_model.invoke(messages)
        content = response.content.strip()

        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        data = json.loads(content)

        if not data.get("found"):
            return (f"未在屏幕上找到「{target}」。\n"
                    f"屏幕内容：{data.get('description', '未知')}\n"
                    f"建议：切换到桌面（win+d）后重试，或用 computer_find_app 搜索。")

        col = int(data["col"])
        row = int(data["row"])

        # 计算格子中心的像素坐标
        cx = col * cell_w + cell_w // 2
        cy = row * cell_h + cell_h // 2

        # 确保坐标在屏幕范围内
        cx = max(0, min(cx, w))
        cy = max(0, min(cy, h))

        return (f"已定位到「{target}」→ 点击坐标 ({cx}, {cy})\n"
                f"所在网格：第 {col} 列 第 {row} 行（共 {_GRID_COLS}×{_GRID_ROWS} 格）\n"
                f"目标外观：{data.get('description', '')}\n"
                f"下一步：computer_click(x={cx}, y={cy})")

    except json.JSONDecodeError:
        return f"定位失败：视觉模型返回格式异常，原始响应: {content[:500]}"
    except Exception as e:
        return f"定位失败: {e}\n{traceback.format_exc()[:500]}"


@tool
def computer_paste(text: str) -> str:
    """
    通过剪贴板粘贴文本（比 computer_type 更适合中文和多行文本）。
    先将文本复制到系统剪贴板，然后执行 Ctrl+V 粘贴。
    **向微信/QQ/任何输入框发送中文消息时，必须用此工具，不要用 computer_type。**

    :param text: 要粘贴的文本内容
    :return: 操作结果
    """
    if not _pyautogui_available:
        return "错误: pyautogui 未安装，无法模拟按键"
    try:
        # 写入剪贴板（优先 pyperclip，否则 PowerShell）
        if _pyperclip_available:
            _pyperclip.copy(text)
        else:
            # PowerShell Set-Clipboard
            ps_cmd = f'Set-Clipboard -Value ([System.Net.WebUtility]::HtmlDecode("{text}"))'
            subprocess.run(
                ["powershell", "-Command",
                 f'$t = @\"\n{text}\n\"@; Set-Clipboard -Value $t'],
                capture_output=True, timeout=5, shell=False
            )
        # 粘贴
        _pyautogui.hotkey("ctrl", "v")
        return f"已粘贴文本 ({len(text)} 字符): {text[:80]}{'...' if len(text) > 80 else ''}"
    except Exception as e:
        # 降级：用 typewrite 逐字输入
        try:
            _pyautogui.typewrite(text, interval=0.03)
            return f"已输入文本（降级模式）: {text[:80]}{'...' if len(text) > 80 else ''}"
        except Exception:
            logger.debug("pyautogui 降级输入也失败", exc_info=True)
            return f"粘贴失败: {e}"


@tool
def computer_find_window(window_title: str, activate: bool = True) -> str:
    """
    在打开的窗口中按标题查找窗口，返回其精确位置和大小。
    找到窗口后可自动激活（切换到前台），然后可用坐标点击窗口内的元素。

    适用场景：
    - 找微信窗口 → computer_find_window("微信")
    - 找 Chrome → computer_find_window("Chrome")
    - 找任意程序窗口 → computer_find_window("程序名")

    :param window_title: 窗口标题的关键词（支持部分匹配），如"微信"、"Chrome"、"记事本"
    :param activate: 是否激活窗口（设为前台），默认 True
    :return: 窗口位置 (left, top, width, height) 和状态
    """
    if not _pygw_available:
        return "错误: pygetwindow 未安装，无法查找窗口"

    try:
        windows = _pygw.getWindowsWithTitle(window_title)
        if not windows:
            return (f"未找到标题包含「{window_title}」的窗口。\n"
                    f"建议：检查程序是否已启动，或先用 computer_find_app 搜索并启动。")

        # 取第一个可见窗口，过滤掉不可见的
        visible = [w for w in windows if w.visible and w.width > 0 and w.height > 0]
        if visible:
            win = visible[0]
        else:
            # 都没 visible 属性，选第一个有合法尺寸的
            valid = [w for w in windows if w.width > 0 and w.height > 0]
            win = valid[0] if valid else windows[0]

        # 检测是否最小化（坐标异常 或 isMinimized）
        was_minimized = False
        if hasattr(win, 'isMinimized') and win.isMinimized:
            was_minimized = True
        elif win.left <= -10000 or win.top <= -10000:
            was_minimized = True

        # 最小化时先还原再激活
        if was_minimized and activate:
            try:
                win.restore()
                import time
                time.sleep(0.4)
                # 重新获取窗口列表，拿到还原后的真实坐标
                windows2 = _pygw.getWindowsWithTitle(window_title)
                if windows2:
                    visible2 = [w for w in windows2 if w.visible and w.width > 0 and w.height > 0]
                    win = visible2[0] if visible2 else windows2[0]
                # 再次检查坐标是否正常
                if win.left <= -10000 or win.top <= -10000:
                    raise Exception("还原后坐标仍异常")
                was_minimized = False
            except Exception as e:
                # restore 失败可能是窗口被收到托盘了，尝试用 alt+tab 切换
                try:
                    _pyautogui.keyDown("alt")
                    _pyautogui.press("tab")
                    _pyautogui.keyUp("alt")
                    time.sleep(0.5)
                except Exception:
                    logger.debug(f"Alt+Tab 切换窗口失败", exc_info=True)
                return (f"窗口「{win.title}」处于最小化或系统托盘状态，无法还原。\n"
                        f"建议：1. 点击任务栏图标手动恢复窗口后重试\n"
                        f"      2. 或用 computer_find_app 重新搜索并启动程序。")

        info = (f"找到窗口「{win.title}」\n"
                f"  位置: left={win.left}, top={win.top}\n"
                f"  大小: {win.width}x{win.height}\n"
                f"  区域: ({win.left},{win.top}) → ({win.left + win.width},{win.top + win.height})")

        if activate:
            try:
                win.activate()
                if was_minimized:
                    info += f"\n  状态: 已还原并激活"
                else:
                    info += f"\n  状态: 已激活（切换到前台）"
            except Exception:
                logger.debug(f"窗口激活失败: {win.title}", exc_info=True)
                info += f"\n  状态: 激活失败，可用 computer_click 点击窗口标题栏"

        # 给出常用点击位置的建议（窗口最小化时不提供坐标，因为坐标无效）
        if win.width > 0 and win.height > 0 and not was_minimized:
            center_x = win.left + win.width // 2
            center_y = win.top + win.height // 2
            bottom_center_x = win.left + win.width // 2
            bottom_y = win.top + win.height - 30
            info += (f"\n  建议点击:\n"
                     f"    - 窗口标题栏（激活窗口）: computer_click(x={center_x}, y={win.top + 15})\n"
                     f"    - 窗口中心: computer_click(x={center_x}, y={center_y})\n"
                     f"    - 输入区域（窗口底部）: computer_click(x={bottom_center_x}, y={bottom_y})")

        return info

    except Exception as e:
        return f"查找窗口失败: {e}\n{traceback.format_exc()[:500]}"


@tool
def computer_ocr_find(text: str, lang: str = "ch_sim+en") -> str:
    """
    使用 EasyOCR 在屏幕上查找指定文字，返回精确的像素坐标。
    **这是最精确的定位方式**——直接识别屏幕上的文字，无需视觉模型估算。

    适用场景：
    - 找桌面图标名称（如"微信"、"Chrome"、"此电脑"）
    - 找任务栏图标标签
    - 找窗口标题、按钮文字、菜单项
    - 找对话框中的任何文字

    :param text: 要查找的文字，如"微信"、"Chrome"、"确定"、"保存"
    :param lang: OCR 语言代码，默认 "ch_sim+en"（中文简体+英文），可选 "en"（纯英文）、"ch_sim"（纯中文）
    :return: 找到的文字坐标列表，按置信度排序
    """
    global _easyocr_reader, _easyocr_init_failed

    if not _easyocr_available:
        return f"OCR 功能不可用：{_easyocr_init_error or 'easyocr 未安装'}"

    if _easyocr_init_failed:
        return "OCR 模型下载失败（网络不可达），已跳过 OCR。请改用 computer_find_window 或 computer_locate。"

    try:
        import numpy as np

        img = _take_screenshot_raw()
        w, h = img.size

        # 限制图片大小以加快 OCR
        max_dim = max(w, h)
        if max_dim > 1920:
            scale = 1920 / max_dim
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = img.size

        # 懒初始化 EasyOCR reader
        if _easyocr_reader is None:
            lang_list = lang.split("+")
            # 抑制 torch 在无 GPU 环境下的 pin_memory 警告
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message=".*pin_memory.*")
                _easyocr_reader = _easyocr.Reader(lang_list, gpu=False)

        # PIL Image → numpy array for EasyOCR
        img_array = np.array(img.convert("RGB"))
        img_array = img_array[:, :, ::-1]  # RGB → BGR

        # OCR 识别
        results = _easyocr_reader.readtext(img_array, detail=1)

        # 搜索匹配的文字
        matches = []
        text_lower = text.lower()
        for bbox, word, conf in results:
            if not word or not word.strip():
                continue
            if conf < 0.2:  # 过滤低置信度
                continue
            if text_lower in word.lower():
                # bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 四点坐标
                x1 = int(bbox[0][0])
                y1 = int(bbox[0][1])
                x2 = int(bbox[2][0])
                y2 = int(bbox[2][1])
                bw = x2 - x1
                bh = y2 - y1
                cx = x1 + bw // 2
                cy = y1 + bh // 2
                matches.append({
                    "text": word.strip(),
                    "center_x": cx,
                    "center_y": cy,
                    "x": x1, "y": y1,
                    "width": bw, "height": bh,
                    "confidence": round(conf * 100, 1)
                })

        if not matches:
            return (f"OCR 未在屏幕上找到「{text}」。\n"
                    f"可能原因：1. 文字不在当前屏幕 2. 字体太小或模糊\n"
                    f"建议：先用 computer_see_and_describe 查看屏幕布局，"
                    f"再用 computer_find_window 定位窗口。")

        # 按置信度排序
        matches.sort(key=lambda m: m["confidence"], reverse=True)

        # 返回前 5 个匹配
        result = f"OCR 找到 {len(matches)} 处「{text}」匹配：\n"
        for i, m in enumerate(matches[:5]):
            result += (f"  [{i + 1}] \"{m['text']}\" 置信度 {m['confidence']}% "
                       f"→ 点击坐标 ({m['center_x']}, {m['center_y']})\n"
                       f"      区域: ({m['x']},{m['y']}) → ({m['x'] + m['width']},{m['y'] + m['height']})\n")

        best = matches[0]
        result += f"\n推荐使用: computer_click(x={best['center_x']}, y={best['center_y']})"
        return result

    except Exception as e:
        err_msg = str(e)
        # 网络不可达 → 缓存失败状态，避免反复重试
        if "WinError 10060" in err_msg or "URLError" in err_msg or "timeout" in err_msg.lower():
            _easyocr_init_failed = True
            return ("OCR 模型下载失败（网络无法访问 GitHub），已禁用 OCR 功能。\n"
                    "请改用以下定位方式：\n"
                    "1. computer_find_window(window_title='窗口名') —— 最推荐\n"
                    "2. computer_find_app(app_name='程序名') → 搜索并启动程序\n"
                    "3. computer_locate(target='描述') —— 视觉网格定位（最后手段）")
        return f"OCR 识别失败: {e}\n{traceback.format_exc()[:300]}"


@tool
def computer_find_app(app_name: str) -> str:
    """
    在 Windows 系统中搜索应用程序，返回其可执行文件路径。
    搜索范围：开始菜单快捷方式、常见安装目录（Program Files）、PATH 环境变量。
    找到后可以用 computer_execute 启动（如 'start \"\" \"路径\"'）。

    :param app_name: 应用程序名称（支持中文），如"微信"、"QQ"、"Chrome"、"记事本"、"WPS"
    :return: 找到的可执行文件路径列表
    """
    import winreg

    results = []
    name_lower = app_name.lower()

    # 1. 搜索开始菜单快捷方式
    start_menu_dirs = [
        os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
        r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
    ]
    for menu_dir in start_menu_dirs:
        if not os.path.exists(menu_dir):
            continue
        for root, dirs, files in os.walk(menu_dir):
            depth = root[len(menu_dir):].count(os.sep)
            if depth > 4:
                continue
            for file in files:
                if not file.endswith(".lnk") and not file.endswith(".url"):
                    continue
                file_lower = file.lower()
                # 文件名匹配（包含关系）
                if name_lower in file_lower or any(
                    part in file_lower for part in name_lower.split()
                ):
                    # 解析 .lnk 文件
                    shortcut_path = os.path.join(root, file)
                    try:
                        ps_cmd = (
                            f'$sh = New-Object -ComObject WScript.Shell; '
                            f'$lnk = $sh.CreateShortcut("{shortcut_path}"); '
                            f'Write-Host $lnk.TargetPath'
                        )
                        result = subprocess.run(
                            ["powershell", "-Command", ps_cmd],
                            capture_output=True, text=True, timeout=3
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            target = result.stdout.strip()
                            if target and os.path.exists(target):
                                results.append(target)
                    except Exception:
                        logger.debug(f"powershell 快捷方式解析失败: {shortcut_path}", exc_info=True)
                        results.append(shortcut_path)  # 降级：直接返回快捷方式路径

    # 2. 搜索常见安装目录
    search_dirs = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs"),
        os.path.expandvars(r"%LOCALAPPDATA%"),
    ]
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            depth = root[len(base_dir):].count(os.sep)
            if depth > 3:
                continue
            for d in dirs:
                if name_lower in d.lower():
                    # 在匹配的目录中找 .exe
                    app_dir = os.path.join(root, d)
                    try:
                        for f in os.listdir(app_dir):
                            if f.lower().endswith(".exe") and name_lower in f.lower():
                                results.append(os.path.join(app_dir, f))
                    except OSError:
                        logger.debug(f"遍历应用目录失败: {app_dir}", exc_info=True)

    # 3. 使用 where 命令搜索 PATH
    try:
        result = subprocess.run(
            ["where", app_name + ".exe"],
            capture_output=True, text=True, timeout=5, shell=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line and os.path.exists(line):
                    results.append(line)
    except Exception:
        logger.debug(f"where 命令搜索 {app_name} 失败", exc_info=True)

    # 4. 搜索注册表 App Paths
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
        )
        i = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(key, i)
                if name_lower in subkey_name.lower():
                    subkey = winreg.OpenKey(key, subkey_name)
                    path, _ = winreg.QueryValueEx(subkey, "")
                    if path and os.path.exists(path):
                        results.append(path)
                i += 1
            except OSError:
                break
    except Exception:
        logger.debug(f"注册表 App Paths 搜索 {app_name} 失败", exc_info=True)

    # 去重并返回
    seen = set()
    unique = []
    for r in results:
        r_lower = r.lower()
        if r_lower not in seen:
            seen.add(r_lower)
            unique.append(r)

    if not unique:
        return (f"未找到「{app_name}」。建议：\n"
                f"1. 检查拼写是否正确\n"
                f"2. 尝试用 computer_execute 执行 'where {app_name}' 手动查找\n"
                f"3. 确认该程序已安装在系统中")

    paths = "\n".join(f"  - {p}" for p in unique[:5])
    launch_cmd = f'start "" "{unique[0]}"'
    return (f"找到「{app_name}」的 {len(unique)} 个相关程序：\n{paths}\n\n"
            f"启动命令（推荐首个）：computer_execute(command='{launch_cmd}')")


# 导出所有电脑操作工具列表
COMPUTER_TOOLS = [
    windows_automation,
    computer_screenshot,
    computer_see_and_describe,
    computer_ocr_find,
    computer_locate,
    computer_find_app,
    computer_find_window,
    computer_paste,
    computer_move,
    computer_click,
    computer_double_click,
    computer_right_click,
    computer_type,
    computer_key_press,
    computer_scroll,
    computer_drag,
    computer_get_screen_size,
    computer_get_cursor_position,
    computer_execute,
]
