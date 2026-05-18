"""
自定义工具
"""
import json
import uuid
import os
import subprocess
import winreg
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal, Tuple, Any, List
from langchain.tools import tool
from pywinauto import Application, Desktop
from pywinauto.findwindows import ElementNotFoundError

BASE_DIR = Path(__file__).parent.absolute()   # memory_processor.py 所在的目录（agent目录）
PENDING_DIR = BASE_DIR / "data" / "pending"
MEMORIES_DIR = BASE_DIR / "data" / "memories"
INDEX_PATH = MEMORIES_DIR / "index.json"


# 辅助函数：确保目录存在
def ensure_dirs():
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

@tool
def log_memory(step_description: str, tool_name: str, input_args: str, output: str, error: str = ""):
    """
    记录一个关键步骤的原始日志，供异步记忆处理器生成摘要。
    应该在每个重要操作（工具调用、错误、用户反馈）后调用。
    """
    ensure_dirs()
    log_entry = {
        "step_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "step_description": step_description,
        "tool_name": tool_name,
        "input": input_args,
        "output": output,
        "error": error,
        "success": error == ""
    }
    filename = f"{log_entry['step_id']}.json"
    filepath = PENDING_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
    return f"Logged memory step {log_entry['step_id']}"


@tool
def retrieve_memory(query: str, max_summaries: int = 10) -> str:
    """
    根据任务描述检索相关记忆的摘要列表（二级渐进的第一、二级）。
    返回摘要文本，Agent 可根据需要再调用 load_full_memory 加载完整内容。
    排序依据：置信度 × 时间衰减（越近使用权重越高）
    """
    ensure_dirs()
    if not INDEX_PATH.exists():
        return "No memory index found. No past experiences available."
    
    with open(INDEX_PATH, 'r') as f:
        index = json.load(f)
    
    # 第一级：从索引中选出最相关的标签（简单文本匹配）
    query_lower = query.lower()
    candidate_tags = []
    for tag_info in index.get("top_tags", []):
        tag = tag_info["tag"]
        if tag.lower() in query_lower or query_lower in tag.lower():
            candidate_tags.append(tag)
    if not candidate_tags:
        candidate_tags = [t["tag"] for t in index.get("top_tags", [])[:3]]
    
    # 辅助函数：解析 frontmatter 中的字段
    def parse_frontmatter(content: str) -> dict:
        """从 Markdown 文件内容中提取 frontmatter 字段"""
        result = {"confidence": 0.5, "last_used": None, "step_summary": "", "task_summary": ""}
        lines = content.split("\n")
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                if not in_frontmatter:
                    in_frontmatter = True
                else:
                    break
                continue
            if in_frontmatter and ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key == "confidence":
                    try:
                        result["confidence"] = float(val)
                    except:
                        pass
                elif key == "last_used":
                    result["last_used"] = val
                elif key == "step_summary":
                    result["step_summary"] = val
                elif key == "task_summary":
                    result["task_summary"] = val
        return result
    
    def calculate_score(confidence: float, last_used_str: str, now=None) -> float:
        """计算综合得分：置信度 × 时间衰减因子（每天衰减5%）"""
        if now is None:
            now = datetime.now()
        if not last_used_str:
            return confidence  # 无时间信息，仅用置信度
        try:
            last_used = datetime.fromisoformat(last_used_str)
            days_diff = (now - last_used).days
            decay = 0.95 ** days_diff  # 每天衰减5%
            return confidence * decay
        except:
            return confidence
    
    # 第二级：按标签搜索摘要文件，收集候选记忆
    memories = []  # 每个元素为 (score, file_name, step_summary, task_summary, confidence)
    for tag in candidate_tags[:3]:  # 最多3个标签
        for md_file in MEMORIES_DIR.glob("*.md"):
            # 简单检查文件中是否包含该标签（避免重复添加同一个文件）
            content = md_file.read_text(encoding='utf-8')
            if f"tags: {tag}" not in content and f"tags:[\"{tag}\"]" not in content:
                continue
            # 解析 frontmatter
            meta = parse_frontmatter(content)
            # 计算得分
            score = calculate_score(meta["confidence"], meta.get("last_used"))
            # 去重：同一个文件只保留最高分（理论上只会出现一次）
            existing = next((m for m in memories if m[1] == md_file.name), None)
            if existing:
                if score > existing[0]:
                    existing[0] = score
            else:
                memories.append([
                    score,
                    md_file.name,
                    meta["step_summary"],
                    meta["task_summary"],
                    meta["confidence"]
                ])
    
    # 按得分降序排序，取前 max_summaries
    memories.sort(key=lambda x: x[0], reverse=True)
    memories = memories[:max_summaries]
    
    if not memories:
        return "No relevant memories found."
    
    # 构建返回的摘要文本
    result = "Found these relevant memories (sorted by relevance, higher score = more useful):\n\n"
    for idx, (score, fname, step_summary, task_summary, conf) in enumerate(memories):
        result += f"[{idx + 1}] File: {fname}\n"
        result += f"    Step summary: {step_summary}\n"
        if task_summary:
            result += f"    Task summary: {task_summary}\n"
        result += f"    Confidence: {conf:.2f}, Relevance score: {score:.3f}\n\n"
    result += "To view full details of a memory, use load_full_memory with the file name."
    return result


@tool
def load_full_memory(filename: str) -> str:
    """
    加载完整记忆内容（三级渐进的第三级）。
    filename 应该是 retrieve_memory 返回结果中的文件名。
    """
    filepath = MEMORIES_DIR / filename
    if not filepath.exists():
        return f"Memory file {filename} not found."
    content = filepath.read_text(encoding='utf-8')
    # 限制长度，避免超出上下文
    if len(content) > 8000:
        content = content[:8000] + "\n... (truncated)"
    return content


@tool
def update_memory_confidence(filename: str, success: bool):
    """
    在根据记忆成功解决问题后调用，增加置信度；失败则降低。
    """
    filepath = MEMORIES_DIR / filename
    if not filepath.exists():
        return f"Memory file {filename} not found."
    
    content = filepath.read_text(encoding='utf-8')
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("confidence:"):
            old_conf = float(line.split(":", 1)[1].strip())
            if success:
                new_conf = min(1.0, old_conf + 0.1)
            else:
                new_conf = max(0.0, old_conf - 0.1)
            new_lines.append(f"confidence: {new_conf:.2f}")
        else:
            new_lines.append(line)
    # 更新 last_used
    for i, line in enumerate(new_lines):
        if line.startswith("last_used:"):
            new_lines[i] = f"last_used: {datetime.now().isoformat()}"
            break
    filepath.write_text("\n".join(new_lines), encoding='utf-8')
    return f"Updated confidence for {filename} to {new_conf:.2f}"


def _find_app_path(app_name: str) -> Optional[str]:
    """根据应用名称（支持中文或英文）查找可执行文件路径"""
    # 1. 尝试 where 命令（在 PATH 中查找）
    try:
        result = subprocess.run(
            ['where', app_name + '.exe'],
            capture_output=True, text=True, timeout=5, shell=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    
    # 2. 搜索常见安装目录（限制深度 3，只找 .exe）
    search_dirs = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        os.path.expanduser(r"~\AppData\Local\Programs"),
        os.path.expanduser(r"~\AppData\Local"),
        os.path.expanduser(r"~\AppData\Roaming"),
    ]
    # 可能的应用名称变体（去除空格，小写）
    name_variants = [app_name, app_name.replace(" ", ""), app_name.lower()]
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            # 限制深度，避免遍历过深
            depth = root[len(base_dir):].count(os.sep)
            if depth > 3:
                continue
            for file in files:
                if file.endswith('.exe'):
                    file_lower = file.lower()
                    for variant in name_variants:
                        if variant.lower() in file_lower:
                            return os.path.join(root, file)
    
    # 3. 搜索开始菜单快捷方式
    start_menu_dirs = [
        os.path.expanduser(r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs"),
        r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
    ]
    for menu_dir in start_menu_dirs:
        if not os.path.exists(menu_dir):
            continue
        for root, dirs, files in os.walk(menu_dir):
            for file in files:
                if file.endswith('.lnk'):
                    shortcut_path = os.path.join(root, file)
                    # 解析快捷方式目标（需要 pywin32 或使用 powershell）
                    try:
                        # 使用 powershell 解析
                        ps_cmd = f'$sh = New-Object -ComObject WScript.Shell; $lnk = $sh.CreateShortcut("{shortcut_path}"); Write-Host $lnk.TargetPath'
                        result = subprocess.run(
                            ['powershell', '-Command', ps_cmd],
                            capture_output=True, text=True, timeout=2
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            target = result.stdout.strip()
                            if target and target.lower().endswith('.exe'):
                                # 检查目标文件名是否包含应用名
                                if any(variant.lower() in target.lower() for variant in name_variants):
                                    return target
                    except:
                        pass
    
    # 4. 查询注册表 App Paths
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths")
        # 枚举子项
        i = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(key, i)
                if subkey_name.lower().startswith(app_name.lower()) or app_name.lower() in subkey_name.lower():
                    subkey = winreg.OpenKey(key, subkey_name)
                    path, _ = winreg.QueryValueEx(subkey, "")
                    if path and os.path.exists(path):
                        return path
                i += 1
            except OSError:
                break
    except:
        pass
    
    return None

def _find_matching_windows(title: Optional[str] = None, class_name: Optional[str] = None) -> List:
    """返回匹配的窗口列表（用于调试和选择）"""
    windows = Desktop(backend="uia").windows()
    matches = []
    for w in windows:
        if title and title in w.window_text():
            matches.append(w)
        elif class_name and w.class_name() == class_name:
            matches.append(w)
        elif not title and not class_name:
            matches.append(w)
    return matches


def _connect_to_window(
    title: Optional[str] = None,
    class_name: Optional[str] = None,
    process_id: Optional[int] = None,
    fuzzy_title: bool = True,
    timeout: int = 5,
    index: int = 0
) -> Application:
    """
    连接到窗口，支持模糊匹配和多窗口选择（index 指定第几个匹配，0 为第一个）
    """
    try:
        if title and fuzzy_title:
            app = Application(backend="uia").connect(title_re=f".*{title}.*", timeout=timeout)
        else:
            app = Application(backend="uia").connect(
                title=title, class_name=class_name, process=process_id, timeout=timeout
            )
        return app
    except Exception as e:
        # 如果是因为多个匹配项，则手动选择第一个
        if "There are 2 elements that match" in str(e):
            matches = _find_matching_windows(title=title, class_name=class_name)
            if index < len(matches):
                return Application(backend="uia").connect(handle=matches[index].handle)
            else:
                raise Exception(f"找到 {len(matches)} 个匹配窗口，但索引 {index} 超出范围。可用的窗口标题: {[w.window_text() for w in matches]}")
        else:
            raise


# 辅助函数：根据参数定位窗口和控件
def _find_window_and_control(
    title: Optional[str] = None,
    class_name: Optional[str] = None,
    process_id: Optional[int] = None,
    control_type: Optional[str] = None,
    auto_id: Optional[str] = None,
    control_class: Optional[str] = None,
    timeout: int = 5,
    fuzzy_title: bool = True,
    window_index: int = 0,
) -> Tuple[Application, Any]:
    """
    连接到窗口并返回控件对象。
    window_index: 当有多个匹配窗口时，选择第几个（0-based）。
    """
    app = _connect_to_window(
        title=title, class_name=class_name, process_id=process_id,
        fuzzy_title=fuzzy_title, timeout=timeout, index=window_index
    )

    if control_type is None and auto_id is None and control_class is None:
        return app, app.top_window()

    dlg = app.top_window()
    try:
        if auto_id:
            ctrl = dlg.child_window(auto_id=auto_id, control_type=control_type, class_name=control_class)
        elif control_type:
            ctrl = dlg.child_window(control_type=control_type, class_name=control_class)
        elif control_class:
            ctrl = dlg.child_window(class_name=control_class)
        else:
            raise ValueError("必须提供 auto_id, control_type 或 control_class 之一")
        ctrl.wait('exists', timeout=timeout)
        return app, ctrl
    except Exception as e:
        raise Exception(f"未找到控件: {e}")


@tool
async def windows_automation(
    action: Literal[
        "start", "connect", "click", "double_click", "right_click",
        "type", "send_keys", "select", "get_text", "set_text", "wait",
        "maximize", "minimize", "restore", "close", "screenshot",
        "get_property", "scroll", "drag_drop", "menu_select"
    ],
    app_path: Optional[str] = None,
    title: Optional[str] = None,
    class_name: Optional[str] = None,
    process_id: Optional[int] = None,
    control_type: Optional[str] = None,
    auto_id: Optional[str] = None,
    control_class: Optional[str] = None,
    text: Optional[str] = None,
    value: Optional[str] = None,
    item: Optional[str] = None,
    keys: Optional[str] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    direction: Optional[str] = None,
    source_auto_id: Optional[str] = None,
    target_auto_id: Optional[str] = None,
    menu_path: Optional[str] = None,
    save_path: Optional[str] = None,
    property_name: Optional[str] = None,
    timeout: int = 10,
    window_index: int = 0,   # 新增：选择第几个匹配窗口
) -> str:
    """
    控制 Windows 应用程序。支持多窗口选择（通过 window_index 参数）。
    当连接微信出现“多个匹配”错误时，可以尝试 window_index=0 或 1。
    """
    try:
        # 处理 start 和 connect 单独，因为它们不需要控件定位
        if action == "start":
            if not app_path:
                return "错误: start 操作需要提供 app_path"
            # 如果不是完整路径（不包含反斜杠或盘符），尝试自动查找
            if not (app_path.startswith('C:') or app_path.startswith('\\')):
                actual_path = _find_app_path(app_path)
                if actual_path:
                    app_path = actual_path
                else:
                    return f"无法自动找到应用 '{app_path}' 的安装路径，请手动提供完整路径。"
            try:
                Application(backend="uia").start(app_path)
                return f"成功启动应用: {app_path}"
            except Exception as e:
                return f"启动失败: {e}"
        
        elif action == "connect":
            try:
                app = _connect_to_window(
                    title=title, class_name=class_name, process_id=process_id,
                    fuzzy_title=True, timeout=timeout, index=window_index
                )
                return f"成功连接到窗口: {title or class_name or process_id} (窗口索引 {window_index})"
            except Exception as e:
                return f"连接失败: {e}"
        
        # 对于需要窗口和控件的操作，定位控件（支持模糊标题）
        try:
            app, control = _find_window_and_control(
                title=title, class_name=class_name, process_id=process_id,
                control_type=control_type, auto_id=auto_id, control_class=control_class,
                timeout=timeout, fuzzy_title=True, window_index=window_index
            )
        except Exception as e:
            return f"定位窗口/控件失败: {e}"

        # 执行具体操作
        if action == "click":
            control.click()
            return f"已点击控件: {auto_id or control_type or control_class}"

        elif action == "double_click":
            control.double_click()
            return "已双击控件"

        elif action == "right_click":
            control.right_click()
            return "已右键点击控件"

        elif action == "type":
            if text is None:
                return "错误: type 操作需要提供 text"
            control.type_keys(text, with_spaces=True)
            return f"已输入文本: {text}"

        elif action == "send_keys":
            if keys is None:
                return "错误: send_keys 操作需要提供 keys"
            control.type_keys(keys)
            return f"已发送按键: {keys}"

        elif action == "select":
            if value is None:
                return "错误: select 操作需要提供 value"
            control.select(value)
            return f"已选择项: {value}"

        elif action == "get_text":
            txt = control.window_text()
            return f"控件文本: {txt}"

        elif action == "set_text":
            if text is None:
                return "错误: set_text 操作需要提供 text"
            control.set_text(text)
            return f"已设置文本: {text}"

        elif action == "wait":
            try:
                control.wait('exists', timeout=timeout)
                return f"控件在 {timeout} 秒内已出现"
            except:
                return f"等待超时: 控件未在 {timeout} 秒内出现"

        elif action == "maximize":
            control.maximize()
            return "窗口已最大化"

        elif action == "minimize":
            control.minimize()
            return "窗口已最小化"

        elif action == "restore":
            control.restore()
            return "窗口已还原"

        elif action == "close":
            control.close()
            return "窗口已关闭"

        elif action == "screenshot":
            if save_path is None:
                return "错误: screenshot 操作需要提供 save_path"
            control.capture_as_image().save(save_path)
            return f"截图已保存至 {save_path}"

        elif action == "get_property":
            if property_name is None:
                return "错误: get_property 操作需要提供 property_name"
            # 获取常见属性
            if hasattr(control, property_name):
                val = getattr(control, property_name)()
            else:
                try:
                    props = control.get_properties()
                    val = props.get(property_name)
                except:
                    return f"无法获取属性 {property_name}"
            return f"属性 {property_name}: {val}"

        elif action == "scroll":
            if direction == "up":
                control.scroll(direction='up', amount=1)
            elif direction == "down":
                control.scroll(direction='down', amount=1)
            elif x is not None and y is not None:
                control.scroll(x, y)
            else:
                return "错误: scroll 操作需要提供 direction 或 x,y"
            return "已执行滚动"

        elif action == "drag_drop":
            if source_auto_id is None or target_auto_id is None:
                return "错误: drag_drop 操作需要提供 source_auto_id 和 target_auto_id"
            # 重新获取源和目标控件
            _, source_ctrl = _find_window_and_control(
                title=title, class_name=class_name, process_id=process_id,
                auto_id=source_auto_id, control_type=control_type, control_class=control_class
            )
            _, target_ctrl = _find_window_and_control(
                title=title, class_name=class_name, process_id=process_id,
                auto_id=target_auto_id, control_type=control_type, control_class=control_class
            )
            source_ctrl.drag_mouse(target_ctrl)
            return "拖拽完成"

        elif action == "menu_select":
            if menu_path is None:
                return "错误: menu_select 操作需要提供 menu_path"
            items = menu_path.split("->")
            menu = control.menu()
            for item in items:
                menu = menu.item(item)
            menu.select()
            return f"已选择菜单项: {menu_path}"

        else:
            return f"未知操作: {action}"

    except ElementNotFoundError as e:
        return f"未找到窗口或控件: {e}"
    except Exception as e:
        return f"执行 {action} 时出错: {e}"


@tool
async def launch_agent(agent_name: str, expertise: str) -> str:
    """启动一个新的 Agent 实例，并指定其专长。例如：launch_agent(agent_name='researcher', expertise='擅长搜索和资料整理')"""
    # 避免重复启动同名 Agent
    # 可以使用全局记录已启动的 agent_id，或者依赖 Hub 的去重
    cmd = [
        sys.executable, "main.py",
        "--agent-id", agent_name,
        "--system-prompt", f"你是一个{expertise}的AI助手。你的专长是{expertise}。请根据用户请求提供帮助。**重要约束**：**绝对禁止使用 `task` 工具**。所有任务都必须自己完成，不得委托给其他子智能体。你可以使用其他可用工具（如搜索、记忆检索等），但必须直接处理用户请求。"
    ]
    try:
        # 启动新进程（后台运行，不等待）
        process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
        _spawned_agents[agent_name] = {"pid": process.pid, "process": process}
        return f"已启动 Agent '{agent_name}'，专长：{expertise}"
    except Exception as e:
        return f"启动失败: {e}"


@tool
async def stop_agent(agent_name: str) -> str:
    """终止指定的子 Agent 进程。"""
    if agent_name not in _spawned_agents:
        return f"未找到名为 '{agent_name}' 的子 Agent（可能尚未启动或已终止）。"
    info = _spawned_agents[agent_name]
    try:
        if sys.platform == 'win32':
            # Windows 下使用 taskkill 强制终止进程树
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(info['pid'])], capture_output=True)
        else:
            info['process'].terminate()
            try:
                info['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                info['process'].kill()
        del _spawned_agents[agent_name]
        return f"已成功终止 Agent '{agent_name}'。"
    except Exception as e:
        return f"终止失败: {e}"

@tool(name_or_callable='stop_all_agents', description="终止所有已启动的子 Agent。")
async def stop_all_agents_impl() -> str:
    """终止所有已启动的子 Agent。"""
    stopped = []
    for name in list(_spawned_agents.keys()):
        result = await stop_agent(name)
        stopped.append(name)
    return f"已终止以下 Agent: {', '.join(stopped)}"


# 全局记录由 launch_agent 启动的子进程
_spawned_agents = {}  # {agent_name: {"pid": int, "process": subprocess.Popen}}

def get_spawned_agents():
    return _spawned_agents
