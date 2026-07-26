"""Git 版本控制工具集。

提供 git_status / git_diff / git_log / git_add / git_commit 五个工具，
让 Agent 可以查看仓库状态、浏览变更、暂存文件和创建提交。
"""

import os
import subprocess
from langchain.tools import tool

_GIT_TIMEOUT = 30

# git 安全协议禁止的操作
_FORBIDDEN_ARGS = frozenset({
    "--no-verify", "--no-gpg-sign", "--no-sign",
})


def _run_git(args: list[str], cwd: str | None = None) -> tuple[str, str, int]:
    """执行 git 命令，返回 (stdout, stderr, returncode)。"""
    cwd = cwd or os.getcwd()
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT,
        )
        return (result.stdout or "").strip(), (result.stderr or "").strip(), result.returncode
    except FileNotFoundError:
        return "", "git 命令未找到，请确认已安装 Git", 1
    except subprocess.TimeoutExpired:
        return "", f"git 命令超时（{_GIT_TIMEOUT}s）", 1


def _sanitize_message(msg: str) -> str:
    """防止 commit message 注入。截断首行，移除危险字符。"""
    first_line = msg.split("\n")[0].strip()
    # 限制长度，移除 shell 元字符
    return first_line[:200]


@tool
def git_status() -> str:
    """查看工作区状态：已修改、已暂存、未跟踪的文件列表。"""
    stdout, stderr, rc = _run_git(["status"])
    if rc != 0:
        return f"Error: {stderr or 'git 命令执行失败'}"
    return stdout or "工作区干净，无变更"


@tool
def git_diff(staged: bool = False) -> str:
    """查看工作区或暂存区的代码变更（diff）。

    Args:
        staged: 设为 True 查看已暂存（git add 后）的变更，默认 False 查看未暂存的变更。
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    stdout, stderr, rc = _run_git(args)
    if rc != 0:
        return f"Error: {stderr or 'git 命令执行失败'}"
    return stdout or "无差异"


@tool
def git_log(max_count: int = 10) -> str:
    """查看最近的提交历史。用于了解仓库的提交风格。

    Args:
        max_count: 最多显示条数，默认 10。
    """
    stdout, stderr, rc = _run_git(
        ["log", "--oneline", "--decorate", "-n", str(max_count)]
    )
    if rc != 0:
        return f"Error: {stderr or 'git 命令执行失败'}"
    return stdout or "尚无提交"


@tool
def git_add(files: str) -> str:
    """将指定文件添加到暂存区（git add）。

    Args:
        files: 要暂存的文件路径，多个文件用空格分隔。如 "agent/git_tools.py agent/brain.py"。
               **注意：绝不使用 "."、"./*" 或 "-A" 等通配符，只能添加明确指定的文件。**
    """
    file_list = files.split()
    if not file_list:
        return "Error: 未指定要暂存的文件"

    # 安全拦截：禁止 add -A / . / *
    for f in file_list:
        if f in (".", ".", "-A", "--all", "*", ".:", "./*"):
            return (
                f"Error: 禁止使用 '{f}' 暂存全部文件。"
                "请明确指定要暂存的单个文件路径。"
            )

    stdout, stderr, rc = _run_git(["add"] + file_list)
    if rc != 0:
        return f"Error: {stderr or 'git add 失败'}"
    return stdout or f"已暂存: {', '.join(file_list)}"


@tool
def git_commit(message: str) -> str:
    """创建提交（git commit）。

    **安全协议（不可绕过）：**
    - 提交前必须先 git_add 明确指定文件
    - 不跳过 hooks（不使用 --no-verify）
    - 不修改已发布的提交（不使用 --amend）

    Args:
        message: 提交信息（单行，简洁描述变更内容和原因）。
    """
    msg = _sanitize_message(message)
    if not msg:
        return "Error: 提交信息不能为空"

    # 检查是否有暂存内容
    stdout, stderr, rc = _run_git(["diff", "--cached", "--stat"])
    if rc != 0:
        return f"Error: {stderr or 'git 命令执行失败'}"
    if not stdout:
        return "没有暂存的变更。请先用 git_add 将文件加入暂存区。"

    stdout, stderr, rc = _run_git(["commit", "-m", msg])
    if rc != 0:
        return f"Error: {stderr or 'git commit 失败'}"
    return stdout or "提交成功"
