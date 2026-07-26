"""docker容器沙箱环境"""
import os
import re
import shutil
import uuid
import time
import logging
from datetime import datetime, timezone
import docker
from typing import Optional, List, Tuple
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileInfo,
    GrepMatch,
    WriteResult,
)

logger = logging.getLogger(__name__)

# 容器标签：用于标识本项目创建的容器，方便启动时清理孤儿容器
PROJECT_LABEL = "agentpro.sandbox"
PROJECT_LABEL_VALUE = "true"
# 运行中容器的最大允许存活时间（秒），超时的视为孤儿
MAX_CONTAINER_AGE_SECONDS = 1800  # 30 分钟

# 沙箱持久化工作区根目录（相对于 agent 目录）
_SANDBOX_WORKSPACES_DIR = os.path.join(os.path.dirname(__file__), "agent_temp", "sandbox_workspaces")
# 共享工作区：所有 Agent 的 Docker 容器共用此目录，解决跨 Agent 文件读取问题
_SHARED_WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "agent_temp", "shared_workspace")


def _cleanup_orphan_containers(docker_client, image: str):
    """清理之前运行遗留的孤儿容器，不会误删其他智能体正在使用的容器。"""
    try:
        base_filters = {"label": f"{PROJECT_LABEL}={PROJECT_LABEL_VALUE}"}

        # 1. 清理已退出的容器（总是安全的）
        exited_filters = {**base_filters, "status": "exited"}
        exited = docker_client.containers.list(all=True, filters=exited_filters)
        for c in exited:
            try:
                c.remove()
                logger.info(f"已清理退出容器: {c.short_id}")
            except Exception:
                logger.debug(f"清理退出容器失败: {c.short_id}", exc_info=True)

        # 2. 对于运行中的容器，仅清理超长时间运行的（>30 分钟），
        #    避免误删其他智能体正在使用的容器。
        running_filters = {**base_filters, "status": "running"}
        running = docker_client.containers.list(all=True, filters=running_filters)
        now = time.time()
        for c in running:
            try:
                created = c.attrs.get("Created", "")
                if created:
                    # Docker 返回的 Created 格式: "2024-01-01T00:00:00.000000000Z"
                    created_dt = datetime.fromisoformat(
                        created.replace("Z", "+00:00")
                    )
                    age = now - created_dt.timestamp()
                    if age > MAX_CONTAINER_AGE_SECONDS:
                        logger.warning(
                            f"容器 {c.short_id} 已运行 {age:.0f}s，超过限制，视为孤儿，强制清理"
                        )
                        c.kill()
                        c.remove()
                    else:
                        logger.debug(
                            f"容器 {c.short_id} 运行中（{age:.0f}s），可能是其他智能体在使用，跳过"
                        )
            except Exception:
                logger.debug(f"检查运行中容器 {c.short_id} 失败", exc_info=True)
    except Exception as e:
        logger.warning(f"孤儿容器清理失败: {e}")


class DockerSandboxBackend(BaseSandbox):
    """
    安全加固的 Docker 沙箱后端。

    容器生命周期规则：
    - 命令正常完成 → 立即删除容器
    - 命令超时 → 强制终止并删除容器
    - 任何异常 → finally 块保证删除容器

    工作区持久化：
    - /workspace 和 /home/pwuser 挂载自宿主机持久化目录
    - pip install --user、文件下载等在多次 execute() 之间保留
    - 每个 Agent 独立工作区，互不干扰
    - 仅容器本身是临时的（每次执行后销毁重建）
    """

    def __init__(self,
                 # image: str = "my-agent-base:latest",       # 本地已经构建好镜像了
                 image: str = "python:3.12-slim",               # 通用镜像
                 mem_limit: str = "256m",
                 cpu_limit: float = 0.5,
                 network_disabled: bool = True,
                 desktop_path: Optional[str] = None,
                 skills_host_path: Optional[str] = None,
                 user: str = "root",            # 以 root 用户运行，避免权限问题
                 read_only_rootfs: bool = False, # 根文件系统只读
                 env: Optional[dict] = None,
                 agent_id: str = "default",     # 智能体 ID，用于隔离持久化工作区
                 **kwargs):
        super().__init__(**kwargs)
        try:
            self.docker_client = docker.from_env(timeout=600)  # HTTP 超时 10 分钟，支持 pip install 等长操作
            self.docker_client.ping()
        except Exception as e:
            raise RuntimeError(f"无法连接到 Docker 守护进程，请确保 Docker Desktop 已启动。错误详情: {e}")

        self.image = image
        self.default_env = env or {}
        self.desktop_path = desktop_path
        self.skills_host_path = skills_host_path
        self.mounts = {}
        if desktop_path:
            self.mounts[desktop_path] = "/desktop"
        self.mem_limit = mem_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
        self.user = user
        self.read_only_rootfs = read_only_rootfs
        self._id = str(uuid.uuid4())
        self._agent_id = agent_id

        # 持久化工作区目录（每个 Agent 独立的 home，但共享 workspace）
        self._workspace_dir = _SHARED_WORKSPACE_DIR  # 所有 Agent 共用此目录
        self._home_dir = os.path.join(_SANDBOX_WORKSPACES_DIR, agent_id, "home")
        os.makedirs(self._workspace_dir, exist_ok=True)
        os.makedirs(self._home_dir, exist_ok=True)

        # 首次使用时初始化 home 目录（确保 shell 配置文件存在）
        self._init_home_dir()

        # 启动时清理本项目遗留的孤儿容器
        _cleanup_orphan_containers(self.docker_client, self.image)

    def _init_home_dir(self):
        """确保 home 目录包含最基础的 shell 配置文件，避免 pip/python 报错。"""
        profile = os.path.join(self._home_dir, ".profile")
        bashrc = os.path.join(self._home_dir, ".bashrc")
        if not os.path.exists(profile):
            try:
                with open(profile, "w") as f:
                    f.write("# AgentPro sandbox persistent home\n")
                    f.write('export PATH="$HOME/.local/bin:$PATH"\n')
                logger.info(f"初始化沙箱 home 目录: {self._home_dir}")
            except OSError:
                pass
        if not os.path.exists(bashrc):
            try:
                with open(bashrc, "w") as f:
                    f.write("# AgentPro sandbox bashrc\n")
            except OSError:
                pass

    @property
    def id(self) -> str:
        return self._id

    # ---- 危险文件系统搜索拦截 ----
    _FIND_FROM_ROOT = re.compile(r'\bfind\s+/(?=\s|$)')
    _GLOB_UNSCOPED = re.compile(r"(^|\s)(glob\s+)(['\"])((?!/)(?:\*\*|[*?][^/\s]))")

    def _sanitize_command(self, command: str) -> tuple:
        """检测并修正危险的文件系统搜索命令，避免容器扫描全文件系统超时。
        返回 (修正后的命令, 修正说明列表)。
        """
        fixes = []

        # 1. find / → 限定至 /workspace/ + maxdepth 兜底
        if self._FIND_FROM_ROOT.search(command) and '-maxdepth' not in command:
            command = self._FIND_FROM_ROOT.sub('find /workspace/', command)
            if ' -exec ' not in command:
                command = command.rstrip() + ' -maxdepth 6'
            fixes.append('find / → /workspace/')

        # 2. glob 无 /workspace 前缀 → 自动添加
        if self._GLOB_UNSCOPED.search(command):
            def _rewrite(m):
                return f"{m.group(1) or ''}{m.group(2)}{m.group(3)}/workspace/{m.group(4)}"
            command = self._GLOB_UNSCOPED.sub(_rewrite, command)
            fixes.append('glob → /workspace/ 限定')

        return command, fixes

    def execute(self, command: str, *, timeout: Optional[int] = None, env: Optional[dict] = None) -> ExecuteResponse:
        container = None
        sanitized, fixes = self._sanitize_command(command)
        if fixes:
            logger.info(f"命令已自动修正 ({', '.join(fixes)}): {command[:100]} → {sanitized[:100]}")
            command = sanitized
        try:
            # 使用持久化工作区（而非临时目录），确保 pip 安装、文件下载等在多次 execute 之间保留
            volumes = {
                self._workspace_dir: {"bind": "/workspace", "mode": "rw"},
                self._home_dir: {"bind": "/home/pwuser", "mode": "rw"},
            }
            if self.desktop_path:
                volumes[self.desktop_path] = {"bind": "/desktop", "mode": "ro"}
            if self.skills_host_path:
                volumes[self.skills_host_path] = {"bind": "/agent/skills", "mode": "ro"}
            conversation_host_path = os.path.join(os.getcwd(), "conversation_history")
            os.makedirs(conversation_host_path, exist_ok=True)
            volumes[conversation_host_path] = {"bind": "/conversation_history", "mode": "rw"}

            environment = {**self.default_env, **(env or {})}
            environment.setdefault("HOME", "/home/pwuser")
            cmd = ["/bin/sh", "-c", command]

            container = self.docker_client.containers.run(
                image=self.image,
                command=cmd,
                working_dir="/workspace",
                mem_limit=self.mem_limit,
                nano_cpus=int(self.cpu_limit * 1e9) if self.cpu_limit else None,
                network_disabled=self.network_disabled,
                detach=True,
                remove=False,
                environment=environment,
                volumes=volumes,
                user=self.user,
                read_only=self.read_only_rootfs,
                pids_limit=200,
                labels={PROJECT_LABEL: PROJECT_LABEL_VALUE},
            )

            wait_timeout = timeout if timeout is not None else 60
            try:
                result = container.wait(timeout=wait_timeout)
            except Exception as exc:
                # 区分超时 / Docker API 错误 / 其他异常，便于排查
                is_timeout = isinstance(exc, (TimeoutError,))
                # requests 库的超时异常类名通常包含 "Timeout"
                exc_name = type(exc).__name__
                if not is_timeout and "Timeout" in exc_name:
                    is_timeout = True

                if is_timeout:
                    logger.warning(
                        f"容器 {container.short_id} 执行超时（>{wait_timeout}s），"
                        f"命令可能过于复杂或死循环，强制终止"
                    )
                elif isinstance(exc, docker.errors.APIError):
                    logger.warning(
                        f"容器 {container.short_id} Docker API 异常: {exc}，强制终止"
                    )
                else:
                    logger.warning(
                        f"容器 {container.short_id} 执行异常 ({exc_name}: {exc})，强制终止"
                    )

                try:
                    container.kill()
                except Exception:
                    logger.debug(f"强制终止容器失败: {container.short_id}", exc_info=True)
                return ExecuteResponse(
                    output=f"Command {'timed out' if is_timeout else 'failed'}: {exc}",
                    exit_code=-1,
                    truncated=False,
                )

            stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")
            return ExecuteResponse(
                output=stdout + stderr,
                exit_code=result['StatusCode'],
                truncated=False,
            )

        except docker.errors.APIError as e:
            return ExecuteResponse(output=f"Docker API error: {e}", exit_code=-1, truncated=False)
        except Exception as e:
            return ExecuteResponse(output=f"Unexpected error: {e}", exit_code=-1, truncated=False)
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    logger.debug(f"清理容器失败: {container.short_id}", exc_info=True)

    def upload_files(self, files: List[Tuple[str, bytes]]):
        """将文件直接写入持久化工作区（绕过容器，在宿主机上完成）。"""
        for path, content in files:
            # 路径相对于 /workspace
            rel_path = path.lstrip("/")
            dest = os.path.join(self._workspace_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(content)

    def download_files(self, paths: List[str]) -> List[object]:
        """从持久化工作区（宿主机）直接读取文件，无需启动容器。"""
        from types import SimpleNamespace
        results = []
        for path in paths:
            rel_path = path.lstrip("/")
            src = os.path.join(self._workspace_dir, rel_path)
            res = SimpleNamespace()
            res.path = path
            try:
                with open(src, "rb") as f:
                    res.content = f.read()
                res.error = None
            except FileNotFoundError:
                # 回退到容器内读取（处理 /agent/skills 等非 workspace 路径）
                res = self._download_from_container(path)
            except Exception as e:
                res.content = None
                res.error = str(e)
            results.append(res)
        return results

    def _download_from_container(self, path: str):
        """回退方案：从容器中读取文件（用于非 workspace 路径）。"""
        from types import SimpleNamespace
        volumes = {
            self._workspace_dir: {"bind": "/workspace", "mode": "rw"},
        }
        if self.skills_host_path:
            volumes[self.skills_host_path] = {"bind": "/agent/skills", "mode": "ro"}

        container = None
        try:
            container = self.docker_client.containers.run(
                image=self.image,
                command=["cat", path],
                working_dir="/workspace",
                volumes=volumes,
                network_disabled=True,
                detach=True,
                remove=False,
                user="root",
                mem_limit="128m",
                nano_cpus=int(0.5 * 1e9),
                labels={PROJECT_LABEL: PROJECT_LABEL_VALUE},
            )
            result = container.wait(timeout=10)
            stdout = container.logs(stdout=True, stderr=False)
            stderr = container.logs(stdout=False, stderr=True)

            res = SimpleNamespace()
            res.path = path
            if result['StatusCode'] == 0:
                res.content = stdout
                res.error = None
            else:
                res.content = None
                res.error = stderr.decode(errors="replace")
            return res
        except Exception as e:
            res = SimpleNamespace()
            res.path = path
            res.content = None
            res.error = str(e)
            return res
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    logger.debug(f"清理容器失败: {container.short_id}", exc_info=True)

    def clean_workspace(self):
        """清空持久化工作区（保留目录结构，删除所有内容）。"""
        for d in (self._workspace_dir, self._home_dir):
            if os.path.exists(d):
                for item in os.listdir(d):
                    item_path = os.path.join(d, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                        else:
                            os.remove(item_path)
                    except OSError as e:
                        logger.warning(f"清理工作区文件失败: {item_path}: {e}")
        # 重新初始化 home 目录
        self._init_home_dir()

    def close(self):
        pass


class HybridBackend(DockerSandboxBackend):
    """混合后端：文件 I/O 直连宿主机，代码执行仍走 Docker 沙箱。

    重写 BaseSandbox 中全部文件操作（read / write / edit / ls / glob / grep），
    直接在宿主机文件系统上执行，完全跳过 Docker 容器。
    仅 execute() 保留在 Docker 沙箱中运行。
    """

    # ---- 路径映射 ----

    def _host_path(self, sandbox_path: str) -> str:
        """将沙箱虚拟路径映射为宿主机真实路径。"""
        # 已知挂载点 → 宿主机路径
        if sandbox_path.startswith('/workspace/'):
            return os.path.join(self._workspace_dir, sandbox_path[len('/workspace/'):])
        if sandbox_path == '/workspace':
            return self._workspace_dir
        if sandbox_path.startswith('/home/pwuser/'):
            return os.path.join(self._home_dir, sandbox_path[len('/home/pwuser/'):])
        if sandbox_path == '/home/pwuser':
            return self._home_dir
        if self.desktop_path:
            if sandbox_path.startswith('/desktop/'):
                return os.path.join(self.desktop_path, sandbox_path[len('/desktop/'):])
            if sandbox_path == '/desktop':
                return self.desktop_path
        if self.skills_host_path:
            if sandbox_path.startswith('/agent/skills/'):
                return os.path.join(self.skills_host_path, sandbox_path[len('/agent/skills/'):])
            if sandbox_path == '/agent/skills':
                return self.skills_host_path
        # 默认：相对项目根目录解析
        rel = sandbox_path.lstrip('/')
        return os.path.abspath(rel)

    def _sandbox_path(self, host_path: str) -> str:
        """将宿主机路径反转为沙箱虚拟路径（用于 grep/glob 返回值）。"""
        host = os.path.abspath(host_path).replace('\\', '/')
        ws = os.path.abspath(self._workspace_dir).replace('\\', '/')
        home = os.path.abspath(self._home_dir).replace('\\', '/')
        cwd = os.getcwd().replace('\\', '/')

        if host.startswith(ws + '/'):
            return '/workspace/' + host[len(ws) + 1:]
        if host == ws:
            return '/workspace'
        if host.startswith(home + '/'):
            return '/home/pwuser/' + host[len(home) + 1:]
        if host == home:
            return '/home/pwuser'
        if self.desktop_path:
            dtop = os.path.abspath(self.desktop_path).replace('\\', '/')
            if host.startswith(dtop + '/'):
                return '/desktop/' + host[len(dtop) + 1:]
            if host == dtop:
                return '/desktop'
        if self.skills_host_path:
            skills = os.path.abspath(self.skills_host_path).replace('\\', '/')
            if host.startswith(skills + '/'):
                return '/agent/skills/' + host[len(skills) + 1:]
            if host == skills:
                return '/agent/skills'
        if host.startswith(cwd + '/'):
            return '/' + host[len(cwd) + 1:]
        if host == cwd:
            return '/'
        # 兜底
        return '/' + os.path.relpath(host, cwd).replace('\\', '/')

    # ---- 文件操作（直连宿主机） ----

    # 只读挂载点：这些路径允许 read/ls/glob/grep 但禁止 write/edit
    _READONLY_MOUNTS = frozenset({"/desktop", "/agent/skills"})

    def _check_readonly_path(self, sandbox_path: str) -> str | None:
        """检查路径是否在只读挂载点下，是则返回错误信息，否则返回 None。"""
        for mount in self._READONLY_MOUNTS:
            if sandbox_path == mount or sandbox_path.startswith(mount + "/"):
                return (
                    f"Error: '{mount}' 是只读挂载点，不允许修改文件。"
                    f"如需在桌面创建文件，请先写入 /workspace/ 再通过 computer_paste 等工具处理。"
                )
        return None

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        host_path = self._host_path(file_path)
        try:
            with open(host_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except PermissionError:
            return f"Error: Permission denied reading '{file_path}'"
        except IsADirectoryError:
            return f"Error: '{file_path}' is a directory, not a file"
        except OSError as e:
            return f"Error reading '{file_path}': {e}"

        if not lines:
            return "System reminder: File exists but has empty contents"

        selected = lines[offset:offset + limit]
        content = ''.join(selected)
        if content.endswith('\n'):
            content = content[:-1]
        from deepagents.backends.utils import format_content_with_line_numbers
        return format_content_with_line_numbers(content, start_line=offset + 1)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        err = self._check_readonly_path(file_path)
        if err:
            return WriteResult(error=err)
        host_path = self._host_path(file_path)
        if os.path.exists(host_path):
            return WriteResult(error=f"Error: File '{file_path}' already exists")
        try:
            parent = os.path.dirname(host_path) or '.'
            os.makedirs(parent, exist_ok=True)
            with open(host_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return WriteResult(path=file_path, files_update=None)
        except PermissionError:
            return WriteResult(error=f"Error: Permission denied writing '{file_path}'")
        except OSError as e:
            return WriteResult(error=f"Error writing '{file_path}': {e}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        err = self._check_readonly_path(file_path)
        if err:
            return EditResult(error=err)
        from deepagents.backends.utils import perform_string_replacement
        host_path = self._host_path(file_path)
        if not os.path.isfile(host_path):
            return EditResult(error=f"Error: File '{file_path}' not found")
        try:
            with open(host_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except PermissionError:
            return EditResult(error=f"Error: Permission denied reading '{file_path}'")
        except OSError as e:
            return EditResult(error=f"Error reading '{file_path}': {e}")

        result = perform_string_replacement(text, old_string, new_string, replace_all)
        if isinstance(result, str):
            # 错误信息
            return EditResult(error=result)

        new_text, occurrences = result
        try:
            with open(host_path, 'w', encoding='utf-8') as f:
                f.write(new_text)
            return EditResult(path=file_path, files_update=None, occurrences=occurrences)
        except PermissionError:
            return EditResult(error=f"Error: Permission denied writing '{file_path}'")
        except OSError as e:
            return EditResult(error=f"Error writing '{file_path}': {e}")

    def ls_info(self, path: str) -> list[FileInfo]:
        host_path = self._host_path(path)
        try:
            entries = list(os.scandir(host_path))
        except FileNotFoundError:
            return []
        except PermissionError:
            return []

        infos: list[FileInfo] = []
        for entry in sorted(entries, key=lambda e: (not e.is_dir(), e.name.lower())):
            infos.append({
                "path": self._sandbox_path(entry.path),
                "is_dir": entry.is_dir(follow_symlinks=False),
            })
        return infos

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        import glob as glob_mod
        host_path = self._host_path(path)
        cwd_before = os.getcwd()
        try:
            os.chdir(host_path)
            matches = sorted(glob_mod.glob(pattern, recursive=True))
            infos: list[FileInfo] = []
            for m in matches:
                full = os.path.join(host_path, m)
                try:
                    st = os.stat(full)
                    is_dir = os.path.isdir(full)
                except OSError:
                    continue
                # 路径还原为沙箱格式
                sandbox_full = self._sandbox_path(full)
                infos.append({
                    "path": sandbox_full,
                    "is_dir": is_dir,
                    "size": st.st_size,
                    "modified_at": datetime.fromtimestamp(
                        st.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })
            return infos
        except FileNotFoundError:
            return []
        except PermissionError:
            return []
        finally:
            try:
                os.chdir(cwd_before)
            except OSError:
                pass

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        import fnmatch
        search_path = self._host_path(path or '/')
        if not os.path.isdir(search_path):
            return f"Error: '{path or '/'}' is not a directory"

        matches: list[GrepMatch] = []
        # 跳过的目录
        _SKIP_DIRS = frozenset({'.git', '__pycache__', 'node_modules', '.venv',
                                 'venv', '.tox', '.eggs', '.mypy_cache', '.pytest_cache',
                                 'chroma_db', 'pgdata', 'browser_data', 'screenshots',
                                 'agent_temp', 'chat_images', '.claude'})

        for root, dirs, files in os.walk(search_path):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith('.')]
            for fname in files:
                if glob and not fnmatch.fnmatch(fname, glob):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                        for i, line in enumerate(f, 1):
                            if pattern in line:
                                sandbox = self._sandbox_path(fpath)
                                matches.append({
                                    "path": sandbox,
                                    "line": i,
                                    "text": line.rstrip('\n').rstrip('\r'),
                                })
                except (PermissionError, OSError, UnicodeDecodeError):
                    continue
        return matches