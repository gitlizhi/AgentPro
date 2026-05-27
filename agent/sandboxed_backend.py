"""docker容器沙箱环境"""
import tempfile
import os
import uuid
import time
import logging
from datetime import datetime, timezone
import docker
from typing import Optional, List, Tuple
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.protocol import ExecuteResponse

logger = logging.getLogger(__name__)

# 容器标签：用于标识本项目创建的容器，方便启动时清理孤儿容器
PROJECT_LABEL = "agentpro.sandbox"
PROJECT_LABEL_VALUE = "true"
# 运行中容器的最大允许存活时间（秒），超时的视为孤儿
MAX_CONTAINER_AGE_SECONDS = 1800  # 30 分钟


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
                pass

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
                pass
    except Exception as e:
        logger.warning(f"孤儿容器清理失败: {e}")


class DockerSandboxBackend(BaseSandbox):
    """
    安全加固的 Docker 沙箱后端。

    容器生命周期规则：
    - 命令正常完成 → 立即删除
    - 命令超时 → 强制终止并删除
    - 任何异常 → finally 块保证删除
    - 启动时 → 清理所有本项目遗留的孤儿容器
    """

    def __init__(self,
                 # image: str = "my-agent-base:latest",       # 本地已经构建好镜像了
                 image: str = "python:3.12-slim",               # 通用镜像
                 mem_limit: str = "256m",
                 cpu_limit: float = 0.5,
                 network_disabled: bool = True,
                 desktop_path: Optional[str] = None,
                 skills_host_path: Optional[str] = None,
                 user: str = "nobody",          # 以非 root 用户运行
                 read_only_rootfs: bool = False, # 根文件系统只读
                 env: Optional[dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        try:
            self.docker_client = docker.from_env()
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

        # 启动时清理本项目遗留的孤儿容器
        _cleanup_orphan_containers(self.docker_client, self.image)

    @property
    def id(self) -> str:
        return self._id

    def execute(self, command: str, *, timeout: Optional[int] = None, env: Optional[dict] = None) -> ExecuteResponse:
        container = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                volumes = {tmpdir: {"bind": "/workspace", "mode": "rw"}}
                if self.desktop_path:
                    volumes[self.desktop_path] = {"bind": "/desktop", "mode": "rw"}
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
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges:true"],
                    pids_limit=100,
                    labels={PROJECT_LABEL: PROJECT_LABEL_VALUE},
                )

                wait_timeout = timeout if timeout is not None else 30
                try:
                    result = container.wait(timeout=wait_timeout)
                except (docker.errors.APIError, Exception):
                    # 超时或异常 → 强制终止
                    logger.warning(f"容器 {container.short_id} 执行超时/异常，强制终止")
                    try:
                        container.kill()
                    except Exception:
                        pass
                    return ExecuteResponse(
                        output=f"Command timed out after {wait_timeout}s",
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
                    pass

    def upload_files(self, files: List[Tuple[str, bytes]]):
        pass
    
    def download_files(self, paths: List[str]) -> List[object]:
        """从容器中下载多个文件，返回列表，每个元素应包含 path、content、error 属性"""
        from types import SimpleNamespace
        results = []
        volumes = {}
        if self.skills_host_path:
            volumes[self.skills_host_path] = {"bind": "/agent/skills", "mode": "ro"}

        for path in paths:
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
                results.append(res)
            except Exception as e:
                res = SimpleNamespace()
                res.path = path
                res.content = None
                res.error = str(e)
                results.append(res)
            finally:
                if container:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
        return results

    def close(self):
        pass