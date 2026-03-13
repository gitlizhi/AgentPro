"""docker容器沙箱环境"""
import tempfile
import uuid
import docker
from typing import Optional, List, Tuple
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.protocol import ExecuteResponse

class DockerSandboxBackend(BaseSandbox):
    """
    安全加固的 Docker 沙箱后端，遵循最小权限原则。
    """
    def __init__(self,
                 image: str = "python:3.12-slim",
                 mem_limit: str = "256m",
                 cpu_limit: float = 0.5,
                 network_disabled: bool = True,
                 desktop_path: Optional[str] = None,
                 user: str = "nobody",          # 以非 root 用户运行
                 read_only_rootfs: bool = True, # 根文件系统只读
                 **kwargs):
        super().__init__(**kwargs)
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
        except Exception as e:
            raise RuntimeError(f"无法连接到 Docker 守护进程，请确保 Docker Desktop 已启动。错误详情: {e}")

        self.desktop_path = desktop_path
        self.mounts = {}
        if desktop_path:
            # 建议挂载为只读，除非确实需要写入
            self.mounts[desktop_path] = "/desktop"

        self.image = image
        self.mem_limit = mem_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
        self.user = user
        self.read_only_rootfs = read_only_rootfs
        self._id = str(uuid.uuid4())

    @property
    def id(self) -> str:
        return self._id

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        with tempfile.TemporaryDirectory() as tmpdir:
            volumes = {tmpdir: {"bind": "/workspace", "mode": "rw"}}
            if self.desktop_path:
                # 桌面目录挂载为读写（可根据需求改为 ro）
                volumes[self.desktop_path] = {"bind": "/desktop", "mode": "rw"}

            cmd = ["/bin/sh", "-c", command]
            try:
                container = self.docker_client.containers.run(
                    image=self.image,
                    command=cmd,
                    working_dir="/workspace",
                    mem_limit=self.mem_limit,
                    nano_cpus=int(self.cpu_limit * 1e9) if self.cpu_limit else None,
                    network_disabled=self.network_disabled,
                    detach=True,
                    remove=False,
                    volumes=volumes,
                    user=self.user,                      # 非 root 用户
                    read_only=self.read_only_rootfs,     # 根文件系统只读
                    cap_drop=["ALL"],                     # 删除所有能力
                    security_opt=["no-new-privileges:true"],  # 禁止提权
                    # 可选：限制进程数
                    pids_limit=100,
                )

                wait_timeout = timeout if timeout is not None else 30
                result = container.wait(timeout=wait_timeout)
                stdout = container.logs(stdout=True, stderr=False).decode()
                stderr = container.logs(stdout=False, stderr=True).decode()
                container.remove()

                output = stdout + stderr
                return ExecuteResponse(
                    output=output,
                    exit_code=result['StatusCode'],
                    truncated=False
                )

            except docker.errors.ContainerError as e:
                return ExecuteResponse(output=str(e), exit_code=-1, truncated=False)
            except docker.errors.APIError as e:
                if "Timeout" in str(e):
                    return ExecuteResponse(output="Command timed out", exit_code=-1, truncated=False)
                else:
                    return ExecuteResponse(output=f"Docker API error: {e}", exit_code=-1, truncated=False)
            except Exception as e:
                return ExecuteResponse(output=f"Unexpected error: {e}", exit_code=-1, truncated=False)

    def upload_files(self, files: List[Tuple[str, bytes]]):
        pass

    def download_files(self, paths: List[str]) -> List[dict]:
        return []

    def close(self):
        pass