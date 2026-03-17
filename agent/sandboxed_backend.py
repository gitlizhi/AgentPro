"""docker容器沙箱环境"""
import tempfile
import os
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
                 image: str = "my-agent-base:latest",       # 本地已经构建好镜像了
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
        self.default_env = env or {}
        self.desktop_path = desktop_path
        self.skills_host_path = skills_host_path
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

    def execute(self, command: str, *, timeout: Optional[int] = None, env: Optional[dict] = None) -> ExecuteResponse:
        with tempfile.TemporaryDirectory() as tmpdir:
            volumes = {tmpdir: {"bind": "/workspace", "mode": "rw"}}
            if self.desktop_path:
                # 桌面目录挂载为读写（可根据需求改为 ro）
                volumes[self.desktop_path] = {"bind": "/desktop", "mode": "rw"}
            if self.skills_host_path:
                volumes[self.skills_host_path] = {"bind": "/agent/skills", "mode": "ro"}
            # 新增 conversation_history 挂载
            conversation_host_path = os.path.join(os.getcwd(), "conversation_history")
            os.makedirs(conversation_host_path, exist_ok=True)
            volumes[conversation_host_path] = {"bind": "/conversation_history", "mode": "rw"}
            
            # 合并环境变量（容器默认环境 + 传入的 env）
            environment = {**self.default_env, **(env or {})}
            environment.setdefault("HOME", "/home/pwuser")  # Playwright 镜像用户的家目录
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
                    environment=environment,                # 新增
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
    
    def download_files(self, paths: List[str]) -> List[object]:
        """从容器中下载多个文件，返回列表，每个元素应包含 path、content、error 属性"""
        from types import SimpleNamespace
        results = []
        for path in paths:
            try:
                # 构造容器执行 cat 命令
                volumes = {}
                if self.skills_host_path:
                    volumes[self.skills_host_path] = {"bind": "/agent/skills", "mode": "ro"}
                
                container = self.docker_client.containers.run(
                    image=self.image,
                    command=["cat", path],
                    working_dir="/workspace",
                    volumes=volumes,
                    network_disabled=True,
                    detach=True,
                    remove=False,
                    user="root",  # 临时用 root 确保权限
                    mem_limit="128m",
                    nano_cpus=int(0.5 * 1e9),
                )
                result = container.wait(timeout=10)
                stdout = container.logs(stdout=True, stderr=False)
                stderr = container.logs(stdout=False, stderr=True)
                container.remove()
                
                res = SimpleNamespace()
                res.path = path
                if result['StatusCode'] == 0:
                    res.content = stdout
                    res.error = None
                else:
                    res.content = None
                    res.error = stderr.decode()
                results.append(res)
            except Exception as e:
                res = SimpleNamespace()
                res.path = path
                res.content = None
                res.error = str(e)
                results.append(res)
        return results

    def close(self):
        pass