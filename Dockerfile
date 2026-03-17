# 使用 Playwright 官方 Python 镜像作为基础（已包含 Playwright 和 Chromium）
FROM mcr.microsoft.com/playwright/python:v1.51.0-noble

# 安装 uv 包管理器（以 root 身份）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 复制项目依赖清单
COPY pyproject.toml .

# 生成 requirements.txt 并安装所有 Python 依赖（系统级）
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --system -r requirements.txt

# 切换到基础镜像的默认用户（pwuser）
USER pwuser

# 安装 playwright 包（用户级，确保 Python 能找到）
ENV PATH="/home/pwuser/.local/bin:${PATH}"
RUN pip install --user playwright

# 确保 Playwright 浏览器已就绪（镜像已预装，此命令仅验证）
RUN playwright install chromium

# 设置最终工作目录
WORKDIR /workspace