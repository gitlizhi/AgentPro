@echo off
chcp 65001 >nul
title AgentPro 启动器
echo ========================================
echo      AgentPro 项目启动脚本
echo ========================================
echo.

REM 检查 Docker Desktop 是否运行
echo [1/7] 检查 Docker Desktop 状态...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker Desktop 未运行，请先启动 Docker Desktop 并重试。
    pause
    exit /b 1
)
echo        Docker Desktop 已运行。

REM 2. 检查并启动 PostgreSQL 容器
echo [2/7] 检查 PostgreSQL 容器 (my-pgvector)...
docker ps --filter "name=my-pgvector" --format "table {{.Names}}" | findstr /C:"my-pgvector" >nul
if %errorlevel% equ 0 (
    echo        PostgreSQL 容器已在运行。
) else (
    echo        PostgreSQL 容器未运行，正在启动...
    REM 确保 docker-compose.yml 在当前目录（或指定路径）
    if not exist "docker-compose.yml" (
        echo [错误] 未找到 docker-compose.yml 文件，请将文件放在当前目录。
        pause
        exit /b 1
    )
    docker-compose up -d
    if %errorlevel% neq 0 (
        echo [错误] 启动 PostgreSQL 容器失败，请检查 docker-compose.yml 和 Docker 状态。
        pause
        exit /b 1
    )
    echo        PostgreSQL 容器启动命令已执行，等待就绪...
)

REM 等待 PostgreSQL 真正可连接（超时 30 秒）
echo [3/7] 等待 PostgreSQL 就绪...
set TIMEOUT=30
set /a ELAPSED=0
:waitpg
timeout /t 1 /nobreak >nul
set /a ELAPSED+=1
if %ELAPSED% geq %TIMEOUT% (
    echo [警告] PostgreSQL 未在 %TIMEOUT% 秒内就绪，继续启动（可能后续会失败）
    goto :pgready
)
REM 使用 docker exec 运行 pg_isready 检查本地容器（假设容器名为 my-pgvector）
docker exec my-pgvector pg_isready -U root -d AgentPro >nul 2>&1
if %errorlevel% equ 0 (
    echo        PostgreSQL 已就绪！
    goto :pgready
) else (
    goto :waitpg
)
:pgready


REM 激活虚拟环境（根据实际情况修改路径，若使用 uv 则调整）
echo [4/7] 检查 Python 环境...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo        已激活虚拟环境 .venv
) else (
    echo        未找到虚拟环境，使用系统 Python
)

echo [5/7 start Hub 服务器...
start "AgentPro Hub" /MIN cmd /c "python -m hub.server 2>&1 | findstr /v "INFO""
if %errorlevel% neq 0 (
    echo [错误] Hub 服务器启动失败，请检查 hub/server.py 是否存在。
    pause
    exit /b 1
)
echo        Hub 服务器已启动（后台运行）

REM wait Hub start
timeout /t 2 /nobreak >nul

REM start主 Agent（main.py）
echo [6/7] start Agent...
start "AgentPro Main" /MIN cmd /c "python main.py 2>&1"
if %errorlevel% neq 0 (
    echo [错误] 主 Agent 启动失败，请检查 main.py 和依赖。
    pause
    exit /b 1
)
echo        主 Agent 已启动（后台运行）

REM wait Agent init
timeout /t 3 /nobreak >nul

REM start Web client（client.py）
echo [7/7] start Web client...
start "AgentPro Web" /MIN cmd /c "python client.py 2>&1"
if %errorlevel% neq 0 (
    echo [错误] Web 客户端启动失败，请检查 client.py 是否存在。
    pause
    exit /b 1
)
echo        Web 客户端已启动（后台运行）

REM 等待 Web 服务就绪
timeout /t 3 /nobreak >nul

REM 打开浏览器
echo 正在打开浏览器...
start http://127.0.0.1:8000

echo.
echo ========================================
echo 所有组件已启动！
echo - Hub 日志: 可在对应命令行窗口查看
echo - Agent 日志: 可在对应命令行窗口查看
echo - Web 界面: http://127.0.0.1:8000
echo ========================================
echo 按任意键关闭本窗口（不会停止后台服务）
pause >nul