@echo off
chcp 65001 >nul
title AgentPro 启动器
echo ========================================
echo      AgentPro 项目启动脚本
echo ========================================
echo.

REM 检查 Docker Desktop 是否运行
echo [1/5] 检查 Docker Desktop 状态...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker Desktop 未运行，请先启动 Docker Desktop 并重试。
    pause
    exit /b 1
)
echo        Docker Desktop 已运行。

REM 激活虚拟环境（根据实际情况修改路径，若使用 uv 则调整）
echo [2/5] 检查 Python 环境...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo        已激活虚拟环境 .venv
) else (
    echo        未找到虚拟环境，使用系统 Python
)

echo [3/5] start Hub 服务器...
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
echo [4/5] start Agent...
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
echo [5/5] start Web client...
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