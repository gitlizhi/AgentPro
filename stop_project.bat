@echo off
chcp 65001 >nul
title AgentPro 停止器
echo ========================================
echo      AgentPro 项目停止脚本
echo ========================================
echo.

REM 查找并终止 hub.server 进程
echo [1/3] 停止 Hub 服务器...
for /f "tokens=2" %%i in ('tasklist /fi "windowtitle eq AgentPro Hub" /nh') do (
    taskkill /pid %%i /f >nul 2>&1
    echo        已终止 Hub 进程 (PID: %%i)
)
for /f "tokens=2" %%i in ('tasklist /fi "windowtitle eq AgentPro Main" /nh') do (
    taskkill /pid %%i /f >nul 2>&1
    echo        已终止 Main Agent 进程 (PID: %%i)
)
for /f "tokens=2" %%i in ('tasklist /fi "windowtitle eq AgentPro Web" /nh') do (
    taskkill /pid %%i /f >nul 2>&1
    echo        已终止 Web 客户端进程 (PID: %%i)
)

REM 备用：通过端口或 Python 脚本名终止（防止窗口标题不匹配）
echo [2/3] 扫描残留进程...
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /nh') do (
    taskkill /pid %%i /f >nul 2>&1
)
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq pythonw.exe" /nh') do (
    taskkill /pid %%i /f >nul 2>&1
)

echo [3/3] 释放端口...
REM 释放 8765 和 8000 端口（可选）
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8765 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
    echo        已释放端口 8765
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
    echo        已释放端口 8000
)

echo.
echo ========================================
echo 所有相关进程已停止！
echo ========================================
pause