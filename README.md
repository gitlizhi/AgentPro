# AgentPro

<div align="center">

**基于 LangGraph + LangChain 的多智能体协作平台**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-orange.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-purple.svg)](https://langchain-ai.github.io/langgraph/)

</div>

---

## 目录

- [项目介绍](#项目介绍)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [记忆系统](#记忆系统)
- [技能系统](#技能系统)
- [Computer Use（桌面自动化）](#computer-use桌面自动化)
- [清理脚本](#清理脚本)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目介绍

AgentPro 是一个**生产级多智能体协作平台**。多个 AI Agent 通过 WebSocket Hub 互联，可以私聊、群聊、委派任务、使用技能、操控桌面应用、执行代码。Agent 具备长期记忆和自我反思能力，能自动从过往任务中提炼经验并生成可复用技能。

### 能力一览

```
用户 → Agent
        │
        ├── 💬  多智能体私聊 / 群聊协作
        ├── 🧠  长期记忆（ChromaDB）+ 短期记忆（PostgreSQL）
        ├── 🔄  记忆转经验 → 自动生成可复用技能
        ├── 📐  上下文管理（SystemMessage 分层 + Token 预算 + 工具输出压缩）
        ├── 🖥️  Computer Use（桌面自动化：鼠标/键盘/OCR/视觉定位）
        ├── 🌐  浏览器自动化（Playwright）
        ├── 📦  Docker 沙箱代码执行
        ├── ⏰  定时提醒（APScheduler）
        ├── 👤  人机协作（HITL 审批）
        └── 💡  主动思考与互动
```

---

## 核心特性

### 智能体能力

| 特性 | 说明 |
|------|------|
| **多智能体架构** | 多个独立 Agent 并行运行，通过 Hub 消息路由，互不干扰 |
| **多智能体协作** | Agent 间可私聊通信、群聊讨论、委派任务，协同完成复杂目标 |
| **意图识别** | LLM 对用户输入分类（聊天/复杂任务/提醒/查询），快速路由 |
| **主动思考** | 后台定期生成内在想法，结合记忆和对话主动与用户互动 |
| **人机协作 (HITL)** | 关键操作触发人工审批，审批通过后继续执行 |

### 记忆与学习

| 特性 | 说明 |
|------|------|
| **短期记忆** | PostgreSQL checkpoint 持久化对话历史，重启后恢复上下文 |
| **长期记忆** | ChromaDB 向量存储用户画像和事件，自动去重，同步 Markdown 文件 |
| **记忆转经验** | 复杂任务完成后自动反思，成功经验生成可复用技能，失败总结教训 |
| **每日整合** | 凌晨 3:00 自动整理记忆，LLM 去重合并相似事实 |

### 自动化能力

| 特性 | 说明 |
|------|------|
| **Computer Use** | 19 个桌面操作工具：截图/OCR/视觉定位/鼠标键盘/UIAutomation/命令执行 |
| **浏览器自动化** | Playwright 驱动 Chromium，操控网页 |
| **Docker 沙箱** | 隔离的命令执行环境，安全加固，用完即焚 |
| **定时提醒** | APScheduler + PostgreSQL，支持自然语言设置提醒 |

### 扩展性

| 特性 | 说明 |
|------|------|
| **技能系统** | `SKILL.md` + 脚本定义可扩展技能，渐进式披露，按需加载 |
| **多模型支持** | DeepSeek / 智谱 GLM / OpenAI / Anthropic / Ollama，一键切换 |
| **多模态** | 支持视觉模型（GLM-4.6V / GLM-4.1V-Thinking-Flash）处理图片 |
| **上下文管理** | 借鉴 Claude Code 分层模型：SystemMessage/HumanMessage 角色分离、Token 预算主动跟踪、工具输出自动压缩、Agent 专属上下文文件 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│  浏览器 (client.html)                                 │
│  - 智能体列表、对话面板、群组房间、启动智能体表单       │
└──────────────┬───────────────────────────────────────┘
               │ WebSocket
┌──────────────▼───────────────────────────────────────┐
│  client.py (FastAPI + super_user WebSocket 客户端)    │
│  - 前端页面托管 / REST API                            │
└──────────────┬───────────────────────────────────────┘
               │ WebSocket
┌──────────────▼───────────────────────────────────────┐
│  Hub Server (hub/server.py)                          │
│  - 中心消息代理 / 房间管理 / 在线广播                  │
└──┬──────────┬──────────────┬─────────────────────────┘
   │          │              │
┌──▼────┐ ┌───▼────┐ ┌──────▼──────┐
│Agent A│ │Agent B │ │reminder_bot │
│core.py│ │core.py │ │(定时提醒)    │
│brain  │ │brain   │ └─────────────┘
└──┬────┘ └───┬────┘
   │          │
   │  LangGraph Agent (deepagents)
   │     ├── Computer Tools (19 个桌面自动化工具)
   │     ├── Browser Tools (Playwright)
   │     ├── DockerSandboxBackend
   │     ├── ContextManager (token 预算 + 工具压缩)
   │     ├── ConversationTracker (轮次控制)
   │     ├── TaskBuffer (任务缓冲)
   │     └── ChromaDB Memory
```

> 详细架构请参阅 [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## 快速开始

### 环境要求

- **Python** 3.12+
- **PostgreSQL** 数据库
- **Docker Desktop**（沙箱执行和浏览器自动化需要）

### 1. 克隆仓库

```bash
git clone https://github.com/gitlizhi/AgentPro.git
cd AgentPro
```

### 2. 安装依赖

```bash
pip install uv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### 3. 配置环境变量

创建 `.env` 文件：

```bash
# PostgreSQL 连接（必填）
POSTGRES_URI=postgresql://user:password@localhost:5432/agentpro

# API 密钥（至少配置一个）
ZHIPU_API_KEY=your_zhipu_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Hub 配置（可选，使用默认值即可）
HUB_HOST=localhost
HUB_PORT=8765
```

### 4. 启动 Docker Desktop

下载并启动 [Docker Desktop](https://www.docker.com/products/docker-desktop/)。

### 5. 一键启动

```bash
# Windows
双击 start_project.bat

# 或命令行
python main.py
```

启动后访问 `http://127.0.0.1:8000` 进入 Web 界面。

### 停止项目

```bash
双击 stop_project.bat
```

---

## 配置说明

### 主要环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `POSTGRES_URI` | PostgreSQL 连接字符串 | 必填 |
| `ZHIPU_API_KEY` | 智谱 AI API 密钥 | - |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | - |
| `HUB_HOST` | Hub 服务器主机 | `localhost` |
| `HUB_PORT` | Hub 服务器端口 | `8765` |
| `MEMORY_MARKDOWN_DIR` | 长期记忆 Markdown 目录 | `./agent_memory` |
| `CHROMA_PERSIST_DIR` | ChromaDB 持久化目录 | `./chroma_db` |
| `BROWSER_HEADLESS` | 设为 `1` 启用无头模式（不显示浏览器窗口） | `0` |
| `BROWSER_CHANNEL` | 浏览器通道：`chrome`（系统 Chrome）或 `chromium`（Playwright 自带） | `chrome` |
| `BROWSER_CDP_PORT` | CDP 远程调试端口，设置后连接已有 Chrome（如 `9222`），不启动新实例 | 空（不启用） |
| `CHROME_PATH` | Chrome 可执行文件路径，不设置则自动从注册表查找 | 空（自动查找） |
| `AGENT_SHOW_CONSOLE` | 设为 `1` 显示子智能体控制台窗口（调试用，默认隐藏） | 空（隐藏） |

### 模型配置

在 `agent/model_config.py` 中预定义了多个模型，支持按需扩展：

| 配置名 | 模型 | 用途 |
|--------|------|------|
| `default` | GLM-4.7 | 默认聊天模型 |
| `vision` | GLM-4.6V | 通用图片理解 |
| `computer_vision` | GLM-4.1V-Thinking-Flash | 桌面操作视觉识别 |
| `deepseek` | DeepSeek-V4 | 备选主力模型 |
| `ollama` | llama3.1 | 本地模型 |

支持扩展 OpenAI、Anthropic Claude、Gemini 等兼容格式的模型。

---

## 项目结构

```
AgentPro/
├── agent/                          # 核心智能体模块
│   ├── brain.py                    # 大脑决策层（LLM 调用、意图识别、工具注册）
│   ├── context_manager.py          # 上下文管理（Token 预算 + 工具输出压缩）
│   ├── core.py                     # 智能体主类（WebSocket 管理、消息路由）
│   ├── communication.py            # WebSocket 客户端
│   ├── computer_tools.py           # Computer Use：19 个桌面自动化工具
│   ├── tools.py                    # LangChain 工具（含 UIAutomation 封装）
│   ├── browser_tools.py            # Playwright 浏览器自动化
│   ├── message_buffer.py           # Agent 间消息缓冲队列
│   ├── tools_factory.py            # 工具工厂函数
│   ├── sandboxed_backend.py        # Docker 沙箱执行环境
│   ├── model_config.py             # 多模型配置管理
│   ├── prompts.py                  # 集中式提示词管理
│   ├── memory.py                   # ChromaDB 长期记忆
│   ├── memory_consolidation.py     # 每日记忆去重整合
│   ├── conversation_memory_extractor.py  # 对话记忆后台提取
│   ├── reflection.py               # 任务反思 + 技能生成
│   ├── skill_tools.py              # 技能检索工具
│   ├── skill_version_manager.py    # 技能版本管理
│   ├── conversation_tracker.py     # 智能体对话轮次控制
│   ├── task_buffer.py              # 任务步骤缓冲
│   ├── scheduler.py                # APScheduler 调度器
│   ├── tasks.py                    # 后台任务（提醒、记忆整理）
│   ├── intent.py                   # 意图枚举与描述
│   ├── db.py                       # PostgreSQL 连接池
│   ├── utils.py                    # 工具函数
│   └── skills/                     # 内置技能（SKILL.md）
│       ├── user-profile/           # 用户画像按需加载技能
│       ├── computer-automation/    # 桌面自动化技能
│       └── browser-automation/     # 浏览器自动化技能
│
├── hub/                            # WebSocket Hub
│   └── server.py                   # 中心消息路由 + 房间管理
│
├── agent_memory/                   # 长期记忆 Markdown 文件
├── agent/agent_context/            # Agent 专属上下文文件（CLAUDE.md 模式）
├── chroma_db/                      # ChromaDB 持久化目录
├── screenshots/                    # 桌面截图保存目录
│
├── client.py                       # FastAPI 服务端 + super_user 客户端
├── client.html                     # Web 前端界面
├── main.py                         # 应用入口
│
├── start_project.bat               # 一键启动（Windows）
├── stop_project.bat                # 一键停止（Windows）
├── clean_db.py                     # 数据库与临时文件清理脚本
├── pyproject.toml                  # 项目配置与依赖
├── ARCHITECTURE.md                 # 详细技术架构文档
└── README.md
```

---

## 记忆系统

### 三层记忆架构

```
用户对话 / 智能体交互
       │
  ┌────▼─────────────┐
  │ 短期记忆           │  PostgreSQL checkpoint
  │ (对话历史持久化)    │  重启恢复上下文
  └────┬─────────────┘
       │
  ┌────▼─────────────┐
  │ 长期记忆           │  ChromaDB 向量存储（facts + events）
  │ (后台自动提取)      │  + agent_memory/*.md（仅用户画像）
  │ 每 5 分钟增量提取   │  + 语义去重
  └────┬─────────────┘
       │
  ┌────▼─────────────┐
  │ 经验记忆           │  任务反思 → 技能生成
  │ (可复用技能库)      │  每日凌晨 3:00 整合
  └──────────────────┘

┌─────────────────────────────────────────┐
│ 检索方式：按需加载（非注入式）             │
│                                          │
│ Agent 判断需要用户背景时                   │
│   → 调用 load_user_profile 工具           │
│   → 读取 agent_memory/super_user.md       │
│   → 获取去重后的用户画像（仅 facts）        │
│                                          │
│ 对比旧方案：不再每次对话都注入记忆到提示词，  │
│ 节省 token，Agent 按需主动获取。           │
└─────────────────────────────────────────┘
```

### 关键设计

- **按需加载**：记忆不再自动注入系统提示词，Agent 通过 `load_user_profile` 工具主动获取用户画像，节省 token 并让 Agent 明确知道自己有哪些用户信息
- **facts vs events**：facts 是用户画像（"用户是 Python 开发者"），events 是操作记录（"搜索了新闻"）。仅 facts 写入 Markdown 文件，`load_user_profile` 仅返回 facts
- **MD 文件纯净**：不再写入 source/type/thread_id 等元数据，每行只有时间戳和事实内容
- **增量提取**：跟踪每个 thread 的处理进度，只提取新增消息
- **语义去重**：cosine 相似度 < 0.15 视为重复，自动过滤
- **每日整合**：凌晨 3:00 全量 LLM 去重合并，保持记忆库整洁

---

## 技能系统

Agent 通过**渐进式披露**获取操作指令——平时只在系统提示词中放一行"用 `load_skill` 加载技能"，需要时才加载完整 `SKILL.md`。这样既节省 token，又能随时扩展新能力。

### 内置技能

| 技能 | 触发词 | 说明 |
|------|--------|------|
| `user-profile` | 用户信息、个性化推荐、了解用户… | 按需加载用户画像（配合 `load_user_profile` 工具） |
| `computer-automation` | 打开应用、操作微信、电脑操作… | Windows 桌面自动化完整流程 |
| `browser-automation` | 打开网页、浏览器操作… | Playwright 浏览器操控 |

### 技能生命周期

```
执行任务 → 完成 (task_complete=True)
  → 反思 worker 分析成败
  → 成功且可复用 → 生成 SKILL.md
  → 向量化存入 ChromaDB
  → 下次 search_skills 可检索
  → 低价值技能自动归档
```

---

## Computer Use（桌面自动化）

AgentPro 具备完整的 **Windows 桌面自动化**能力，让 AI Agent 能像人一样操控电脑。

### 能力矩阵

| 类别 | 工具 | 能力 |
|------|------|------|
| **屏幕感知** | `computer_screenshot`、`computer_see_and_describe` | 截图 + 视觉模型理解屏幕内容 |
| **精确定位** | `computer_ocr_find`、`computer_locate` | EasyOCR 文字定位 / 20×14 网格视觉定位 |
| **UIA 操控** | `windows_automation`（22 种操作） | pywinauto 无障碍树，零坐标误差 |
| **窗口管理** | `computer_find_window`、`computer_find_app` | 查找/激活窗口，搜索并启动应用 |
| **鼠标操作** | `computer_move`/`click`/`double_click`/`right_click`/`scroll`/`drag` | 完整鼠标控制 |
| **键盘操作** | `computer_type`/`key_press`/`paste` | 英文输入 / 热键 / 中文剪贴板粘贴 |
| **命令执行** | `computer_execute` | Windows cmd，30s 超时 |

### 四级定位策略

```
Tier 0: UIAutomation（最精确）─── 无障碍树直接操控，像素级精确
  ↓ 非原生界面
Tier 1: OCR 文字定位 ─── EasyOCR 识别屏幕文字，返回精确坐标
  ↓ 文字不可见
Tier 2: 窗口查找 ─── pygetwindow 标题匹配，返回窗口矩形
  ↓ 无窗口标题
Tier 3: 视觉网格定位 ─── 20×14 网格 + GLM-4.1V 视觉模型
```

> 详细文档：[ARCHITECTURE.md 第十二章](./ARCHITECTURE.md#十二computer-use--windows-桌面自动化)

---

## 清理脚本

`clean_db.py` 是数据库与临时文件的统一清理工具，支持按类别精确清理或一键全清。

```bash
# 查看所有表的统计信息（不删除）
python clean_db.py --stats

# 清理工单
python clean_db.py --tickets              # 所有工单（含终态）
python clean_db.py --tickets --active-only # 仅活跃工单

# 清理编排计划
python clean_db.py --orchestration
python clean_db.py --orchestration --active-only

# 清理聊天与会话
python clean_db.py --chat                 # 所有聊天消息
python clean_db.py --conversations        # 所有会话线程（级联删除消息）

# 清理短期记忆（LangGraph 检查点）
python clean_db.py --checkpoints
python clean_db.py --checkpoints --thread "thread_id"  # 指定线程

# 按智能体清理（聊天 + 检查点 + 会话 + 关联工单 + 子任务 + 提醒）
python clean_db.py --agent agent_main

# 清理提醒
python clean_db.py --reminders

# 清理群聊房间
python clean_db.py --rooms               # 所有房间
python clean_db.py --room room_xxx       # 指定房间

# 清理临时文件
python clean_db.py --screenshots          # 浏览器截图
python clean_db.py --tool-outputs         # 工具输出日志
python clean_db.py --temp                 # 所有临时文件

# 一键全清（危险，需确认）
python clean_db.py --all
python clean_db.py --all --force          # 跳过确认
```

---

## 贡献指南

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 打开 Pull Request

代码请遵循 PEP 8 规范。

---

## 许可证

本项目采用 [MIT 许可证](./LICENSE)。

---

## 致谢

- [LangChain](https://www.langchain.com/) 和 [LangGraph](https://langchain-ai.github.io/langgraph/) 团队
- [deepagents](https://github.com/hwchase17/deepagents) 技能系统
- [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) Computer Use 范式
- [智谱 AI](https://open.bigmodel.cn/) 和 [DeepSeek](https://www.deepseek.com/) 模型 API
