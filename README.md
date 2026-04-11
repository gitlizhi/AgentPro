# AgentPro - 智能体框架

AgentPro 是一个基于 LangChain 和 LangGraph 构建的高级 AI 智能体框架，支持多智能体协作、短期记忆、长期记忆、记忆转经验（可自动根据过往记忆进行经验总结）、HITL、意图识别、定时任务、主动思考和未来多模态扩展。它旨在提供一个灵活、可扩展的生产级智能体系统，让你能够快速构建自己的 AI 助手。

## ✨ 核心特性

- **多智能体架构**：支持同时运行多个独立智能体，通过 Hub 进行消息路由，互不干扰。
- **短期记忆**：基于 PostgreSQL 的检查点（Checkpointer）实现对话历史持久化，智能体重启后仍能恢复上下文。
- **长期记忆**：使用 ChromaDB 向量数据库存储用户事实，并同步为可读的 Markdown 文件，支持智能去重和整理。
- **记忆转经验**：复杂任务处理后，可将处理步骤，踩坑经历转为经验，下次处理类似任务，可从经验中提取，显著提升智能体处理速度，让智能体越来越聪明。
- **意图识别**：利用大模型对用户输入进行分类（聊天、复杂任务、设置提醒、查询提醒等），并提取关键参数，快速响应。
- **定时提醒**：集成 APScheduler，支持用户设置一次性提醒，到期自动发送通知。
- **人机协作**：新增人机协作功能，在执行关键的操作时，会主动触发HITL，可以让人类确认后再进行操作，提高安全性。
- **主动思考与沟通**：后台任务定期生成内在想法，结合记忆和近期对话，可在不繁忙时，偶尔主动与用户互动。
- **技能系统**：内置 deepagents，支持通过 `SKILL.md` 和脚本定义可扩展的技能，实现渐进式披露。
- **多模态扩展**：支持通过切换视觉模型（如 GLM-4.6V）处理图片输入。
- **WebSocket 通信**：所有智能体通过统一的 Hub 进行通信，支持点对点和广播消息。

## 🛠️ 技术栈

- **核心框架**：LangChain, LangGraph, deepagents
- **数据库**：PostgreSQL（短期记忆、提醒存储）、ChromaDB（长期记忆）、SQLAlchemy（APScheduler 作业存储）
- **消息通信**：WebSockets（自定义 Hub）
- **调度器**：APScheduler (AsyncIOExecutor)
- **模型**：智谱 AI（GLM-4.7, GLM-4.6v, GLM-4-Flash），支持 OpenAI 兼容格式
- **异步运行时**：Python 3.12+，asyncio

## 🚀 快速开始

### 环境要求

- Python 3.12 或更高版本
- PostgreSQL 数据库（用于短期记忆和提醒）
- （可选）ChromaDB 缓存目录（自动创建）

### 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/gitlizhi/AgentPro.git
   cd AgentPro
   ```

2. 创建虚拟环境并安装依赖（使用 uv 或 pip）
   ```bash
      pip install uv
      uv venv
      source .venv/bin/activate  # Windows: .venv\Scripts\activate
      uv pip install -e .
   ```

3. 配置环境变量
   
     ```bash
      # PostgreSQL 连接（需自定义一个链接）
      POSTGRES_URI=postgresql://user:password@localhost:5432/agentpro  
      # 智谱 AI API 密钥
      ZHIPU_API_KEY=your_api_key_here
      # Hub 配置
      HUB_HOST=localhost
      HUB_PORT=8765
      ```
4. 安装docker desktop
  官网地址：https://www.docker.com/products/docker-desktop/
  安装后启动 docker desktop

5. 一键启动 
    ```bash
   双击  start_project.bat
   # 停止脚本    stop_project.bat
   ```


客户端支持以下命令（新版客户端已经移除旧命令）：

- /new 消息：开始新对话（生成新 thread_id）

    ```text
   agentpro/
   ├── agent/                      # 核心智能体模块
   │   ├── brain.py                 # 大脑决策层
   │   ├── core.py                  # 智能体主类
   │   ├── communication.py         # WebSocket 通信
   │   ├── db.py                    # 数据库连接池
   │   ├── memory.py                # 长期记忆（ChromaDB + Markdown）
   │   ├── memory_consolidation.py  # 记忆整理（去重、合并）
   │   ├── scheduler.py             # APScheduler 调度器
   │   ├── tasks.py                 # 后台任务（发送提醒、记忆整理）
   │   ├── utils.py                 # 工具函数（模型调用）
   │   ├── intent.py                # 意图枚举和描述
   │   ├── model_config.py          # 模型配置管理
   │   └── skills/                  # 技能目录（按需加载）
   │   └── data/                    # 记忆转经验（按需加载，渐进式披露）
   │       ├── memories/            # 过往经验目录
   │       ├── pengding/            # 待处理经验
   ├── hub/                         # 消息 Hub
   │   └── server.py
   ├── agent_memory/                # 长期记忆 Markdown 文件
   ├── chroma_db/                   # ChromaDB 持久化目录
   ├── client.py                       # 客户端
   ├── .env.example                 # 环境变量示例
   ├── main.py                      # 应用入口
   ├── start_project.bat                  # 一键启动脚本
   ├── stop_project.bat                  # 停止脚本
   ├── clean_checkpoints.py         # 清理短期记忆脚本
   ├── requirements.txt             # 依赖列表（可选）
   ├── pyproject.toml               # 项目配置（uv/pip）
   └── README.md
   ```


## 🔧 配置说明
### 主要配置项
|变量名|	说明|	默认值|
| --- | --- | --- |
|POSTGRES_URI|	PostgreSQL 连接字符串|	postgresql://...|
|ZHIPU_API_KEY|	智谱 AI API 密钥|	无
|HUB_HOST	Hub| 服务器主机|	localhost|
|HUB_PORT	Hub| 服务器端口|	8765|
|MEMORY_MARKDOWN_DIR	|长期记忆 Markdown 文件目录|	./agent_memory|
|CHROMA_PERSIST_DIR|	ChromaDB 持久化目录	|./chroma_db|


### 模型配置

在 agent/model_config.py 中预定义了多个模型配置：

- default: GLM-4.7（默认聊天模型）

- vision: GLM-4.6v（视觉模型）

- 其他如 deepseek, claude, gemini 等可自行扩展。


## 🧠 主动思考与内在自驱力
智能体每小时会随机生成一个想法，并可能主动向用户发送消息。这模拟了内在的思考能力，让智能体更像一个真正的伙伴。你可以在 brain.py 的 _generate_thought 方法中自定义思考类型和生成逻辑。

## 🗂️ 记忆系统
- 短期记忆：由 checkpointer 自动保存每个对话线程的消息历史，支持重启恢复。

- 长期记忆：通过 remember_fact 技能存储用户事实到 ChromaDB，并同步为 Markdown 文件（agent_memory/<user_id>.md）。每日凌晨3点（可在main.py更改时间）自动整理，使用大模型去重和合并相似事实。

## 📦 依赖管理

项目使用 uv 进行依赖管理，pyproject.toml 已列出所有必要依赖。你也可以使用 pip 安装。

### 主要依赖（不全，建议按照pyproject.toml进行加载）：

- langchain>=1.2.10

- langchain-openai>=1.1.10

- langgraph-checkpoint-postgres>=3.0.4

- deepagents>=0.4.5

- chromadb>=1.5.2

- apscheduler>=3.11.2

- psycopg[binary]>=3.2.0

- httpx>=0.27.0

- websockets>=16.0

- python-dotenv>=1.2.2


## 运行清理脚本（删除指定或所有短期记忆）：

   ```bash
   python clean_checkpoints.py --all
   python clean_checkpoints.py --thread "agent_17_super_user_xxx"
   ```

## 🤝 贡献指南
### 欢迎贡献！请遵循以下步骤：

1. Fork 仓库

2. 创建功能分支 (git checkout -b feature/amazing-feature)

3. 提交更改 (git commit -m 'Add amazing feature')

4. 推送到分支 (git push origin feature/amazing-feature)

5. 打开 Pull Request

请确保代码符合 PEP 8 规范，并为新功能添加相应测试。

## 📄 许可证
本项目采用 MIT 许可证。详见 LICENSE 文件。

## 🙏 致谢
LangChain 团队提供的强大框架

deepagents 项目带来的技能系统灵感

智谱 AI 提供的优秀模型 API



