# AgentPro 技术架构文档

## 一、项目概述

AgentPro 是一个基于 LangGraph + LangChain 的多智能体协作平台。多个 AI Agent 通过 WebSocket Hub 互联，可以私聊、群聊、委派任务、使用技能，并通过 Docker 沙箱执行代码。后端以 DeepSeek V4 为主力模型，PostgreSQL 做状态持久化，ChromaDB 做向量记忆存储。

### 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangGraph + deepagents |
| LLM 后端 | DeepSeek V4（兼容 OpenAI API） |
| 通信 | WebSocket（Hub-Spoke 模型） |
| 状态持久化 | PostgreSQL（checkpoint、rooms、reminders） |
| 向量记忆 | ChromaDB（用户记忆 + 技能索引） |
| Web 前端 | FastAPI + 原生 HTML/JS（client.py + client.html） |
| 沙箱执行 | Docker（自定义镜像 my-agent-base） |
| 浏览器自动化 | Playwright / Chromium |
| 调度 | APScheduler（PostgreSQL job store） |

---

## 二、系统架构

### 2.1 总览

```
┌──────────────────────────────────────────────────────┐
│  浏览器 (client.html)                                 │
│  - 智能体列表、对话面板、群组房间、启动智能体表单       │
└──────────────┬───────────────────────────────────────┘
               │ WebSocket
┌──────────────▼───────────────────────────────────────┐
│  client.py (FastAPI + super_user WebSocket 客户端)    │
│  - 前端页面托管                                        │
│  - REST API: /chat/*, /agents/launch                 │
│  - 以 super_user 身份桥接浏览器和 Hub                  │
└──────────────┬───────────────────────────────────────┘
               │ WebSocket
┌──────────────▼───────────────────────────────────────┐
│  Hub Server (hub/server.py)                          │
│  - 中心消息代理                                        │
│  - 房间管理 (create/join/leave/invite)                 │
│  - 在线/离线事件广播                                    │
│  - 房间成员持久化 (PostgreSQL)                          │
└──┬──────────┬──────────────┬─────────────────────────┘
   │          │              │
┌──▼────┐ ┌───▼────┐ ┌──────▼──────┐
│Agent A│ │Agent B │ │reminder_bot │
│core.py│ │core.py │ │(定时提醒)    │
│brain  │ │brain   │ └─────────────┘
└──┬────┘ └───┬────┘
   │          │
   │  subprocess.Popen
   │          │
┌──▼──────────▼──────┐
│  子智能体 (临时)     │
│  main.py --agent-id │
└────────────────────┘
```

### 2.2 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **Agent 主控** | `agent/core.py` | WebSocket 连接管理、消息路由分发、房间成员缓存 |
| **Brain 决策层** | `agent/brain.py` | LLM 调用、意图识别、工具注册、对话管理、任务缓冲 |
| **Communication** | `agent/communication.py` | WebSocket 客户端，与 Hub 通信 |
| **Prompt 管理** | `agent/prompts.py` | 所有 LLM 提示词集中管理 |
| **模型配置** | `agent/model_config.py` | 多模型兼容（DeepSeek、OpenAI、Anthropic、Ollama） |
| **配置系统** | `config.py` | Pydantic 配置，含 Database、Hub、Model、Agent、Backend 子配置 |

### 2.3 智能体内部结构

```
Agent (core.py)
  ├── Communication (WebSocket 客户端)
  ├── Brain (brain.py)
  │     ├── LangGraph Agent (deepagents)
  │     │     ├── SubAgents (反思、技能)
  │     │     ├── SummarizationMiddleware
  │     │     ├── FilesystemMiddleware
  │     │     └── TodoListMiddleware
  │     ├── ConversationTracker (轮次计数+硬上限)
  │     ├── TaskBuffer (任务步骤缓冲)
  │     ├── LongTermMemory (ChromaDB)
  │     └── Tools (send_to_agent, launch_agent, log_memory, ...)
  ├── DockerSandboxBackend (代码执行)
  └── Browser Tools (Playwright)
```

---

## 三、通信系统

### 3.1 Hub-Spoke 消息模型

所有消息经过 Hub 中心路由。消息类型：

| 类型 | 方向 | 说明 |
|------|------|------|
| `message` | A→B | 点对点私聊 |
| `group_message` | A→Room | 群聊广播 |
| `register` / `register_ack` | Agent→Hub | 注册 |
| `get_agents` / `agents_list` | Agent→Hub | 查询在线列表 |
| `get_my_rooms` / `rooms_list` | Agent→Hub | 查询所属房间 |
| `get_room_members` / `room_members` | Agent→Hub | 查询房间成员 |
| `room_members_update` | Hub→All | 房间成员变更广播 |
| `agent_online` / `agent_offline` | Hub→All | 上下线广播 |
| `create_room` / `join_room` / `leave_room` | Agent→Hub | 房间操作 |
| `invite_to_room` | Hub→Agent | 被邀请通知 |

### 3.2 群聊处理流程

```
Agent A 发群消息
  → Hub 收到 group_message
  → Hub 遍历 room_members，排除 sender
  → 向每个成员发送 group_message（含 room_id、sender、payload）
  → 各 Agent 的 core.py:_handle_message 接收
  → 调用 _process_group_message(room_id, sender, text)
  → brain.process(group_context={room_id, members})
  → 系统提示词注入群聊上下文：
      [群聊上下文]
      你正在群聊「明红辩论室」中，当前群成员：小明、小红、agent_main。
      群聊规则：
      1. 只回复与你相关、或明确 @你 的消息
      2. 回复时必须使用 send_group_message 工具
      3. 禁止用 send_to_agent 私聊群成员绕过群聊
```

### 3.3 Agent 消息缓冲机制

Agent 接收到其他 Agent 的私聊消息时，不立即处理，而是：

1. 消息累积到 `agent_msg_cache[sender]`
2. 如果有正在处理中的任务 → 只累积，不打断
3. 如果 timer 在 sleep → 取消旧 timer，启动新的 5 秒 timer
4. 5 秒后合并所有累积消息，一次送入 LLM

这样避免 Agent 对每一条消息都立即回复，减少来回次数。

### 3.4 对话轮次限制

参见 [五、智能体私聊管控](#五智能体私聊管控)。

---

## 四、记忆系统

记忆系统由三条流水线组成：

### 4.1 流水线总览

```
┌─────────────────────────────────────────────────────────┐
│  用户对话 / 智能体交互                                    │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────▼──────────┐
    │ 即时记忆（工具调用） │  log_memory(desc, result)
    │ TaskBuffer        │  → 步骤缓冲 → task_complete 触发
    │ （内存中）         │
    └────────┬──────────┘
             │ task_complete=True
    ┌────────▼──────────┐
    │ 反思 & 技能生成     │  reflect_on_task()
    │ agent/reflection  │  → 成功时 create_skill_from_reflection()
    │                   │  → 技能存入 agent/data/skills/ + ChromaDB
    └───────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │ 后台提取（每 5 分钟）                                  │
    │ conversation_memory_worker()                         │
    │  → 扫描 PostgreSQL checkpoint 增量                   │
    │  → LLM 提取 facts（用户画像）+ events（事件记录）      │
    │  → 语义去重后存入 ChromaDB + agent_memory/*.md        │
    └──────────────┬──────────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────────┐
    │ 每日整合（凌晨 3:00）                                  │
    │ consolidate_all_users()                              │
    │  → 读取 agent_memory/*.md 中所有 facts                │
    │  → LLM 去重合并                                       │
    │  → 同步回 ChromaDB                                   │
    └─────────────────────────────────────────────────────┘
```

### 4.2 即时记忆（log_memory）

智能体在关键步骤后调用 `log_memory(description, result)`：

- 记录到 `TaskBuffer`（按 thread_id 分组的步骤列表）
- `task_complete=True` 时触发反思：任务数据写入 JSON → 反思 worker 分析 → 成功时生成可复用技能
- TaskBuffer 同时用于对话终止判断中的活跃任务检测

### 4.3 对话提取（conversation_memory_extractor）

- **触发**：后台 asyncio 任务，每 5 分钟扫描一次
- **增量**：跟踪每个 thread 的已处理消息数（存 ChromaDB metadata），只处理新增
- **提取**：将最近 30 条消息发给 LLM，提取两类信息：

| 类型 | 说明 | 存储位置 |
|------|------|---------|
| facts | 用户身份、偏好、长期约定（必须以"用户"开头） | ChromaDB + MD 文件 |
| events | 任务执行记录、结果、用户反馈 | 仅 ChromaDB |

- **fact 严禁内容**：会话操作记录（"用户启动了智能体X"）、临时指令、智能体行为、会话内容、系统状态
- **去重**：cosine 语义相似度 < 0.15 视为重复，过滤
- **频率控制**：同一 thread 增量不足 20 条消息时跳过

### 4.4 每日整合（memory_consolidation）

- **触发**：APScheduler cron，每天凌晨 3:00 UTC
- **流程**：
  1. 遍历 `agent_memory/` 下所有 `.md` 文件
  2. 解析每条 fact（含时间戳和元数据）
  3. LLM 去重：合并语义相同、去除被包含的、保留更完整的
  4. 差分同步 ChromaDB：删除过时条目，添加新条目
  5. 重写 MD 文件

### 4.5 存储

| 后端 | 路径 | 内容 |
|------|------|------|
| ChromaDB | `./chroma_db/` | 向量化的 facts + events + skills |
| Markdown | `./agent_memory/{user_id}.md` | 仅 type="fact" 的用户画像 |
| PostgreSQL | `checkpoints` 表 | LangGraph 对话状态 |
| JSON | `agent/data/pending_tasks/` | 待反思的任务数据（临时） |

---

## 五、智能体私聊管控

### 5.1 问题背景

多智能体之间可以自由调用 `send_to_agent` 进行私聊。如果不加限制，可能：
- 两个智能体陷入无限循环对话
- 无意义的客套话和重复确认消耗 token
- 任务协调被 LLM 误判为"死循环"而静默终止

### 5.2 解决方案：ConversationTracker

`agent/conversation_tracker.py` 为每一对智能体维护独立的对话状态，包括轮次计数、警告阈值和硬上限。

#### 生命周期

```
新会话开始（tracker 无状态）
  → 生成新 thread_id（隔离的 LangGraph checkpoint）
  → 使用同一个 thread_id 直到会话结束（保证上下文连续）

轮次 1-3: 正常通信，无干预
轮次 4-5: 💡 系统提示注入 "请在 X 轮内完成交流"
轮次 6-7: ⚠️ 最后通牒 "还剩 2 轮 / 最后 1 轮"
轮次 8:   🔒 硬截断 + 30 分钟冷却 + 通知 super_user

会话重置触发条件：
  - 用户发送消息给任一参与方（自动解除所有限制）
  - 智能体发送 [停止交流] 标记
  - 30 分钟冷却到期
```

#### Thread ID 隔离

每次新会话生成独立的 `thread_id`，不同会话的 LangGraph checkpoint 互不干扰。这解决了旧设计中"同一对智能体的所有历史对话混在一个 checkpoint 里"的问题。

#### 门控位置

| 层级 | 位置 | 行为 |
|------|------|------|
| 入口 | `core.py:_handle_message` | 已截断对话直接丢弃 |
| 缓冲 | `brain.py:_handle_with_agent` | 已截断对话不进入延迟处理 |
| 处理 | `brain.py:_process_agent_message` | 硬截断检查 + 轮次警告注入 |
| 发送 | `brain.py:_create_send_to_agent_tool` | 发送前检查 can_send，发送后 record_send |

---

## 六、Docker 沙箱

### 6.1 设计原则

`agent/sandboxed_backend.py` 提供隔离的命令执行环境：

- **安全加固**：非 root 用户（nobody）、根文件系统只读、cap_drop=ALL、no-new-privileges、进程数限制
- **资源限制**：mem_limit、cpu_limit
- **挂载**：临时 workspace（/workspace）、桌面目录（/desktop）、技能脚本（/agent/skills，只读）、对话历史（/conversation_history）

### 6.2 容器生命周期

```
execute() 调用
  → 创建容器（标签: agentpro.sandbox=true）
  → 等待完成（默认 30s 超时）
  ├── 正常完成 → 拿日志 → remove()
  ├── 超时     → kill() → remove()
  └── 异常     → finally 中 remove(force=True)

启动清理（__init__）：
  - exited 容器 → 直接 remove()
  - running 容器 → 仅清理运行超过 30 分钟（视为孤儿）
  - 运行中且 < 30 分钟 → 跳过（可能是其他智能体在使用）
```

---

## 七、技能系统

### 7.1 技能来源

| 来源 | 路径 | 说明 |
|------|------|------|
| 内置技能 | `agent/skills/` | 预置的可复用技能（如 browser-automation） |
| 学习技能 | `agent/data/skills/` | 任务反思后自动生成 |

### 7.2 技能工具

| 工具 | 功能 |
|------|------|
| `list_skills` | 列出所有可用技能 |
| `search_skills` | 按关键词搜索技能 |
| `load_skill` | 加载技能内容（Markdown 指令） |
| `skill_stats` | 查看技能使用统计 |
| `upgrade_skill` | 技能版本升级 |
| `report_skill_result` | 上报技能执行结果 |

### 7.3 技能生命周期

```
执行任务 → task_complete=True
  → 反思 worker: reflect_on_task()
  → 成功且 should_create_skill=True
  → create_skill_from_reflection()
  → Markdown 文件 → agent/data/skills/
  → 向量化 → ChromaDB agent_skills 集合
  → 下次执行时可被 search_skills 检索
```

低价值技能会被 `skill_version_manager` 自动归档。

---

## 八、子智能体

### 8.1 启动机制

智能体可通过 `launch_agent(name, expertise)` 工具创建子智能体：

```
brain.py: launch_agent 工具
  → tools.py: subprocess.Popen
  → main.py --agent-id {name} --system-prompt {expertise}
  → 新进程连接 Hub，注册为新 Agent
  → 专长通过 build_launch_agent_prompt() 注入系统提示
```

### 8.2 生命周期管理

| 工具 | 功能 |
|------|------|
| `launch_agent(name, expertise)` | 启动子智能体 |
| `stop_agent(name)` | 停止指定子智能体 |
| `stop_all_agents_impl()` | 停止所有子智能体 |

---

## 九、调度系统

基于 APScheduler + PostgreSQL job store：

| 任务 | 触发 | 说明 |
|------|------|------|
| `send_reminder` | 用户指定时间 | 通过 reminder_bot 发送提醒消息 |
| `consolidate_all_users` | 每天 3:00 AM | 记忆去重整合 |

---

## 十、数据流示意

### 10.1 用户消息处理

```
用户发送 "帮我搜索马斯克的最新新闻"
  → client.py WebSocket → Hub → Agent core.py
  → _handle_message → _process_message
  → brain.process(user_id="super_user", ...)
  → _classify_intent → COMPLEX_TASKS
  → _handle_with_agent (super_user 路径)
  → 构建系统提示 + 长期记忆 + 群聊上下文
  → LangGraph Agent astream
  → LLM 决策 → 工具调用 (TavilySearch)
  → AI 回复 → comm.send_to_agent(super_user, reply)
  → Hub → client.py → 浏览器显示
```

### 10.2 智能体协作

```
Agent A 调用 send_to_agent(B, "帮我校验这段代码")
  → ConversationTracker 检查 can_send
  → 记录轮次 → comm.send_to_agent(B, msg)
  → Hub 转发 → Agent B core.py
  → 累积到 agent_msg_cache["A"] → 5s timer
  → _process_agent_message
  → ConversationTracker.get_or_create_thread_id → 独立 checkpoint
  → 系统提示 + 轮次警告 + 群聊上下文（如有）
  → LLM 处理 → 回复发送回 A
```

---

## 十一、关键设计决策

| 决策 | 理由 |
|------|------|
| Hub-Spoke 而非 P2P | 中心路由可控、可审计；Hub 负责房间成员持久化和广播 |
| 轮次硬限制替代 LLM 判死 | LLM 判断对话是否"死循环"不可靠，容易误杀；确定性规则更安全 |
| Thread ID 与会话绑定 | 避免不同会话的历史消息混入同一个 checkpoint，减少上下文噪音 |
| 群聊上下文注入系统提示而非用户消息 | 保持用户消息干净，系统级指令不会被 LLM 当成"用户说的" |
| Docker 容器每次 execute 后清理 | 沙箱是一次性执行环境，无状态，不留残留 |
| 延迟合并 Agent 消息（5 秒缓冲） | 减少来回次数，避免 Agent 对每条消息单独回复 |

---

## 十二、文件清单

| 文件 | 职责 |
|------|------|
| `main.py` | 入口，初始化数据库/调度器/智能体/后台 worker |
| `config.py` | 全局配置 |
| `agent/core.py` | Agent 主控，消息路由 |
| `agent/brain.py` | LLM 决策，工具注册，对话管理 |
| `agent/communication.py` | WebSocket 客户端 |
| `agent/prompts.py` | 所有 LLM 提示词 |
| `agent/conversation_tracker.py` | 智能体对话轮次控制 |
| `agent/task_buffer.py` | 任务步骤缓冲 |
| `agent/memory.py` | ChromaDB 记忆存储 |
| `agent/conversation_memory_extractor.py` | 对话记忆后台提取 |
| `agent/memory_consolidation.py` | 每日记忆去重整合 |
| `agent/reflection.py` | 任务反思 + 技能生成 |
| `agent/skill_tools.py` | 技能管理工具 |
| `agent/skill_version_manager.py` | 技能版本管理 |
| `agent/sandboxed_backend.py` | Docker 沙箱 |
| `agent/browser_tools.py` | Playwright 浏览器工具 |
| `agent/tools.py` | 自定义 LangChain 工具 |
| `agent/scheduler.py` | APScheduler 调度 |
| `agent/tasks.py` | 提醒 + 整合任务 |
| `agent/model_config.py` | 多模型配置 |
| `agent/intent.py` | 意图类型枚举 |
| `agent/db.py` | PostgreSQL 连接池 |
| `hub/server.py` | WebSocket Hub |
| `client.py` | FastAPI + super_user 客户端 |
| `client.html` | Web 前端 |
