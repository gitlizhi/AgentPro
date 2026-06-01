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
| **模型配置** | `agent/model_config.py` | 多模型兼容（DeepSeek、OpenAI、Anthropic、Ollama）+视觉模型 |
| **配置系统** | `config.py` | Pydantic 配置，含 Database、Hub、Model、Agent、Backend 子配置 |
| **电脑操作** | `agent/computer_tools.py` | 19 个桌面自动化工具（鼠标/键盘/截图/OCR/视觉定位） |
| **Windows UIA** | `agent/tools.py` | pywinauto 封装，无障碍树控件级精确操控 |

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
  │     ├── LongTermMemory (ChromaDB + 按需加载)
  │     └── Tools (send_to_agent, launch_agent, log_memory, Computer Tools, ...)
  ├── Computer Tools (19 个桌面自动化工具)
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

记忆系统由三条流水线组成，采用**按需加载**设计：

### 4.1 流水线总览

```
┌─────────────────────────────────────────────────────────┐
│  用户对话 / 智能体交互                                    │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────▼──────────┐
    │ 即时记忆（工具调用）│  log_memory(desc, result)
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
    │  → facts → ChromaDB + MD / events → 仅 ChromaDB      │
    └──────────────┬──────────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────────┐
    │ 每日整合（凌晨 3:00）                                  │
    │ consolidate_all_users()                              │
    │  → 读取 agent_memory/*.md 中所有 facts                │
    │  → LLM 去重合并                                       │
    │  → 同步回 ChromaDB                                   │
    └─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  检索方式：按需加载（load_user_profile 工具）              │
│                                                          │
│  用户发消息                                               │
│    → 系统提示词中不再注入记忆（节省 token）                │
│    → Agent 判断需要用户背景时                              │
│    → 调用 load_user_profile 工具                          │
│    → 读取 agent_memory/super_user.md                      │
│    → 返回去重后的用户画像（仅 facts，不含 events）          │
│    → Agent 将用户信息融入回复                              │
└─────────────────────────────────────────────────────────┘
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
| facts | 用户身份、偏好、长期约定（必须以"用户"开头） | ChromaDB + MD 文件（仅用户画像） |
| events | 任务执行记录、结果、用户反馈 | 仅 ChromaDB |

- **fact 严禁内容**：会话操作记录（"用户启动了智能体X"）、临时指令、智能体行为、会话内容、系统状态
- **MD 文件只存事实本身**：不再写入 source/type/thread_id 等元数据，保持文件纯净可读
- **去重**：cosine 语义相似度 < 0.15 视为重复，过滤；`get_user_profile()` 还会在输出时二次去重
- **频率控制**：同一 thread 增量不足 20 条消息时跳过

### 4.4 每日整合（memory_consolidation）

- **触发**：APScheduler cron，每天凌晨 3:00 UTC
- **流程**：
  1. 遍历 `agent_memory/` 下所有 `.md` 文件
  2. 解析每条 fact
  3. LLM 去重：合并语义相同、去除被包含的、保留更完整的
  4. 差分同步 ChromaDB：删除过时条目，添加新条目
  5. 重写 MD 文件（纯事实格式，无元数据杂讯）

### 4.5 检索方式：按需加载

**设计理念**：不再每次对话都将记忆注入系统提示词（浪费 token），而是让 Agent 自行判断何时需要用户信息，通过 `load_user_profile` 工具按需获取。

**工具实现**（`brain.py:_create_load_user_profile_tool`）：
- 读取 `agent_memory/super_user.md` 文件
- 解析所有 `- 事实：` 行
- 去重后格式化返回

**技能支持**（`agent/skills/user-profile/SKILL.md`）：
- 教 Agent 何时应加载用户画像（个性化推荐、涉及用户身份、延续性对话）
- 何时不需要加载（纯技术问答、当前对话已有足够信息）
- 通过 `load_skill('user-profile')` 或 `search_skills` 发现

**对比**：

| 方面 | 旧方案（注入式） | 新方案（按需加载） |
|------|-----------------|-------------------|
| Token 消耗 | 每次对话都消耗 | 仅在需要时消耗 |
| Agent 感知 | Agent 不知道有哪些记忆 | Agent 主动获取，知道用户画像内容 |
| 信息范围 | 语义搜索 top-3（含 events） | 完整用户画像（仅 facts） |
| 维护性 | 分散在 brain.py 两处 | 集中在 memory.py + 一个工具 |

### 4.6 存储

| 后端 | 路径 | 内容 |
|------|------|------|
| ChromaDB | `./chroma_db/` | 向量化的 facts + events + skills |
| Markdown | `./agent_memory/{user_id}.md` | 仅 type="fact" 的用户画像（纯净格式） |
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
| 内置技能 | `agent/skills/` | 预置的可复用技能（user-profile、computer-automation、browser-automation 等） |
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
  → 构建系统提示 + 群聊上下文（长期记忆改为按需加载）
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
| 长期记忆按需加载而非注入 | 每次对话都注入消耗大量 token；Agent 主动调用工具获取，仅在需要时加载，且 Agent 明确知道自己用了哪些信息 |

---

## 十二、Computer Use — Windows 桌面自动化

### 12.1 概述

AgentPro 具备完整的 Windows 桌面自动化能力（Computer Use），允许 AI Agent 直接操控用户电脑：识别屏幕内容、移动鼠标、点击、键盘输入、执行命令、查找窗口等。设计上借鉴了 Open Interpreter 的计算机控制范式——Agent 先通过视觉模型理解屏幕，再执行精确的鼠标/键盘操作。

**核心理念**：先看后动，UIA 优先，OCR 次之，视觉最后。

### 12.2 架构分层

```
┌──────────────────────────────────────────────────────┐
│  AI Agent (LangGraph)                                │
│    → 加载 computer-automation 技能 (SKILL.md)        │
│    → 调用 19 个电脑操作工具                            │
└──────────────┬───────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  视觉感知层           │
    │  GLM-4.1V-Thinking  │  ← 智谱视觉模型
    │  EasyOCR            │  ← 屏幕文字识别
    │  Pillow (ImageGrab) │  ← 截图+网格叠加
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  操控执行层          │
    │  pyautogui          │  ← 鼠标/键盘模拟
    │  pywinauto (UIA)    │  ← Windows 无障碍树
    │  pygetwindow        │  ← 窗口查找/激活
    │  pyperclip          │  ← 剪贴板（中文输入）
    │  subprocess         │  ← Shell 命令执行
    └─────────────────────┘
```

### 12.3 全部 19 个工具

| # | 工具名称 | 所属模块 | 功能 |
|---|---------|---------|------|
| 1 | `windows_automation` | `tools.py` | UIAutomation 精确操控（22 种子操作：connect/click/type/search_controls 等） |
| 2 | `computer_screenshot` | `computer_tools.py` | 截取全屏，可选保存到 screenshots/ 目录 |
| 3 | `computer_see_and_describe` | `computer_tools.py` | 截图 + GLM-4.1V-Thinking-Flash 视觉模型描述屏幕内容 |
| 4 | `computer_ocr_find` | `computer_tools.py` | EasyOCR 中英文混合识别，返回目标文字的像素坐标和置信度 |
| 5 | `computer_locate` | `computer_tools.py` | 20×14 网格叠加 + 视觉模型定位，返回目标格子的中心坐标 |
| 6 | `computer_find_app` | `computer_tools.py` | 搜索开始菜单/Program Files/PATH/注册表，返回 exe 路径 |
| 7 | `computer_find_window` | `computer_tools.py` | pygetwindow 标题查找，返回精确位置/大小，自动激活并处理最小化 |
| 8 | `computer_paste` | `computer_tools.py` | 剪贴板写入 + Ctrl+V，**中文文本输入的唯一正确方式** |
| 9 | `computer_move` | `computer_tools.py` | 移动鼠标到指定坐标 |
| 10 | `computer_click` | `computer_tools.py` | 左键单击（可带坐标或当前位置） |
| 11 | `computer_double_click` | `computer_tools.py` | 双击 |
| 12 | `computer_right_click` | `computer_tools.py` | 右键点击 |
| 13 | `computer_type` | `computer_tools.py` | 逐字符键盘输入（**仅支持英文 ASCII**） |
| 14 | `computer_key_press` | `computer_tools.py` | 组合键/热键：enter, ctrl+c, alt+tab, win+d 等 |
| 15 | `computer_scroll` | `computer_tools.py` | 鼠标滚轮（正=上，负=下） |
| 16 | `computer_drag` | `computer_tools.py` | 鼠标拖拽 |
| 17 | `computer_get_screen_size` | `computer_tools.py` | 获取屏幕分辨率 |
| 18 | `computer_get_cursor_position` | `computer_tools.py` | 获取当前鼠标位置 |
| 19 | `computer_execute` | `computer_tools.py` | 执行 Windows cmd 命令（30s 超时，输出截断 4000 字符） |

### 12.4 四级定位策略（按优先级严格遵循）

这是 Computer Use 系统最核心的设计——从最精确到最模糊的定位降级链：

| 层级 | 工具 | 适用场景 | 精度 |
|------|------|---------|------|
| **Tier 0: UIAutomation** | `windows_automation(action='search_controls'/'click')` | Windows 原生应用（微信、Office、资源管理器） | 像素级精确，基于无障碍树，无需坐标 |
| **Tier 1: OCR 文字定位** | `computer_ocr_find(text='目标')` | 非原生界面（浏览器、Electron、游戏）、桌面图标、任务栏 | 像素级精确，EasyOCR 中英文识别 |
| **Tier 2: 窗口查找** | `computer_find_window(title='窗口名')` | 已知窗口标题的程序 | 编程级精确，返回窗口矩形和建议点击坐标 |
| **Tier 3: 视觉网格** | `computer_locate(target='描述')` | 以上方式全部失败时的最后手段 | 网格近似（20×14 格），可能有偏差，需验证 |

**UIA 不可用时的判断标准**：
- `windows_automation(action='list_controls')` 返回空或结果极少
- 应用使用自绘界面（Electron、网页内容、Java Swing、游戏）
- → 自动降级到 Tier 1 OCR

### 12.5 视觉网格定位原理（computer_locate）

```
1. 截取全屏 (Pillow ImageGrab)
2. 限制长边 ≤ 1920px
3. _add_grid_to_image() 绘制 20 列 × 14 行网格：
   - 白色细线（每格，RGBA 70%透明度）
   - 黄色粗线（每 5 格，RGBA 160%透明度）
   - 列号徽章（顶部，黑底白字，18px Arial）
   - 行号徽章（左侧，黑底白字，18px Arial）
4. 保存调试图片 → screenshots/locate_grid.jpg
5. 发送网格截图 + 定位提示词 → GLM-4.1V-Thinking-Flash
6. 视觉模型返回 {"found": true, "col": 10, "row": 3, "description": "..."}
7. 计算: cx = col × cell_w + cell_w // 2,  cy = row × cell_h + cell_h // 2
8. 返回坐标 → 供 computer_click 使用
```

**设计考量**：20×14 配合 18px 字体，即使视觉模型压缩图片到 1024px 宽，数字仍有约 10px，可被清晰辨认。

### 12.6 视觉模型配置

```python
# model_config.py
"computer_vision": {
    "provider": ModelProvider.ZHIPU,
    "model_name": "GLM-4.1V-Thinking-Flash",
    "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    "temperature": 0,  # 定位任务需要确定性输出
}
```

该模型用于两个工具：
- `computer_see_and_describe`: 截图描述（自由文本）
- `computer_locate`: 网格定位（JSON 结构化输出）

### 12.7 依赖管理的三层降级

`computer_tools.py` 采用优雅降级策略，所有关键依赖均为可选：

| 依赖 | 用途 | 不可用时的降级 |
|------|------|---------------|
| **Open Interpreter** (`interpreter`) | 可选的 computer 模块 | 完全降级到 pyautogui + Pillow，功能一致 |
| **pyautogui** | 核心鼠标/键盘控制 | 所有鼠标/键盘工具返回错误信息 |
| **pygetwindow** | 窗口查找/激活 | `computer_find_window` 返回错误 |
| **pyperclip** | 剪贴板操作 | 降级到 PowerShell `Set-Clipboard` |
| **EasyOCR** | 屏幕文字识别 | OCR 工具返回错误；**网络失败时缓存状态**避免反复重试 |

**EasyOCR 失败缓存机制**：首次初始化若因网络问题（WinError 10060/URLError/timeout）下载模型失败，设置 `_easyocr_init_failed = True`，后续调用直接返回降级建议，不再尝试。

### 12.8 中文文本输入方案

这是桌面自动化中的关键难点——pyautogui 的 `typewrite` 仅支持 ASCII 字符。解决方案：

```
computer_paste(text='你好，这是中文消息')
  │
  ├── 方案 A（优先）: pyperclip.copy(text) → pyautogui.hotkey('ctrl', 'v')
  │
  ├── 方案 B（降级）: PowerShell Set-Clipboard → Ctrl+V
  │     $t = @"\n{text}\n"@; Set-Clipboard -Value $t
  │
  └── 方案 C（最终降级）: pyautogui.typewrite() 逐字输入（仅英文）
```

**规范**：Agent 被明确指示「中文 = `computer_paste`，绝不使用 `computer_type`」。这在 `BRAIN_COMPUTER_INSTRUCTIONS` 和 `computer-automation` 技能中均有强调。

### 12.9 与系统其他部分的集成

#### 12.9.1 工具注册（brain.py）

```python
# brain.py line 41
from agent.computer_tools import COMPUTER_TOOLS

# brain.py line 150
tools = [..., browser] + room_tools + COMPUTER_TOOLS
# COMPUTER_TOOLS 包含 windows_automation + 18 个 computer_* 工具
```

所有电脑工具以**平铺方式**注册到 LangGraph Agent，Agent 在每次 LLM 调用时都能看到全部 19 个工具。

#### 12.9.2 提示词注入策略

系统采用**技能加载**而非直接注入的方式传递操作指令：

```
BRAIN_BASE_PROMPT 中仅有一行：
  "桌面应用操作前先 load_skill('computer-automation')"

→ Agent 调用 load_skill('computer-automation')
→ 加载 agent/skills/computer-automation/SKILL.md (258 行)
→ 获得完整的四级定位策略、工具速查表、常见失败模式
```

**为什么不用直接注入**：`prompts.py` 中定义了 `BRAIN_COMPUTER_INSTRUCTIONS`（80 行完整的电脑操作指令），但目前在 `build_brain_system_prompt()` 第 132 行被注释掉。这样可以：
- 节省每次 LLM 调用的 token 消耗（约 2000+ tokens）
- 只在需要桌面操作时才加载指令
- 技能文档比系统提示词更易维护和迭代

#### 12.9.3 HITL 安全策略

```python
# brain.py lines 160-164
interrupt_on={
    # "windows_automation": {"allowed_decisions": ["approve", "reject"]},
    "launch_agent": {"allowed_decisions": ["approve", "reject"]},
    # "computer_execute": {"allowed_decisions": ["approve", "reject"]},
    # "browser": {"allowed_decisions": ["approve", "reject"]},
}
```

**当前状态**：电脑操作工具（`windows_automation`、`computer_execute`）的 HITL 审批已被注释掉，Agent 可自主执行桌面操作。唯一的安全机制是 pyautogui 的 FAILSAFE（鼠标移到屏幕四角触发异常）。

如需启用审批，取消对应行注释即可——每次 `windows_automation` 或 `computer_execute` 调用都会弹出前端审批框。

### 12.10 典型工作流：在微信中发送中文消息

这是 Computer Use 最经典的使用场景，展示了完整的分层定位 + 中文输入流程：

```
方案 A（UIA 精确操控，推荐）：
  ① windows_automation(action='connect', title='微信')
  ② windows_automation(action='search_controls', title='微信', text='输入')
  ③ windows_automation(action='click', title='微信', auto_id='输入框id')
  ④ computer_paste(text='你好，这是测试消息')    ← 中文必须粘贴
  ⑤ computer_key_press(keys='enter')              ← 发送

方案 B（坐标方式，UIA 不可用时降级）：
  ① computer_find_window(window_title='微信')     → 找到并激活窗口
  ② computer_ocr_find(text='输入') 或 computer_click(x, y)
  ③ computer_paste(text='消息内容')
  ④ computer_key_press(keys='enter')
```

### 12.11 安全与防护

| 机制 | 说明 |
|------|------|
| **pyautogui FAILSAFE** | 鼠标移到屏幕四角触发 `pyautogui.FailSafeException`，紧急停止 |
| **computer_execute 超时** | 30 秒超时 + 输出截断 4000 字符，防止无限等待 |
| **HITL（可配置）** | 支持对 `windows_automation` 和 `computer_execute` 启用人工审批，当前为关闭状态 |
| **依赖优雅降级** | 任何依赖缺失都不会导致崩溃，返回明确错误信息引导 Agent 使用替代方案 |
| **OCR 失败缓存** | EasyOCR 初始化失败后不再重试，避免重复网络超时阻塞任务 |

---

## 十三、文件清单

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
| `agent/memory.py` | ChromaDB 记忆存储 + 用户画像按需读取 |
| `agent/conversation_memory_extractor.py` | 对话记忆后台提取 |
| `agent/memory_consolidation.py` | 每日记忆去重整合 |
| `agent/reflection.py` | 任务反思 + 技能生成 |
| `agent/skill_tools.py` | 技能管理工具 |
| `agent/skill_version_manager.py` | 技能版本管理 |
| `agent/sandboxed_backend.py` | Docker 沙箱 |
| `agent/browser_tools.py` | Playwright 浏览器工具 |
| `agent/computer_tools.py` | 19 个桌面自动化工具（Computer Use） |
| `agent/tools.py` | 自定义 LangChain 工具（含 windows_automation） |
| `agent/scheduler.py` | APScheduler 调度 |
| `agent/tasks.py` | 提醒 + 整合任务 |
| `agent/model_config.py` | 多模型配置 |
| `agent/intent.py` | 意图类型枚举 |
| `agent/db.py` | PostgreSQL 连接池 |
| `hub/server.py` | WebSocket Hub |
| `client.py` | FastAPI + super_user 客户端 |
| `client.html` | Web 前端 |
| `agent/skills/user-profile/SKILL.md` | 用户画像按需加载技能 |
| `agent/skills/computer-automation/SKILL.md` | 桌面自动化技能文档 |
| `agent/skills/browser-automation/SKILL.md` | 浏览器自动化技能文档 |
