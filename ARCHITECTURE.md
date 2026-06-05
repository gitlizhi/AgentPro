# AgentPro 技术架构文档

## 一、项目概述

AgentPro 是一个基于 LangGraph + LangChain 的多智能体协作平台。多个 AI Agent 通过 WebSocket Hub 互联，可以私聊、群聊、委派任务、使用技能，并通过 Docker 沙箱执行代码。后端以 DeepSeek V4 为主力模型，PostgreSQL 做状态持久化，ChromaDB 做向量记忆存储。

### 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangGraph + deepagents |
| LLM 后端 | DeepSeek V4（兼容 OpenAI API） |
| 通信 | WebSocket（Hub-Spoke 模型） |
| 状态持久化 | PostgreSQL（checkpoint、rooms、reminders、orchestration、delegation） |
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
| **Context 管理** | `agent/context_manager.py` | Token 预算跟踪、工具输出压缩、上下文中间件 |
| **Communication** | `agent/communication.py` | WebSocket 客户端，与 Hub 通信 |
| **Delegation** | `agent/delegation.py` | TDP 任务委托协议核心（TaskTicket + DelegationManager） |
| **Orchestration** | `agent/orchestration.py` | 多智能体任务编排（OrchestrationPlan + DAG 子任务） |
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
  │     │     ├── SummarizationMiddleware (历史消息摘要)
  │     │     ├── ToolOutputCompactionMiddleware (工具输出压缩)
  │     │     ├── FilesystemMiddleware
  │     │     └── TodoListMiddleware
  │     ├── ContextManager (token 预算 + 工具输出压缩)
  │     ├── Agent 专属上下文 (agent/agent_context/{agent_id}.md)
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

`agent/message_buffer.py` 的 `MessageBuffer` 类实现了延迟合并策略——Agent 接收到其他 Agent 的私聊消息时，不立即处理，而是：

1. 消息累积到缓冲区（按发送者分组）
2. 如果有正在处理中的任务 → 只累积，不打断
3. 如果 timer 在等待 → 取消旧 timer，启动新的 5 秒 timer
4. 5 秒后合并所有累积消息，一次送入 LLM

```python
class MessageBuffer:
    def __init__(self, delay_seconds: float = 5.0)
    def enqueue(self, user_id, user_input, group_context, on_process_callback) -> None
    def cancel(self, user_id) -> None
    def cancel_all(self) -> None
    def is_processing(self, user_id) -> bool
```

这样避免 Agent 对每一条消息都立即回复，减少来回次数。

### 3.4 对话轮次限制

参见 [五、智能体私聊管控](#五智能体私聊管控)。

### 3.5 WebSocket 自动重连

`agent/communication.py` 内置指数退避重连机制，确保网络中断后自动恢复：

```
连接断开 → 等待 1s → 重连 → 失败 → 等待 2s → 重连 → … → 最大 60s
成功连接后重置计数器，持续监听直至 _running=False
```

**关键设计：**
- 接收循环内的非致命错误（JSON 解析失败、单条消息处理异常）不触发重连，仅记录日志并继续
- 只有 `ConnectionClosed` 才退出接收循环触发重连
- `close()` 设置 `_running=False` 后循环自然退出，不会无限重连
- **断连超时自毁**：累计断连超过 `_RECONNECT_MAX_TOTAL`（60 秒）则 `os._exit(1)` 杀死进程，防止僵尸进程长期占用资源。成功重连后计时器归零，短暂闪断不会累积触发

### 3.7 前端消息路由（thread_id 全链路）

Agent 执行任务时产生的所有消息（工具调用进度、审批请求、流式回复）通过 `thread_id` 精确路由到前端对应的会话窗口，防止 Agent A 的消息出现在 Agent B 的对话中。

```
brain.py: _safe_send(text, type, ...)
  → 自动附加 self.thread_id 到 payload
    → comm.send_to_agent(super_user, payload)
      → Hub → client.py (super_user 桥接器)
        → 提取 payload.thread_id，注入所有转发到前端的消息
          → 前端 WS: data.thread_id 路由到正确会话
```

**三层防护机制：**

| 层级 | 文件 | 机制 |
|------|------|------|
| 后端发送 | `brain.py:_safe_send` | 发送给 super_user 时自动附加 `self.thread_id` |
| Hub 桥接 | `client.py:connect_to_hub` | 从 hub payload 提取 `thread_id`，注入所有转发消息（message、tool_call_start/end、approval_request） |
| 前端过滤 | `client.html` | 消息按 `data.thread_id` 路由到对应会话缓存；`tool_call_start/end` 和 `approval_request` 仅当 `data.from === activeAgent` 时处理；`selectAgent()` 立即置空 `activeThreadId` 防止竞态 |

**前端路由优先级**：
1. 显式 `thread_id`（来自后端，权威路由）
2. 当前活跃会话的 `activeThreadId`（用户正在查看）
3. 该 Agent 的首个会话（fallback，兼容旧消息）

### 3.6 上下文管理（Context Management）

上下文管理系统借鉴了 Claude Code 的分层上下文模型，将系统指令、动态上下文、用户输入清晰地分离到不同的消息角色中，并通过主动的 token 预算跟踪和工具输出压缩来维持上下文窗口的健康状态。

**核心理念**：系统级信息走 `SystemMessage`，用户输入走 `HumanMessage`，角色边界清晰——LLM 能够区分"指令"和"输入"。

#### 3.6.1 上下文分层模型

```
┌──────────────────────────────────────────────┐
│  Layer 1: 静态系统提示词 (System Prompt)       │
│  - BRAIN_BASE_PROMPT（角色、工具、规则）       │
│  - BRAIN_REFLECTION_GUIDE（反思指南）          │
│  - BRAIN_DESKTOP_INSTRUCTIONS（桌面指令）      │
│  - Agent 专属上下文文件（*.md）                 │
├──────────────────────────────────────────────┤
│  Layer 2: 动态上下文 (SystemMessage)           │
│  - 对话对象信息（人类/其他 Agent）              │
│  - 对话轮次警告（ConversationTracker）         │
│  - 群聊上下文（房间 ID、成员列表、规则）        │
│  - 图片描述（视觉模型输出）                     │
│  - 相关技能经验（ChromaDB 检索注入）            │
├──────────────────────────────────────────────┤
│  Layer 3: 用户消息 (HumanMessage)             │
│  - 仅包含用户实际输入的内容                     │
│  - 不再混入任何系统级上下文前缀或后缀            │
├──────────────────────────────────────────────┤
│  Layer 4: 工具输出后处理                       │
│  - ToolOutputCompactionMiddleware            │
│  - 超过 2000 字符的工具结果自动压缩              │
│  - 完整输出保存到 agent/agent_temp/tool_outputs/ │
└──────────────────────────────────────────────┘
```

**对比旧方案**：

| 方面 | 旧方案（注入式） | 新方案（分层式） |
|------|-----------------|-------------------|
| 系统上下文位置 | 拼接到 HumanMessage 前缀 | 独立的 SystemMessage |
| 技能经验注入 | 拼接到 HumanMessage 后缀 | 追加到 SystemMessage 列表 |
| Token 预算感知 | 无 | 主动跟踪，80% 警告，85% 触发压缩 |
| 工具输出控制 | 无全局策略 | 自动压缩超长输出（>2000 字符） |
| Agent 上下文文件 | 无 | `agent/agent_context/{agent_id}.md` |

#### 3.6.2 消息构建流程

```
Brain._handle_with_agent() / _process_agent_message()
  │
  ├── 1. _build_system_contexts(image_data)
  │     ├── 对话对象信息（人类/Agent + 在线状态）
  │     ├── 对话轮次警告（如有）
  │     ├── 群聊上下文（如有）
  │     └── 图片描述（如有）
  │
  ├── 2. 追加技能经验（super_user 路径）
  │     └── _get_relevant_skill_lessons() → ChromaDB 语义检索
  │
  ├── 3. 构建消息列表
  │     ├── {"role": "system", "content": ctx1}
  │     ├── {"role": "system", "content": ctx2}
  │     └── {"role": "user", "content": user_input}  ← 纯净的用户输入
  │
  └── 4. Token 预算检查（仅 super_user）
        └── context_manager.check_budget() → 超出 80% 时通知用户
```

#### 3.6.3 Token 预算管理

`agent/context_manager.py` 中的 `TokenBudget` 类使用字符计数启发式估算 token 用量：

| 参数 | 默认值     | 说明                                   |
|------|---------|--------------------------------------|
| `DEFAULT_TOKEN_BUDGET` | 625,000 | 总预算（DeepSeek V4 上下文窗口 1M，保守预算）       |
| `CHARS_PER_TOKEN` | 2.8     | 字符/token 比率（中英文混合的保守估计）              |
| `WARN_THRESHOLD` | 80%     | 超过此比例通过 `_safe_send` 提醒用户            |
| `COMPACT_THRESHOLD` | 85%     | 超过此比例触发 SummarizationMiddleware 提前压缩 |

预算检查在每次 `_handle_with_agent` 调用时执行（仅在 super_user 路径），一次性提醒，不在每次流事件中重复。

#### 3.6.4 工具输出压缩

`ToolOutputCompactionMiddleware` 作为 LangChain AgentMiddleware 注册在中间件链中，在**每次工具执行后**拦截 `ToolMessage`，检查内容长度：

```
工具执行 → handler(request) 返回 ToolMessage
  → ToolOutputCompactionMiddleware.awrap_tool_call
    → content > 2000 字符？
      ├── 否 → 原样返回
      └── 是 → 压缩处理：
            ├── 保留前 2000 字符
            ├── 保留末尾 300 字符（通常含状态/错误信息）
            ├── 中间标注省略字符数
            ├── 完整输出写入 agent/agent_temp/tool_outputs/{tool}_{hash}.txt
            └── 返回压缩后的 ToolMessage
```

**压缩格式示例**：
```
[前 2000 字符内容...]

[... 中间省略 15234 字符 ...]

[末尾 300 字符内容...]

[完整输出已保存至 agent/agent_temp/tool_outputs/tavily_search_a1b2c3d4.txt，共 17534 字符]
```

#### 3.6.5 Agent 专属上下文文件

类似 Claude Code 的 `CLAUDE.md`，每个 Agent 可以在 `agent/agent_context/{agent_id}.md` 中定义专属的持久化上下文：

- **加载时机**：Agent 启动时（`Brain.__init__`）
- **注入位置**：静态系统提示词末尾（`_build_system_prompt` 方法）
- **格式**：自由 Markdown，建议包含角色定义、工作原则、约束条件等
- **缺失处理**：文件不存在时静默跳过，不影响正常启动

**示例**（`agent/agent_context/agent_main.md`）：
```markdown
## 角色
你是 AgentPro 平台的主智能体，负责协调其他子智能体完成复杂任务。

## 工作原则
- 优先复用已有技能，避免重复造轮子
- 复杂任务优先考虑委派给子智能体
- 保持回复简洁专业
```

**注意事项**：上下文文件的内容会在**每一轮对话**中都占用 token，应保持精简。如需大量操作指南，应使用技能系统（`load_skill`）按需加载。

#### 3.6.6 上下文管理的中间件链

`Brain.__init__` 中注册的中间件按顺序执行：

```
1. SummarizationMiddleware
   - 触发条件：历史 token > 20,000
   - 保留策略：最近 30 条消息 + 保持 AI/Tool 消息对完整性
   - 行为：旧消息替换为摘要 HumanMessage

2. ToolOutputCompactionMiddleware
   - 触发条件：每次工具调用后
   - 保留策略：>2000 字符的工具输出压缩为头+尾+引用
   - 行为：返回压缩后的 ToolMessage
```

两条中间件互补：`SummarizationMiddleware` 处理**历史消息的总量控制**，`ToolOutputCompactionMiddleware` 处理**单条工具输出的粒度控制**。

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

### 5.2 主方案：Task Delegation Protocol (TDP)

`agent/delegation.py` 引入了**任务委托协议**，将智能体私聊从"自由文本聊天"升级为"结构化工单管理"。

#### 核心概念

每段智能体私聊必须绑定一个 `TaskTicket`（任务工单），工单有明确的生命周期、交付标准和多重自动终止条件。

#### 工单状态机

```
┌─────────┐  accept   ┌──────────┐   工作    ┌─────────────┐  deliver  ┌────────┐
│ PENDING  │─────────→│ ACCEPTED │────────→│ IN_PROGRESS  │─────────→│ CLOSED │
└─────────┘           └──────────┘          └─────────────┘           └────────┘
     │                     │                      │                      ▲
     └→ DECLINED           └→ NEGOTIATING         └→ TIMED_OUT ─────────┘
                            (最多2轮澄清)            (超时自动关闭)
```

#### 委托工具集

`agent/tools_factory.py:create_delegation_tools()` 提供7个结构化工具：

| 工具 | 用途 | 计入轮次 |
|------|------|---------|
| `delegate_task(agent, description, expected_output, max_rounds)` | 创建工单并发送委托 | - |
| `accept_task(ticket_id)` / `decline_task(ticket_id, reason)` | 响应委托（握手） | 否 |
| `deliver_result(ticket_id, summary)` | 交付最终结果 → CLOSED | - |
| `request_clarification(ticket_id, question)` | 澄清需求（最多2轮） | 是 |
| `cancel_task(ticket_id, reason)` | 任一方取消 | 否 |
| `report_progress(ticket_id, status)` | 可选进度汇报 | 否 |

#### 多重自动终止条件

| 条件 | 机制 |
|------|------|
| 显式交付 | `deliver_result` 调用 → CLOSED |
| 轮次耗尽 | `record_round()` 达 max_rounds → TIMED_OUT |
| 空闲超时 | `_ticket_timeout_loop` 检测 10 分钟无活动 → TIMED_OUT |
| 语义停滞 | `_should_terminate_conversation` 检测重复循环 → CANCELLED |
| 用户介入 | super_user 可随时取消任意工单 |

#### 路由层门控

`core.py:_handle_message` 增加三层门控：
1. TDP 协议消息（`_tdp` 字段）直接放行
2. 有活跃工单 → 允许（受工单轮次预算约束）
3. 无活跃工单 → 走旧 `ConversationTracker` 兜底

#### 关键文件

| 文件 | 职责 |
|------|------|
| `agent/delegation.py` | TaskTicket、TicketState、DelegationManager 核心逻辑 |
| `agent/tools_factory.py` | `create_delegation_tools` + 改造后的 `send_to_agent` |
| `agent/brain.py` | 集成 DelegationManager，工单上下文注入，超时循环 |
| `agent/core.py` | 消息路由层 TDP 门控 |

### 5.3 兜底方案：ConversationTracker

`agent/conversation_tracker.py` 保留作为兜底机制，在无活跃 TDP 工单时提供基础的轮次计数和硬截断保护。

#### 生命周期

```
新会话开始（tracker 无状态）
  → 生成新 thread_id（隔离的 LangGraph checkpoint）
  → 使用同一个 thread_id 直到会话结束（保证上下文连续）

轮次 1-3: 正常通信，无干预
轮次 4-5: 系统提示注入 "请在 X 轮内完成交流"
轮次 6-7: 最后通牒 "还剩 2 轮 / 最后 1 轮"
轮次 8:   硬截断 + 30 分钟冷却 + 通知 super_user

会话重置触发条件：
  - 用户发送消息给任一参与方（自动解除所有限制）
  - 智能体发送 [停止交流] 标记
  - 30 分钟冷却到期
```

#### TDP vs ConversationTracker 对比

| 维度 | TDP | ConversationTracker |
|------|-----|---------------------|
| 控制模型 | 工单生命周期 + 状态机 | 纯轮次计数 |
| 任务结构 | 结构化委托（desc/expected/max_rounds） | 自由文本聊天 |
| 完成信号 | 显式 `deliver_result` | LLM 自觉发 `[停止交流]` |
| 无工单私聊 | 允许（兼容旧行为） | 允许（仅轮次限制） |
| 用户可见性 | 所有工单状态通知 super_user | 仅截断时通知 |
| 超时检测 | 10 分钟空闲自动超时 | 无 |

---

### 5.4 多智能体任务编排 (Orchestration)

在 TDP 单任务委托之上，增加了**任务分解、并行派发、进度聚合、自动汇总**的编排层，使主 Agent 能协调多个子 Agent 共同完成复杂任务，并在全部完成后主动向用户呈现最终报告。

#### 5.4.1 核心概念

`agent/orchestration.py` 定义了编排计划（`OrchestrationPlan`），将一个复杂任务分解为多个有向无环图（DAG）结构的子任务（`SubTask`），自动派发给合适的智能体并行执行，所有子任务完成后由 orchestrator 自动汇总并向用户呈现最终结果。

**角色分工：**

```
Orchestrator (agent_main)          Workers (搜索智能体 / 分析智能体)
┌─────────────────────┐           ┌──────────────────────┐
│ 创建编排计划         │           │ 接受子任务            │
│ 分解为 DAG 子任务    │           │ 执行具体工作           │
│ 并行派发子任务       │──TDP工单──→│ 交付分析结果           │
│ 监控进度（被动）     │←──交付────│                      │
│ 派发下一批子任务     │           └──────────────────────┘
│ ★自动汇总最终报告    │
└─────────────────────┘
```

**关键原则**：Orchestrator 只委派数据收集、分析等具体工作。最终汇总并向用户呈现报告是 orchestrator 自己的职责，不委派给 worker。

#### 5.4.2 子任务状态机

```
PENDING ──dispatch──→ DISPATCHED ──accepted──→ IN_PROGRESS ──deliver──→ COMPLETED
   │                     │                         │
   └─────────────────────┴─── (fail/reassign) ────→ FAILED
```

每个 `SubTask` 包含字段：`id`（如 `st_1`）、`description`、`assigned_to`、`status`、`depends_on`（DAG 依赖列表）、`result`（交付摘要）、`ticket_id`（TDP 工单引用）、`suggested_role`（角色提示）。

#### 5.4.3 计划状态机

```
PLANNING → READY → EXECUTING → COMPLETED / PARTIALLY_COMPLETED / FAILED
                                  ↓
                              CANCELLED（用户取消）
```

- `PLANNING`：空计划，子任务未填充
- `READY`：子任务已填充，等待派发
- `EXECUTING`：至少一个子任务已派发
- `COMPLETED`：全部子任务成功
- `PARTIALLY_COMPLETED`：部分成功、部分失败
- `FAILED`：全部失败

#### 5.4.4 编排工具集

`agent/tools_factory.py:create_orchestration_tools()` 提供 4 个工具：

| 工具 | 用途 | 关键行为 |
|------|------|---------|
| `create_task_plan(description)` | LLM 分析复杂任务，生成 2-5 个子任务，含 DAG 依赖和建议角色 | 一次性 JSON 返回完整计划，避免多轮交互 |
| `dispatch_subtasks(plan_id)` | 并行派发所有就绪子任务，自动匹配在线智能体 | 每个子任务创建 TDP 工单，携带 `_orchestration` 和 `orchestration_plan_id` |
| `check_plan_progress(plan_id)` | 查看计划整体进度 | **仅限用户主动询问时使用**，含 20 秒防抖 |
| `reassign_subtask(plan_id, subtask_id, new_agent)` | 失败子任务重新分配 | 重置子任务状态，创建新工单 |

#### 5.4.5 编排完整生命周期

这是编排系统最重要的设计——一个复杂任务从用户输入到最终报告的完整数据流：

```
阶段 1: 任务分解
─────────────────
用户: "深度分析2020-2025年美国经济"
  → agent_main 调用 create_task_plan()
    → LLM 分解为 3 个子任务（prompts.py:build_task_decomposition_prompt）:
        st_1: 收集经济数据       (无依赖)
        st_2: 事件与政策分析      (无依赖，可与 st_1 并行)
        st_3: 深度综合分析        (依赖 st_1, st_2)
    → OrchestrationManager.create_plan() → 状态=READY
    → 依赖验证: 移除 LLM 幻觉的不存在引用（如 st_0）
    → 展示计划给用户审阅

阶段 2: 并行派发
─────────────────
agent_main 调用 dispatch_subtasks(plan_id)
  → plan.get_ready_subtasks() 返回 st_1, st_2（无未满足依赖）
  → 按 suggested_role 模糊匹配在线智能体:
      st_1 → 搜索智能体 (suggested_role: "搜索专家")
      st_2 → 分析智能体 (suggested_role: "数据分析师")
  → 每个子任务:
      ├── DelegationManager.create_ticket(allow_parallel=True)
      │     └── 跳过 pair-key 冲突检查，允许多个子任务派发给同一智能体
      ├── ticket.orchestration_plan_id = plan_id  ★ 关键：ticket 记住自己的 plan
      ├── dm._save_ticket(ticket)  → PostgreSQL 持久化
      ├── om.dispatch_subtask()  → _ticket_index[ticket_id] = (plan_id, subtask_id)
      └── comm.send_to_agent(worker, {
            "_tdp": "delegation",
            "_orchestration": plan_id,          ← 消息级传递（第一层）
            "ticket_id": ticket.ticket_id,
            "max_rounds": 12,
            "expected_output": "..."
          })
  → plan.state = EXECUTING
  → 返回: "你无需轮询进度，子任务完成后系统自动通知"

阶段 3: Worker 侧
─────────────────
Worker 收到委托消息:
  → core.py Gate 1: TDP 协议消息（_tdp="delegation"）→ 放行
  → brain.py delegation handler:
      ├── 创建镜像工单（assignee 侧视角）
      ├── ticket.orchestration_plan_id = tdp.get("_orchestration")  ★ 从消息提取
      └── dm._save_ticket(ticket)
  → LLM 决策: 调用 accept_task(ticket_id)
      ├── 状态转换: PENDING → ACCEPTED
      └── 自动携带 _orchestration（从 ticket.orchestration_plan_id 读取）

Worker 执行任务:
  → 搜索/分析/写文件
  → 进度消息（"正在搜索..."）→ 被 orchestrator Gate 3 静默拦截

Worker 调用 deliver_result(ticket_id, summary):
  → delegation.py: deliver_result():
      ├── ACCEPTED → IN_PROGRESS（自动桥接）
      ├── IN_PROGRESS → CLOSED
      └── 记录 result_summary
  → tools_factory.py: 发送交付消息:
      {
        "_tdp": "delivery",
        "ticket_id": ticket_id,
        "_orchestration": ticket.orchestration_plan_id,  ★ 从 ticket 读取
        "text": "[TDP 交付] ..."
      }

阶段 4: Orchestrator 侧 — 程序化交付处理
─────────────────────────────────────────
Orchestrator 收到交付消息:
  → core.py Gate 1: TDP 协议消息 → 放行
  → brain.py delivery handler（_handle_with_agent）:

      1. 关闭工单:
         ticket.state = CLOSED, ticket.result_summary = summary

      2. ★ 三级回退获取 plan_id:
         ① tdp.get("_orchestration", "")           ← 消息字段（worker 传播）
         ② getattr(ticket, 'orchestration_plan_id') ← ticket 字段（直接存储）
         ③ _ticket_index.get(ticket_id)             ← 反查索引（最后防线）

      3. om.mark_completed(ticket_id, summary)
         → SubTask.status = COMPLETED, plan.completed_count++

      4. 判断计划状态:
         ├── plan.is_complete() = True:
         │     ├── 发送 "🎯 编排计划全部完成！" 通知 super_user
         │     └── ★ schedule_background_task(_synthesize_plan_results)
         │           → 等 1.5s（释放 process_lock）
         │           → 收集所有 subtask.result
         │           → 构建汇总 prompt → self.process() → LLM 自动汇总
         │           → 最终报告呈现给用户
         │
         └── plan.is_complete() = False:
               ├── plan.get_ready_subtasks() 获取依赖已满足的下一批
               ├── _dispatch_ready_subtasks(plan, ready) → 自动派发
               ├── 通知 super_user: "子任务 X 已交付，自动派发下一批: st_3 → 分析智能体"
               └── 无需 LLM 参与，纯程序化推进

      5. 设置 _skip_llm_for_issuer = True
         → 编排 TDP 消息不唤醒 LLM，避免身份混淆

阶段 5: 自动汇总（最终呈现）
─────────────────────────────
_synthesize_plan_results(plan_id) 后台任务:
  → 等待当前 process() 释放 _process_lock
  → 收集所有子任务结果:
      "### st_1: 数据收集 (执行者: 搜索智能体)\n**交付摘要**: ..."
      "### st_2: 事件分析 (执行者: 分析智能体)\n**交付摘要**: ..."
      "### st_3: 深度分析 (执行者: 分析智能体)\n**交付摘要**: ..."
  → 构建汇总 prompt:
      "你发起的编排任务已全部完成。以下是各子任务交付结果。请整合并向用户呈现最终报告。"
  → self.process(user_id="super_user", ...)
  → LLM 生成完整报告 → 流式发送给用户
```

#### 5.4.6 设计原则：Orchestrator ↔ Worker 是 API 调用，不是聊天

编排通信的核心设计哲学：**orchestrator 和 worker 之间的 TDP 消息是程序化信号，不是 LLM 对话**。

**问题**：早期设计中，orchestrator 的 LLM 会"看到"worker 的 TDP 消息，导致：
- 角色混淆（orchestrator 误以为自己是 worker）
- 身份混乱（"我是搜索智能体"）
- 无意义回复（"好的，我来处理"）
- 关键交付被埋没在 LLM 的聊天历史中

**解决方案——多层过滤体系**：

```
Orchestrator (agent_main) 的消息处理路径:

收到消息
  │
  ├── core.py Gate 1: _tdp 字段存在？
  │     ├── 是 → 构建 tdp_notification，跳过 Gate 2/3
  │     └── 否 → Gate 2/3 检查
  │
  ├── brain.py _handle_with_agent:
  │     │
  │     ├── TDP 协议消息处理（程序化，不涉及 LLM）:
  │     │   ├── "delegation" → 创建镜像工单
  │     │   ├── "delivery"   → 关闭工单 + 编排回调（mark_completed + 派发/汇总）
  │     │   ├── "acceptance" → 状态转换 + 编排回调（mark_accepted）
  │     │   ├── "decline"    → 编排回调（mark_failed）
  │     │   └── "progress"   → 静默丢弃（orchestrator 不需要看进度）
  │     │
  │     ├── _skip_llm_for_issuer 检查:
  │     │   └── True → return（不调用 LLM）
  │     │
  │     └── _is_orchestrating 检查:
  │         └── True 且非被委托方 → return（编排模式下禁止 LLM 处理 agent-to-agent 消息）
  │
  └── 只有非编排的普通消息才进入 LLM
```

**`_orchestration` 字段的多层保障**：

编排信息（plan_id）通过三层机制传递，确保 worker 的交付消息一定能被 orchestrator 关联到正确的计划：

| 层级 | 机制 | 存储位置 | 作用 |
|------|------|---------|------|
| 1 | 消息 payload 的 `_orchestration` 字段 | WebSocket 消息 | worker 的 `deliver_result` 从 `ticket.orchestration_plan_id` 读取并传播 |
| 2 | `TaskTicket.orchestration_plan_id` | 数据库 + 内存 | ticket 创建时写入，进程重启后可恢复 |
| 3 | `OrchestrationManager._ticket_index` | 内存（重启后从 DB 重建） | `ticket_id → (plan_id, subtask_id)` 反查，不依赖任何消息字段 |

> **设计动机**：早期版本仅依赖消息 payload 的 `_orchestration` 字段（层级 1），但 worker 的 `deliver_result` 工具没有传播该字段，导致 orchestrator 收到交付后无法关联计划——整个编排回调被跳过。现改为三层保障，ticket 本身存储 `orchestration_plan_id`（层级 2），`_ticket_index` 作为最后防线（层级 3），即使前两层都丢失也能正确路由。

#### 5.4.7 关键设计决策详解

**DAG 依赖模型**：
- 子任务通过 `depends_on` 字段声明依赖关系（如 `st_3` 依赖 `st_1` 和 `st_2`）
- `get_ready_subtasks()` 返回所有依赖已满足（前序任务状态为 COMPLETED）的 PENDING 子任务
- LLM 返回的依赖序号自动规范化为 `"st_N"` 格式（防御整数/字符串差异）
- **依赖验证**：`set_subtasks()` 自动移除对不存在子任务的引用（防御 LLM 幻觉如 `st_0`）

**`allow_parallel` — 并行派发到同一智能体**：
- 普通 TDP 委托中，同一对智能体之间只能有一个活跃工单（pair-key 机制）
- 编排场景下，多个子任务可能需要派发给同一个 worker
- `create_ticket(allow_parallel=True)` 跳过 pair-key 冲突检查
- 编排子任务使用 12 轮预算（高于普通委托的 8 轮），因为编排任务可能更复杂

**防轮询设计**：
- `dispatch_subtasks` 返回值明确告知 "你无需轮询进度，子任务完成后系统自动通知"
- `check_plan_progress` 工具描述开头注明 "仅应在用户明确询问时使用"
- `get_progress_summary()` 内置 20 秒防抖：同一 plan 在 20 秒内重复查询且进度无变化时，返回警告而非正常结果

**`_is_orchestrating` 守卫**：
- 检查当前 agent 是否还有活跃的编排计划（state 为 READY 或 EXECUTING）
- 编排模式下，非 TDP 且非工单的消息直接丢弃，不唤醒 LLM
- 这防止了 worker 的进度消息、闲聊等触发 orchestrator 的不必要 LLM 调用

**`_synthesize_plan_results` — 自动汇总**：
- 计划全部完成后，通过后台任务触发 LLM 汇总
- 等待 1.5 秒确保当前 `process()` 释放 `_process_lock`
- 将所有子任务的 `result` 字段注入 LLM 上下文
- LLM 自然生成最终报告并呈现给用户
- 若用户在此过程中发送新消息，旧任务被取消，新消息优先

**任务分解原则（prompt 层）**：
- `build_task_decomposition_prompt` 明确告知 LLM：
  - "你是 orchestrator，只委派数据收集、分析等具体执行工作"
  - "不要创建'撰写最终报告'或'向用户汇报'类子任务"
  - "worker 交付分析结果后，系统会自动通知你整合并向用户呈现"
- 这是第一道防线：从源头避免 orchestrator 将"最终呈现"委派出去

#### 5.4.8 智能体匹配策略

`dispatch_subtasks` 和 `_dispatch_ready_subtasks` 根据 `suggested_role` 字段模糊匹配在线智能体：

1. 子串匹配：`role_hint in agent_id.lower()`（如 "搜索" 匹配 "搜索智能体"）
2. 无匹配时选第一个非自身的在线智能体
3. 无在线智能体时返回提示，稍后手动重试

**排除规则**：
- `super_user`（人类用户）不可作为委派目标
- `reminder_bot`（系统机器人）不可作为委派目标
- orchestrator 自身不可作为委派目标（`send_to_agent` 工具阻止自发送）

#### 5.4.9 与 TDP 的集成

| 层级 | 组件 | 职责 |
|------|------|------|
| `OrchestrationPlan` / `SubTask` | `orchestration.py` | 计划生命周期、子任务 DAG、状态追踪 |
| `OrchestrationManager` | `orchestration.py` | 计划 CRUD、ticket↔subtask 反查索引、进度格式化、防抖 |
| `TaskTicket` | `delegation.py` | 单工单生命周期、状态机、轮次控制、超时检测 |
| `DelegationManager` | `delegation.py` | 工单 CRUD、pair-key 管理、`allow_parallel` 支持 |
| `_ticket_index` | `orchestration.py` | `ticket_id → (plan_id, subtask_id)` 反查，编排回调的基石 |
| `orchestration_plan_id` | `delegation.py` | ticket 上的一等字段，连接工单和编排计划 |
| TDP 工具 | `tools_factory.py` | `deliver_result`/`accept_task`/`decline_task` 自动传播 `_orchestration` |
| 编排工具 | `tools_factory.py` | `create_task_plan`/`dispatch_subtasks`/`check_plan_progress`/`reassign_subtask` |
| 交付处理程序 | `brain.py` | 三级回退获取 plan_id → mark_completed → 派发/汇总 |
| 接受处理程序 | `brain.py` | 编排工单不提前取消（保留到交付完成） |
| Gate 系统 | `core.py` | 三层门控：TDP 协议→活跃工单→编排守卫 |
| `_synthesize_plan_results` | `brain.py` | 计划完成后触发 LLM 自动汇总 |
| `_dispatch_ready_subtasks` | `brain.py` | 子任务交付后自动派发下一批（程序化，无 LLM） |

#### 5.4.10 数据库持久化

编排计划和委托工单的状态持久化到 PostgreSQL，进程重启后自动恢复。

**表结构：**

| 表 | 核心字段 | 恢复条件 |
|---|---------|---------|
| `orchestration_plans` | `plan_id`, `description`, `issuer`, `state`, `completed_count`, `failed_count`, `created_at`, `updated_at` | `state NOT IN ('completed', 'partially_completed', 'failed', 'cancelled')` |
| `orchestration_subtasks` | `plan_id`, `subtask_id`, `description`, `assigned_to`, `status`, `depends_on`(JSONB), `result`, `ticket_id`, `suggested_role` | 随计划级联恢复 |
| `delegation_tickets` | `ticket_id`, `issuer`, `assignee`, `state`, `round_count`, …, `orchestration_plan_id` | `state NOT IN ('closed', 'declined', 'timed_out', 'cancelled')` |

**保存时机：**

| 管理器 | 触发操作 | 保存方式 |
|--------|---------|---------|
| `OrchestrationManager` | `create_plan`, `set_subtasks`, `dispatch_subtask`, `mark_accepted`, `mark_completed`, `mark_failed`, `cancel_plan` | `await _save_plan()` — 同步等待，操作频率低 |
| `DelegationManager` | `create_ticket`, `transition`, `record_round`, `record_clarification` | `asyncio.create_task(_do())` — fire-and-forget，操作频率高，异步避免阻塞 |

**恢复流程：**

```
Agent.run()
  → brain.recover_state()  ← 在 connect() 之前执行
    → dm.load_active()     ← 从 delegation_tickets 恢复非终态工单
      → 重建 _by_id、_by_pair 内存索引
      → 恢复 orchestration_plan_id（连接 ticket 到 plan）
    → om.load_active()     ← 从 orchestration_plans/subtasks 恢复非终态计划
      → 重建 _ticket_index（从 persisted ticket_id 字段）
      → 规范化 depends_on（防御历史数据中的整数格式）
  → 如有活跃工单 → 启动超时检测循环
```

**设计决策：**
- 子任务使用 `DELETE + INSERT` 而非逐条 `UPDATE`——编排计划通常只有 2-5 个子任务，性能影响可忽略，逻辑更简单
- 不保存/不恢复终态数据——终态计划和工单不参与恢复，避免无用数据堆积
- `orchestration_plan_id` 在 ticket 上持久化——进程重启后 ticket 仍能找到所属 plan

#### 5.4.11 Orchestration vs 普通 TDP 委托

| 维度 | 普通 TDP 委托 | 编排 |
|------|-------------|------|
| 任务数量 | 1 对 1 | 1 对 N（DAG 并行） |
| 分解方式 | 用户手动指定 | LLM 自动分解 |
| 依赖管理 | 无 | DAG 拓扑排序 |
| 进度聚合 | 无 | 自动追踪 + 防抖查询 |
| 结果汇总 | 委托方 LLM 审查 | 自动汇总 + LLM 呈现 |
| 通信模式 | LLM 对话 | 程序化信号（不经过 LLM） |
| 工单轮次 | 8 轮 | 12 轮（更复杂） |
| pair-key | 一对一互斥 | `allow_parallel` 允许多工单 |
| orchestrator LLM | 参与审查 | **不参与**（程序化推进，最终汇总除外） |

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
| 任务可中断（Stop Task + 消息覆盖） | 通过 asyncio Task 取消 + agent_status 协议实现；支持手动停止按钮和发送新消息自动中断两种方式 |
| 新消息自动中断当前任务 | 用户发送新消息时，若 Agent 正忙，自动取消当前任务并以新消息开始，保持同一 thread_id 保证上下文连续 |
| astream 超时保护（ensure_future+wait） | Python 3.12 的 `asyncio.wait_for` 会将任务内部的 `CancelledError`（如 LLM HTTP 层抛出）错误掩码为 `TimeoutError`，导致 Agent 误判超时而崩溃。改用 `ensure_future`+`asyncio.wait` 组合，正确保持 `CancelledError` 的传播语义 |
| LLM 流处理中的网络异常捕获 | `_handle_with_agent` 和 `_process_agent_message` 原只捕获 `TimeoutError`/`CancelledError`，但 `httpx.ConnectError` 等网络异常直接使 Agent 崩溃。新增 `except (OSError, ConnectionError)` 兜底，确保网络中断时 Agent 优雅降级 |
| 浏览器多层反检测体系 | 单层反检测（如仅修改 navigator.webdriver）已无法应对现代检测。采用纵深防御：Chrome 启动参数 + 真实 UA + playwright-stealth JS 注入 + CDP 连接系统 Chrome（TLS 指纹匹配），逐层加码直至通过 |
| `_safe_send` 统一 WebSocket 发送 | 原有代码中多处裸调 `send_to_agent`，一次 WebSocket 异常即可中断整个 astream 流或抑制 CancelledError。`_safe_send` 将异常降级为日志警告，适用于通知类消息（状态、错误提示、工具进度）；关键消息（审批请求）仍独立 try/except |
| 工具进度双通道架构 | 工具调用信息通过两条独立的通道到达前端：① `tool_call_start`/`tool_call_end` 结构化事件 → 输入框上方状态条（实时进度）；② 同一条消息的 text 字段 → 聊天记录中的可折叠条目（历史回溯）。两通道在 `client.py` 中通过 `payload.type` 分流，互不干扰 |
| 系统上下文 SystemMessage 而非 HumanMessage | 借鉴 Claude Code 的分层上下文模型。动态上下文（对话对象、轮次警告、群聊信息、图片描述、技能经验）作为 `SystemMessage` 注入，用户消息仅包含用户实际输入。LLM 能清晰区分"指令"和"输入"，避免角色边界模糊 |
| Token 预算主动跟踪 | 使用字符计数启发式估计 token 用量（保守比率 2.8 字符/token），在 80% 时主动通知用户，85% 时触发 SummarizationMiddleware 提前压缩。相比纯被动等待中间件触发（20K token），主动预算让用户有心理预期 |
| 工具输出自动压缩 | 单条工具输出超 2000 字符时自动压缩为头+尾+省略标记+文件引用的格式。防止单次大结果（如网页抓取 50KB HTML）撑爆上下文窗口。完整输出保存到临时文件供后续查阅 |
| Agent 专属上下文文件（CLAUDE.md 模式） | 每个 Agent 在 `agent/agent_context/{agent_id}.md` 中定义持久化上下文，启动时注入系统提示词。内容需精简（每轮都消耗 token），大量操作指南应使用技能系统按需加载 |
| agent_status 缓存恢复 | `client.py` 维护 `agent_statuses` 缓存，新前端连接时补发所有 Agent 状态，解决页面刷新后停止按钮消失的问题。Agent 离线时清理缓存避免过期数据 |
| 前端 thread_id 路由 + 事件过滤 | 消息通过 `thread_id` 全链路（brain→client.py→前端）精确路由到对应会话窗口；`tool_call_start/end` 和 `approval_request` 按 `data.from` 过滤，仅当前活跃 Agent 的事件才更新 UI，彻底解决跨 Agent 消息串扰 |
| Agent 断连超时自毁 | `communication.py` 累计断连超过 60 秒直接 `os._exit(1)`，防止网络彻底中断时 Agent 进程无限挂起重试成为僵尸进程 |
| 多智能体任务编排（DAG 并行派发） | 在 TDP 单工单之上增加编排层。LLM 一次性分解复杂任务为 2-5 个子任务（DAG），按拓扑依赖并行派发给在线智能体，子任务交付后自动派发下一批，全部完成后自动汇总向用户呈现最终报告 |
| 编排通信：程序化信号而非 LLM 对话 | Orchestrator↔Worker 的 TDP 消息（delegation/delivery/acceptance）在 brain.py 中程序化处理，不唤醒 LLM。通过 `_skip_llm_for_issuer` + `_is_orchestrating` 守卫防止 LLM 参与编排流程，彻底消除角色混淆和身份混乱 |
| `_orchestration` 三层保障 | 编排信息（plan_id）通过三层机制传递：① 消息 payload 字段 ② `TaskTicket.orchestration_plan_id` 持久化字段 ③ `_ticket_index` 内存反查索引。即使前两层丢失也能正确路由交付到对应 plan |
| 编排结果的自动汇总 | 计划全部完成后，`_synthesize_plan_results` 后台任务收集所有子任务结果，触发 orchestrator LLM 自动汇总并向用户呈现最终报告。Orchestrator 只委派数据收集分析，最终呈现保留给自己 |
| 编排与委托的数据库持久化 | OrchestrationManager 和 DelegationManager 的状态变更实时写入 PostgreSQL（`orchestration_plans`、`orchestration_subtasks`、`delegation_tickets` 三张表）。Agent 启动时 `recover_state()` 恢复所有活跃工单和计划，重建内存索引。OrchestrationManager 同步等待持久化，DelegationManager 异步 fire-and-forget |
| 防轮询与防抖 | `dispatch_subtasks` 返回值明确告知 "无需轮询"，`check_plan_progress` 内置 20 秒防抖——进度无变化时返回警告而非正常结果，防止 LLM 形成轮询正反馈 |
| 依赖验证与 LLM 幻觉防御 | `set_subtasks()` 自动移除对不存在子任务的引用（如 `st_0`），`depends_on` 自动规范化为 `"st_N"` 格式。prompt 层禁止使用序号 0 |

### 11.3 工具调用进度流

```
brain.py: _handle_with_agent 事件循环
  │
  ├── 检测 AIMessage.tool_calls
  │     └── _safe_send("🔧 {name}", type="tool_call_start", tool_name=...)
  │           │
  │           └── send_to_agent(user_id, {text, type, tool_name, ...})
  │                 │
  │                 └── Hub → client.py
  │                       │
  │                       ├── payload.type == "tool_call_start"
  │                       │     ├── {"type": "tool_call_start", ...} → 前端 WS
  │                       │     │     └── 状态条: "正在执行: {tool_name}..."
  │                       │     └── {"type": "message", ...} → 前端 WS
  │                       │           └── 聊天记录: 可折叠条目
  │                       │
  ├── 检测 ToolMessage
  │     └── _safe_send("🛠️ {content}", type="tool_call_end", tool_name=...)
  │           │
  │           └── (同上双通道) → 状态条"完成" + 聊天记录可折叠
  │
  └── 检测 AIMessage.content（普通文本回复）
        └── _send_ai_message(msg) → _safe_send(content)
              └── 无 type 字段 → client.py else 分支 → 纯聊天消息
```

**折叠规则**：短单行（≤150 字符）直接展示；长单行截取前 120 字符 + `…` 做摘要；多行以第一行为摘要。用户点击 `<details>` 展开查看完整内容。

### 11.1 任务中断机制（Stop Task）

用户可在前端点击停止按钮中断 Agent 的长时间运行任务。设计原则：**以 `brain.process()` 的生命周期作为"任务中"的边界**。

**流程：**

```
用户点击 [⏹] → 前端 ws.send({type:"stop_task", agent_id})
  → client.py 转发到 Hub
    → Hub.route_message() 路由到目标 Agent
      → core.py → brain.stop_current_task()
        → asyncio.Task.cancel() 取消当前 process()
          → CancelledError 在 astream() 的 await 点抛出
            → process() 捕获 → 发送 "⏹ 任务已停止" + agent_status:idle
              → 前端隐藏停止按钮
```

**agent_status 协议：**

Agent 通过 `agent_status` 消息实时通知前端自身状态，前端据此显示/隐藏停止按钮：

| 字段 | 说明 |
|------|------|
| `type` | `"agent_status"` |
| `agent_id` | Agent ID |
| `status` | `"busy"`（任务执行中）或 `"idle"`（空闲） |

- `process()` 入口发送 `busy`，出口（finally）发送 `idle`
- CancelledError 也被捕获，出口同样发送 `idle`
- Hub 将 `agent_status` 消息转发到 `super_user`（前端）
- 前端维护 `agentBusyStates` 映射，按当前活跃对话控制停止按钮显隐

**关键实现细节：**

| 组件 | 文件 | 职责 |
|------|------|------|
| 任务跟踪 | `agent/brain.py` | `_current_task` 存储当前 asyncio Task；`stop_current_task()` 调用 `.cancel()` |
| 取消处理 | `agent/brain.py` | `process()` / `process_group_message()` 中捕获 `asyncio.CancelledError`，清理后发送停止消息 |
| 状态通知 | `agent/brain.py` | `_notify_status()` 通过 `comm.send()` 发送 `agent_status` |
| 消息路由 | `agent/core.py` | 接收 `stop_task` 消息，调用 `brain.stop_current_task()` |
| Hub 转发 | `hub/server.py` | 路由 `stop_task` 和 `agent_status` 消息 |
| 前端按钮 | `client.html` | `#stopBtn` 根据 `agentBusyStates[activeAgent]` 控制显隐 |

**agent_status 缓存（刷新恢复）**：`client.py` 中维护 `agent_statuses: dict[str, str]` 缓存每个 Agent 最后上报的状态。当前端新连接建立时，在发送在线 Agent 列表后补发所有已缓存的 status，使页面刷新后停止按钮立即恢复正确状态。Agent 离线时清理对应缓存。

**设计考量：**

- **为什么用 asyncio Task 取消而非标志位轮询？** 标志位需要在循环中主动检查，而 `astream()` 是阻塞调用，等待 LLM 响应期间无法检查。`Task.cancel()` 在 await 点立即抛出 `CancelledError`，响应即时。
- **为什么 busy/idle 以 `process()` 为边界？** Agent 可能在任务中发多条消息、调用多个工具，但只要 `process()` 没返回，就代表任务未完成。用 `process()` 的入口/出口统一判断，逻辑简单可靠。
- **群聊不支持停止？** 停止按钮仅在私聊（`activeAgent` 非空）时显示。群聊涉及多个 Agent，停止单个 Agent 的语义不够清晰，暂不处理。

### 11.2 新消息自动中断当前任务

用户发送新消息时，若 Agent 正在执行任务（`is_busy = True`），自动取消当前任务并以新消息开始。设计原则：**以用户意图优先——新消息代表用户的最新需求，旧任务应立即让路**。

**流程：**

```
用户在 Agent 执行任务中发送新消息
  → client.py WebSocket → Hub → Agent core.py:_handle_message
    → 检测 msg_type == "message" && sender == "super_user"
      → 检查 brain.is_busy
        ├── True  → brain.stop_current_task()
        │           → asyncio.Task.cancel() 取消当前 process()
        │             → CancelledError 在 astream() 的 await 点抛出
        │               → process() finally 块发送 agent_status:idle
        │           → await asyncio.sleep(0.15) 确保取消传播完毕
        │           → asyncio.create_task(_process_message(sender, new_msg, ...))
        │             → brain.process(thread_id_override=同一 thread_id)
        │               → 发送 agent_status:busy，开始新任务
        │
        └── False → 直接 asyncio.create_task(_process_message(...))
```

**与 agent_status 协议的协作：**

| 事件 | agent_status 消息 | 前端效果 |
|------|-------------------|---------|
| 用户发新消息，Agent 正忙 | 旧任务 cancelled → `idle`；新任务开始 → `busy` | 停止按钮短暂隐藏后重新显示 |
| 用户发新消息，Agent 空闲 | 新任务开始 → `busy` | 停止按钮显示 |
| 新任务完成 | `idle` | 停止按钮隐藏 |

**关键实现细节：**

| 组件 | 文件 | 职责 |
|------|------|------|
| 入口检查 | `agent/core.py:88-91` | 在 `_handle_message` 中检测 `is_busy`，调用 `stop_current_task()`，等待 0.15s 后创建新任务 |
| 任务取消 | `agent/brain.py` | `stop_current_task()` 调用 `_current_task.cancel()`；`process()` 的 finally 块发送 `idle` |
| 上下文连续性 | `agent/core.py:165` | 新任务使用相同的 `thread_id`，LangGraph checkpoint 保持对话历史连贯 |
| 状态通知 | `agent/brain.py` | `_notify_status()` 通过 `comm.send()` 发送 `agent_status` |

**设计考量：**

- **为什么保持同一 thread_id？** 取消旧任务后，用户希望 Agent"记住刚才在做什么"。复用 thread_id 让 LangGraph checkpoint 中的历史消息保留，Agent 在新消息的上下文中仍然可以看到之前的工具调用和结果，从而理解用户的纠正意图（如"不对，用百度搜索"）。

- **为什么只对 super_user 生效？** 其他 Agent 的消息走的是独立的缓冲+合并路径（见 3.3），不应打断当前用户任务。只有用户（super_user）的消息才具有"立即中断"的优先级。

- **为什么需要 0.15s 延迟？** `Task.cancel()` 在 await 点抛出 `CancelledError`，但取消信号的传播是异步的。短暂等待确保旧任务的 finally 块（发送 `idle` 状态、清理资源）执行完毕，避免新旧任务的状态通知交错。

- **与 Stop Task 按钮的关系？** 两者共享同一底层机制（`asyncio.Task.cancel()`），但触发方式不同：Stop Task 是用户显式点击按钮（`stop_task` 消息），消息中断是用户发送新消息隐式触发。前者适合"彻底不想继续了"，后者适合"方向错了，换个方式"。

- **消息丢弃风险？** 旧任务在 cancel 后，其未发送的回复（`comm.send_to_agent`）会被丢弃。这是预期行为——旧任务的输出对用户已无意义。新任务从头开始处理用户的最新消息。

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

#### 13.9.1 工具注册（brain.py）

```python
# brain.py line 41
from agent.computer_tools import COMPUTER_TOOLS

# brain.py line 150
tools = [..., browser] + room_tools + COMPUTER_TOOLS
# COMPUTER_TOOLS 包含 windows_automation + 18 个 computer_* 工具
```

所有电脑工具以**平铺方式**注册到 LangGraph Agent，Agent 在每次 LLM 调用时都能看到全部 19 个工具。

#### 13.9.2 提示词注入策略

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

#### 13.9.3 HITL 安全策略

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

## 十三、聊天图片系统

### 13.1 概述

AgentPro 支持聊天中展示图片，来源分两类：

| 来源 | 场景 | 存储目录 | 访问路径 |
|------|------|---------|---------|
| 用户上传 | 用户在聊天框中选择图片发送 | `chat_images/` | `/chat_images/{hash}.{ext}` |
| 智能体截图 | 浏览器/桌面自动化执行后截图 | `screenshots/` | `/screenshots/screenshot_{ts}.png` |

### 13.2 用户图片上传流程

```
用户选择图片
  → FileReader.readAsDataURL() → 内存中的 base64 Data URL
  → uploadImage() → POST /chat/upload-image  { image: "data:image/png;base64,..." }
    → 服务端解析 MIME 类型和扩展名
    → SHA256 哈希前 16 位命名（防重复）
    → 写入 chat_images/{hash}.png
    → 返回 { url: "/chat_images/{hash}.png" }
  → 路径存入 messageCache、数据库、WebSocket 消息
  → <img src="/chat_images/{hash}.png"> 在聊天气泡中展示
```

### 13.3 智能体截图展示

智能体在浏览器/桌面操作完成后，截图工具返回 markdown 格式：

```
![截图](/screenshots/screenshot_1712345678.png)
```

前端通过 `marked.parse()` 将 markdown 转为 HTML `<img>` 标签，浏览器直接从 `/screenshots/` 路径加载。

两个截图工具 (`computer_screenshot` / `browser.screenshot`) 的返回值中已包含正确的 markdown 格式，智能体直接复制到回复中即可展示。

### 13.4 存储架构

```
┌─────────────────────────────────────────┐
│  前端 (client.html)                      │
│  - currentImageBase64 (暂存，仅内存)     │
│  - imageUrl (上传后的路径，持久化)        │
│  - messageCache[].image = path          │
│  - roomMessages[].image = path          │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  client.py (FastAPI)                    │
│  - POST /chat/upload-image → 存盘返回URL │
│  - /chat_images/ → StaticFiles 挂载     │
│  - /screenshots/ → StaticFiles 挂载     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  PostgreSQL chat_messages               │
│  - image TEXT (存储路径，非 base64)      │
│  - 路径示例: /chat_images/a1b2c3d4.png  │
└─────────────────────────────────────────┘
```

### 13.5 关键设计决策

| 决策 | 理由 |
|------|------|
| 图片存磁盘、DB 只存路径 | base64 存 DB 会导致严重膨胀（每条消息数 MB），路径仅几十字节 |
| 内容哈希命名 | 相同图片不重复存储；SHA256 前 16 位碰撞概率极低 |
| 前端上传后再发送消息 | 确保消息发送时图片已持久化，消息缓存和 DB 中存的都是可靠路径 |
| `displayMessage(image)` 参数贯穿全链路 | 缓存、API、WebSocket、renderMessages 统一传递 image，避免切换工具开关后图片丢失 |
| 用户图片和截图分开目录 | 职责不同：`chat_images/` 持久化用户上传，`screenshots/` 临时截图（7 天自动清理） |

### 13.6 前端渲染

```javascript
function displayMessage(sender, text, time, image = null) {
    // image 为服务器路径如 /chat_images/xxx.png 或 /screenshots/xxx.png
    const imageHtml = image
        ? `<img src="${image}" class="chat-image" onclick="viewFullImage(this.src)" loading="lazy">`
        : '';
    // marked.parse() 同时处理 markdown 中嵌入的图片语法
    msgDiv.innerHTML = `<div class="bubble">${imageHtml}${marked.parse(text)}</div>`;
}
```

点击聊天气泡中的图片可全屏放大预览（`viewFullImage` 创建 overlay）。

---

## 十四、浏览器自动化

### 14.1 概述

AgentPro 内置基于 Playwright 的浏览器自动化系统，支持 19 种网页操作。默认使用系统安装的 Chrome 浏览器（`channel="chrome"`），cookie 和登录态通过持久化会话目录 (`browser_data/`) 自动保持。

详见 `agent/skills/browser-automation/SKILL.md`。

### 14.2 反自动化检测体系

现代网站（如 BOSS 直聘）广泛部署了自动化检测机制（navigator.webdriver 检测、WebGL 指纹、Chrome 自动化标志等）。AgentPro 采用**多层纵深防御**策略绕过检测：

#### 第一层：Chrome 启动参数伪装

```python
_STEALTH_ARGS = [
    "--disable-blink-features=AutomationControlled",   # 隐藏 navigator.webdriver
    "--disable-features=IsolateOrigins,site-per-process",
    "--no-first-run", "--no-default-browser-check",
    "--disable-infobars", "--disable-sync",
    "--disable-default-apps", "--disable-translate",
    "--disable-component-extensions-with-background-pages",
    "--disable-client-side-phishing-detection",
    "--disable-component-update",
    "--password-store=basic", "--use-mock-keychain",
]
```

#### 第二层：真实 Chrome User-Agent

```python
_STEALTH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
```

#### 第三层：playwright-stealth 注入

使用 `playwright-stealth` 库（2.x API）修改浏览器 JavaScript 环境，消除可被检测的自动化痕迹：

```python
from playwright_stealth import Stealth

_stealth = Stealth()
# 对现有页面注入
for p in self._context.pages:
    _stealth.apply_stealth_sync(p)
# 对后续新页面自动注入
self._context.on("page", lambda p: _stealth.apply_stealth_sync(p))
```

Stealth 修改的内容包括：`navigator.webdriver` → `false`、`navigator.plugins` 数组填充、`navigator.permissions.query` 行为修正、WebGL vendor/renderer 改为真实 GPU 值、`chrome.runtime` 对象注入等。

#### 第四层：系统 Chrome + CDP 连接（终极方案）

前三层防护对部分站点仍不够，因为它们检测 TLS 握手指纹（JA3/JA4）——Playwright 自带 Chromium 的 TLS 指纹与真实 Chrome 不同。

**解决方案**：使用系统安装的真实 Chrome，通过 CDP (Chrome DevTools Protocol) 连接：

```
用户手动启动 Chrome：
  chrome.exe --remote-debugging-port=9222

AgentPro 设置环境变量：
  BROWSER_CDP_PORT=9222

AgentPro 通过 connect_over_cdp() 连接，
使用用户已登录的 Chrome 会话，
TLS 指纹、cookie、登录态完全真实。
```

CDP 模式的代码路径（`browser_tools.py:164-181`）：

```python
if _BROWSER_CDP_PORT:
    cdp_url = f"http://127.0.0.1:{_BROWSER_CDP_PORT}"
    self._browser = self._playwright.chromium.connect_over_cdp(cdp_url)
    contexts = self._browser.contexts
    self._context = contexts[0] if contexts else self._browser.new_context()
    pages = self._context.pages
    self._page = pages[-1] if pages else self._context.new_page()
    # 用已有标签页，无需新建
```

**CDP 模式关键行为**：
- 浏览器由用户手动管理（标签页、登录、cookie 等）
- AgentPro 仅接管单个标签页，不影响用户其他标签
- `close()` 仅断开 CDP 连接，不关闭浏览器
- 页面新开时，`playwright-stealth` 自动注入所有标签页

### 14.3 浏览器环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BROWSER_HEADLESS` | 设为 `1` 启用无头模式（不显示窗口） | `0`（可见） |
| `BROWSER_CHANNEL` | 浏览器通道：`chrome`（系统 Chrome）或 `chromium`（Playwright 自带） | `chrome` |
| `BROWSER_CDP_PORT` | CDP 远程调试端口，设置后连接已有 Chrome 而非启动新实例 | 空（不启用） |
| `CHROME_PATH` | Chrome 可执行文件路径（`channel="chrome"` 时自动从注册表查找，也可手动指定） | 空（自动查找） |

### 14.4 任务完成后自动关闭

Agent 完成浏览器任务后必须关闭浏览器释放资源。机制：

```
BRAIN_BASE_PROMPT 明确指示：
  "任务完成后必须截图并调用 browser(action='close') 关闭浏览器释放资源"

Agent 操作流程：
  → browser.screenshot → 保存最终截图
  → 回复中包含 markdown 截图展示
  → browser.close → 关闭/断开浏览器会话
```

### 14.5 DPI 缩放问题修复

**问题**：Windows 上拖拽 Playwright 浏览器窗口导致前端 UI 整体放大。

**根因**：`--disable-gpu` 参数在非无头模式下触发了 Windows DPI 虚拟化，拖拽窗口在不同 DPI 显示器间移动时，系统错误地缩放了整个应用窗口。

**修复**：
- 非无头模式下移除 `--disable-gpu` 参数
- 添加 `--force-device-scale-factor=1` 锁定 DPI 缩放比例
- 无头模式仍保留 `--disable-gpu`（无头模式不涉及窗口渲染）

```python
args=[
    "--no-sandbox", "--disable-setuid-sandbox",
    "--disable-dev-shm-usage", "--force-device-scale-factor=1",
] + (["--disable-gpu"] if self.headless else []),
```

### 14.6 架构要点

| 要点 | 说明 |
|------|------|
| 单线程执行器 | Playwright 同步 API 要求所有 greenlet 操作在同一线程；使用 `ThreadPoolExecutor(max_workers=1)` 确保线程亲和性 |
| 全局单例 | `_browser_session` 全局唯一切片，多次工具调用共享同一浏览器会话 |
| 懒初始化 | 首次调用 `browser` 工具时才启动浏览器，避免无浏览器任务时资源浪费 |
| 优雅降级 | `launch_persistent_context` 失败 → 自动降级为普通 `launch` + `new_context` |
| 线程安全 | `_lock` 双重检查锁保证延迟初始化线程安全 |
| Windows 兼容 | `sync_playwright()` 子进程需要 `ProactorEventLoop`，初始化后恢复原策略 |

---

## 十五、文件清单

| 文件 | 职责 |
|------|------|
| `main.py` | 入口，初始化数据库/调度器/智能体/后台 worker |
| `config.py` | 全局配置 |
| `agent/core.py` | Agent 主控，消息路由 |
| `agent/brain.py` | LLM 决策，工具注册，对话管理，上下文编排 |
| `agent/context_manager.py` | Token 预算跟踪 + 工具输出压缩 + 上下文中间件 |
| `agent/communication.py` | WebSocket 客户端 |
| `agent/prompts.py` | 所有 LLM 提示词 |
| `agent/agent_context/*.md` | Agent 专属上下文文件（类似 Claude Code 的 CLAUDE.md） |
| `agent/conversation_tracker.py` | 智能体对话轮次控制（TDP 兜底） |
| `agent/delegation.py` | TDP 任务委托协议核心（TaskTicket + DelegationManager） |
| `agent/orchestration.py` | 多智能体任务编排（OrchestrationPlan + DAG 并行派发） |
| `agent/task_buffer.py` | 任务步骤缓冲 |
| `agent/memory.py` | ChromaDB 记忆存储 + 用户画像按需读取 |
| `agent/conversation_memory_extractor.py` | 对话记忆后台提取 |
| `agent/memory_consolidation.py` | 每日记忆去重整合 |
| `agent/reflection.py` | 任务反思 + 技能生成 |
| `agent/skill_tools.py` | 技能管理工具 |
| `agent/skill_version_manager.py` | 技能版本管理 |
| `agent/sandboxed_backend.py` | Docker 沙箱 |
| `agent/browser_tools.py` | Playwright 浏览器工具 |
| `agent/message_buffer.py` | Agent 间消息缓冲队列 |
| `agent/tools_factory.py` | 工具工厂函数（send_to_agent、room_tools 等） |
| `agent/computer_tools.py` | 19 个桌面自动化工具（Computer Use） |
| `agent/tools.py` | 自定义 LangChain 工具（含 windows_automation） |
| `agent/scheduler.py` | APScheduler 调度 |
| `agent/tasks.py` | 提醒 + 整合任务 |
| `agent/model_config.py` | 多模型配置 |
| `agent/intent.py` | 意图类型枚举 |
| `agent/db.py` | PostgreSQL 连接池 |
| `clean_db.py` | 数据库与临时文件清理脚本 |
| `hub/server.py` | WebSocket Hub |
| `client.py` | FastAPI + super_user 客户端 |
| `client.html` | Web 前端 |
| `agent/skills/user-profile/SKILL.md` | 用户画像按需加载技能 |
| `agent/skills/computer-automation/SKILL.md` | 桌面自动化技能文档 |
| `agent/skills/browser-automation/SKILL.md` | 浏览器自动化技能文档 |

---

## 十六、数据库清理与运维

### 16.1 概述

`clean_db.py` 是数据库与临时文件的全能清理工具，统一替代旧有的 `clean_checkpoints.py`。支持按类别精确删除或一键全清，涵盖工单、编排计划、聊天记录、检查点、提醒、房间、临时文件等所有可清理数据。

### 16.2 支持的数据表

| 表 | 清理选项 | 说明 |
|---|---------|------|
| `delegation_tickets` | `--tickets [--active-only]` | 委托工单，支持仅清理活跃（非终态）工单 |
| `orchestration_plans` / `orchestration_subtasks` | `--orchestration [--active-only]` | 编排计划及子任务（级联删除） |
| `chat_messages` | `--chat` | 聊天消息内容 |
| `conversation_threads` | `--conversations` | 会话线程元数据（级联删除聊天消息） |
| `checkpoints` / `checkpoint_blobs` / `checkpoint_writes` | `--checkpoints [--thread ID]` | LangGraph 短期记忆检查点 |
| `reminders` | `--reminders` | 定时提醒 |
| `rooms` / `room_members` | `--rooms [--room ID]` | 群聊房间及成员 |

### 16.3 按智能体清理

`--agent AGENT_ID` 会清理与该智能体关联的**全部数据**：

1. 聊天消息（`chat_messages WHERE thread_id LIKE 'private_{agent_id}_%'`）
2. 会话线程（`conversation_threads WHERE agent_id = ...`）
3. 检查点（所有匹配的 thread_id）
4. 关联工单（issuer 或 assignee 是该智能体）
5. 关联子任务（`orchestration_subtasks WHERE assigned_to = ...`）
6. 提醒（`reminders WHERE user_id = ...`）
7. 空编排计划清理（子任务全部删除后级联清理父计划）

### 16.4 临时文件清理

| 选项 | 目录 | 内容 |
|------|------|------|
| `--screenshots` | `screenshots/` | 浏览器/桌面截图 |
| `--tool-outputs` | `agent/agent_temp/tool_outputs/` | 压缩工具输出侧文件 |
| `--temp` | 全部以上 + `__pycache__` | 所有临时文件 |

### 16.5 设计决策

| 决策 | 理由 |
|------|------|
| 惰性导入数据库模块 | `--help` 不需要安装 psycopg，通过 `_get_db()` 仅在执行实际数据库操作时才导入 |
| `--active-only` 过滤终态数据 | 终态工单/计划可作为审计留痕，仅清理活跃状态可避免影响正在运行的任务 |
| 级联删除而非依赖 ON DELETE CASCADE | 子任务/消息显式先删除，避免依赖数据库外键约束的隐性行为 |
| `--agent` 全链路清理 | 智能体可能在多个表中留下数据（聊天、工单、子任务、提醒），逐表清理容易遗漏 |
| `--all --force` 双层防护 | `--all` 需确认输入 "yes"，`--force` 跳过确认，防止误操作 |
