# AgentPro

<div align="center">

[![中文 README](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-red.svg)](README.md)

**Multi-Agent Collaboration Platform Powered by LangGraph + LangChain**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-orange.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-purple.svg)](https://langchain-ai.github.io/langgraph/)

</div>

---

## Table of Contents

- [Introduction](#introduction)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Memory System](#memory-system)
- [Skill System](#skill-system)
- [Computer Use](#computer-use)
- [Cleanup Scripts](#cleanup-scripts)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

AgentPro is a **production-grade multi-agent collaboration platform**. Multiple AI Agents interconnect through a WebSocket Hub, enabling private chat, group chat, task delegation, skill usage, desktop automation, and code execution. Agents possess long-term memory and self-reflection capabilities, automatically extracting experiences from past tasks to generate reusable skills.

### Capabilities

```
User → Agent
        │
        ├── 💬  Multi-agent private / group chat collaboration
        ├── 🧠  Long-term memory (ChromaDB) + Short-term memory (PostgreSQL)
        ├── 🔄  Memory → Experience → Auto-generate reusable skills
        ├── 📐  Context management (SystemMessage layering + Token budget + Tool output compaction)
        ├── 🖥️  Computer Use (desktop automation: mouse/keyboard/OCR/visual positioning)
        ├── 🌐  Browser automation (Playwright)
        ├── 📦  Docker sandbox code execution
        ├── ⏰  Scheduled reminders (APScheduler)
        ├── 👤  Human-in-the-loop (HITL approval)
        └── 💡  Proactive thinking & interaction
```

---

## Core Features

### Agent Capabilities

| Feature | Description |
|------|------|
| **Multi-Agent Architecture** | Multiple independent Agents run in parallel, routed through Hub, no interference |
| **Multi-Agent Collaboration** | Agents can private chat, group discuss, delegate tasks, collaborate on complex goals |
| **Intent Recognition** | LLM classifies user input (chat/complex task/reminder/query) for fast routing |
| **Proactive Thinking** | Background periodic inner thoughts, combines memory and conversation for proactive interaction |
| **Human-in-the-Loop (HITL)** | Critical operations trigger manual approval before execution |

### Memory & Learning

| Feature | Description |
|------|------|
| **Short-Term Memory** | PostgreSQL checkpoint persists conversation history, recovers context on restart |
| **Long-Term Memory** | ChromaDB vector storage for user profiles and events, auto dedup, synced to Markdown |
| **Memory → Experience** | Auto-reflect after complex tasks; successes generate reusable skills, failures capture lessons |
| **Daily Consolidation** | Auto-organize memories at 3:00 AM, LLM dedup and merge similar facts |

### Automation

| Feature | Description |
|------|------|
| **Computer Use** | 19 desktop operation tools: screenshot/OCR/visual positioning/mouse & keyboard/UIAutomation/command execution |
| **Browser Automation** | Playwright-driven Chromium, manipulate web pages |
| **Docker Sandbox** | Isolated command execution environment, security hardened, ephemeral |
| **Scheduled Reminders** | APScheduler + PostgreSQL, supports natural language reminder creation |

### Extensibility

| Feature | Description |
|------|------|
| **Skill System** | `SKILL.md` + scripts define extensible skills, progressive disclosure, on-demand loading |
| **Multi-Model** | DeepSeek / Zhipu GLM / OpenAI / Anthropic / Ollama, one-click switching |
| **Multi-Modal** | Supports vision models (GLM-4.6V / GLM-4.1V-Thinking-Flash) for image processing |
| **Context Management** | Inspired by Claude Code layered model: SystemMessage/HumanMessage role separation, proactive token budget tracking, auto tool output compaction, agent-specific context files |

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│  Browser (client.html)                                │
│  - Agent list, chat panel, group rooms, launch form   │
└──────────────┬───────────────────────────────────────┘
               │ WebSocket
┌──────────────▼───────────────────────────────────────┐
│  client.py (FastAPI + super_user WebSocket client)    │
│  - Frontend hosting / REST API                        │
└──────────────┬───────────────────────────────────────┘
               │ WebSocket
┌──────────────▼───────────────────────────────────────┐
│  Hub Server (hub/server.py)                          │
│  - Central message broker / room management / online broadcast │
└──┬──────────┬──────────────┬─────────────────────────┘
   │          │              │
┌──▼────┐ ┌───▼────┐ ┌──────▼──────┐
│Agent A│ │Agent B │ │reminder_bot │
│core.py│ │core.py │ │(scheduler)  │
│brain  │ │brain   │ └─────────────┘
└──┬────┘ └───┬────┘
   │          │
   │  LangGraph Agent (deepagents)
   │     ├── Computer Tools (19 desktop automation tools)
   │     ├── Browser Tools (Playwright)
   │     ├── DockerSandboxBackend
   │     ├── ContextManager (token budget + tool compaction)
   │     ├── ConversationTracker (round control)
   │     ├── TaskBuffer (task buffering)
   │     └── ChromaDB Memory
```

> See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed architecture.

---

## Quick Start

### Requirements

- **Python** 3.12+
- **PostgreSQL** database
- **Docker Desktop** (required for sandbox execution and browser automation)

### 1. Clone the Repository

```bash
git clone https://github.com/gitlizhi/AgentPro.git
cd AgentPro
```

### 2. Install Dependencies

```bash
pip install uv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### 3. Configure Environment Variables

Create `.env` file:

```bash
# PostgreSQL connection (required)
POSTGRES_URI=postgresql://user:password@localhost:5432/agentpro

# API keys (at least one required)
ZHIPU_API_KEY=your_zhipu_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Hub configuration (optional, defaults work)
HUB_HOST=localhost
HUB_PORT=8765
```

### 4. Start Docker Desktop

Download and launch [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### 5. Launch

```bash
# Windows
Double-click start_project.bat

# Or command line
python main.py
```

Visit `http://127.0.0.1:8000` in your browser after startup.

### Stop

```bash
Double-click stop_project.bat
```

---

## Configuration

### Key Environment Variables

| Variable | Description | Default |
|--------|------|--------|
| `POSTGRES_URI` | PostgreSQL connection string | Required |
| `ZHIPU_API_KEY` | Zhipu AI API key | - |
| `DEEPSEEK_API_KEY` | DeepSeek API key | - |
| `HUB_HOST` | Hub server host | `localhost` |
| `HUB_PORT` | Hub server port | `8765` |
| `MEMORY_MARKDOWN_DIR` | Long-term memory Markdown directory | `./agent_memory` |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence directory | `./chroma_db` |
| `BROWSER_HEADLESS` | Set to `1` for headless mode (hide browser window) | `0` |
| `BROWSER_CHANNEL` | Browser channel: `chrome` (system Chrome) or `chromium` (Playwright bundled) | `chrome` |
| `BROWSER_CDP_PORT` | CDP remote debugging port; when set, connects to existing Chrome (e.g. `9222`) instead of launching new | Empty (disabled) |
| `CHROME_PATH` | Chrome executable path; auto-detected from registry if not set | Empty (auto) |
| `AGENT_SHOW_CONSOLE` | Set to `1` to show sub-agent console windows (for debugging; hidden by default) | Empty (hidden) |

### Model Configuration

Pre-defined models in `agent/model_config.py`, extensible as needed:

| Config | Model | Use |
|--------|------|------|
| `default` | GLM-4.7 | Default chat model |
| `vision` | GLM-4.6V | General image understanding |
| `computer_vision` | GLM-4.1V-Thinking-Flash | Desktop operation visual recognition |
| `deepseek` | DeepSeek-V4 | Alternative primary model |
| `ollama` | llama3.1 | Local model |

Supports OpenAI, Anthropic Claude, Gemini, and other OpenAI-compatible format models.

---

## Project Structure

```
AgentPro/
├── agent/                          # Core agent module
│   ├── brain.py                    # Brain decision layer (LLM calls, intent recognition, tool registration)
│   ├── context_manager.py          # Context management (token budget + tool output compaction)
│   ├── core.py                     # Agent main class (WebSocket management, message routing)
│   ├── communication.py            # WebSocket client
│   ├── computer_tools.py           # Computer Use: 19 desktop automation tools
│   ├── tools.py                    # LangChain tools (including UIAutomation wrapper)
│   ├── browser_tools.py            # Playwright browser automation
│   ├── message_buffer.py           # Inter-agent message buffer queue
│   ├── tools_factory.py            # Tool factory functions
│   ├── sandboxed_backend.py        # Docker sandbox execution environment
│   ├── model_config.py             # Multi-model configuration management
│   ├── prompts.py                  # Centralized prompt management
│   ├── memory.py                   # ChromaDB long-term memory
│   ├── memory_consolidation.py     # Daily memory dedup & consolidation
│   ├── conversation_memory_extractor.py  # Background conversation memory extraction
│   ├── reflection.py               # Task reflection + skill generation
│   ├── skill_tools.py              # Skill retrieval tools
│   ├── skill_version_manager.py    # Skill version management
│   ├── conversation_tracker.py     # Agent conversation round control
│   ├── task_buffer.py              # Task step buffer
│   ├── scheduler.py                # APScheduler scheduler
│   ├── tasks.py                    # Background tasks (reminders, memory consolidation)
│   ├── intent.py                   # Intent enum & descriptions
│   ├── db.py                       # PostgreSQL connection pool
│   ├── utils.py                    # Utility functions
│   └── skills/                     # Built-in skills (SKILL.md)
│       ├── user-profile/           # On-demand user profile loading skill
│       ├── computer-automation/    # Desktop automation skill
│       └── browser-automation/     # Browser automation skill
│
├── hub/                            # WebSocket Hub
│   └── server.py                   # Central message routing + room management
│
├── agent_memory/                   # Long-term memory Markdown files
├── agent/agent_context/            # Agent-specific context files (CLAUDE.md model)
├── chroma_db/                      # ChromaDB persistence directory
├── screenshots/                    # Desktop screenshot storage
│
├── client.py                       # FastAPI server + super_user client
├── client.html                     # Web frontend interface
├── main.py                         # Application entry point
│
├── start_project.bat               # One-click start (Windows)
├── stop_project.bat                # One-click stop (Windows)
├── clean_db.py                     # Database & temp file cleanup script
├── pyproject.toml                  # Project config & dependencies
├── ARCHITECTURE.md                 # Detailed technical architecture docs
└── README.md
```

---

## Memory System

### Three-Layer Memory Architecture

```
User conversation / Agent interaction
       │
  ┌────▼─────────────┐
  │ Short-Term Memory  │  PostgreSQL checkpoint
  │ (conversation       │  context recovered on restart
  │  history persistence)│
  └────┬─────────────┘
       │
  ┌────▼─────────────┐
  │ Long-Term Memory   │  ChromaDB vector storage (facts + events)
  │ (background auto    │  + agent_memory/*.md (user profiles only)
  │  extraction every   │  + semantic dedup
  │  5 minutes)         │
  └────┬─────────────┘
       │
  ┌────▼─────────────┐
  │ Experience Memory  │  Task reflection → Skill generation
  │ (reusable skills)  │  Daily consolidation at 3:00 AM
  └──────────────────┘

┌─────────────────────────────────────────┐
│ Retrieval: On-demand (not injected)      │
│                                          │
│ When Agent needs user background:        │
│   → Call load_user_profile tool          │
│   → Read agent_memory/super_user.md      │
│   → Get deduped user profile (facts only)│
│                                          │
│ Compared to old approach: no longer      │
│ injects memory into every prompt, saving │
│ tokens. Agent loads on demand.           │
└─────────────────────────────────────────┘
```

### Key Design Decisions

- **On-demand loading**: Memory is no longer auto-injected into system prompts. Agent uses `load_user_profile` tool to actively fetch user profiles, saving tokens and making the Agent explicitly aware of what information it has
- **facts vs events**: facts are user profiles ("User is a Python developer"), events are operation records ("searched for news"). Only facts are written to Markdown files; `load_user_profile` only returns facts
- **Clean MD files**: No metadata like source/type/thread_id is written; each line only has timestamp and fact content
- **Incremental extraction**: Tracks processing progress per thread, only extracts new messages
- **Semantic dedup**: cosine similarity < 0.15 treated as duplicate, automatically filtered
- **Daily consolidation**: Full LLM dedup and merge at 3:00 AM to keep the memory store clean

---

## Skill System

Agents obtain operating instructions via **progressive disclosure** — the system prompt contains only one line: "Use `load_skill` to load skills". Full `SKILL.md` is loaded only when needed. This saves tokens while allowing new capabilities to be added at any time.

### Built-in Skills

| Skill | Trigger Keywords | Description |
|------|--------|------|
| `user-profile` | user info, personalized recommendations, learn about user... | On-demand user profile loading (with `load_user_profile` tool) |
| `computer-automation` | open app, operate WeChat, computer operations... | Full Windows desktop automation workflow |
| `browser-automation` | browse web, browser operations... | Playwright browser control |

### Skill Lifecycle

```
Execute task → Complete (task_complete=True)
  → Reflection worker analyzes success/failure
  → On success & reusable → Generate SKILL.md
  → Vectorize & store in ChromaDB
  → Retrievable via search_skills next time
  → Low-value skills auto-archived
```

---

## Computer Use

AgentPro has full **Windows desktop automation** capability, enabling AI Agents to control computers like humans.

### Capability Matrix

| Category | Tools | Capability |
|------|------|------|
| **Screen Perception** | `computer_screenshot`, `computer_see_and_describe` | Screenshot + vision model screen understanding |
| **Precise Positioning** | `computer_ocr_find`, `computer_locate` | EasyOCR text positioning / 20×14 grid visual positioning |
| **UIA Control** | `windows_automation` (22 operations) | pywinauto accessibility tree, zero coordinate error |
| **Window Management** | `computer_find_window`, `computer_find_app` | Find/activate windows, search and launch apps |
| **Mouse Ops** | `computer_move`/`click`/`double_click`/`right_click`/`scroll`/`drag` | Full mouse control |
| **Keyboard Ops** | `computer_type`/`key_press`/`paste` | English input / hotkeys / Chinese clipboard paste |
| **Command Execution** | `computer_execute` | Windows cmd, 30s timeout |

### Four-Tier Positioning Strategy

```
Tier 0: UIAutomation (most precise) ─── Direct accessibility tree control, pixel-level precision
  ↓ Non-native UI
Tier 1: OCR text positioning ─── EasyOCR screen text recognition, returns precise coordinates
  ↓ Text not visible
Tier 2: Window lookup ─── pygetwindow title matching, returns window rectangle
  ↓ No window title
Tier 3: Visual grid positioning ─── 20×14 grid + GLM-4.1V vision model
```

> Detailed docs: [ARCHITECTURE.md Section 12](./ARCHITECTURE.md#十二computer-use--windows-桌面自动化)

---

## Cleanup Scripts

`clean_db.py` is a unified database and temp file cleanup tool supporting per-category precise removal or one-click full reset.

```bash
# View stats for all tables (no deletion)
python clean_db.py --stats

# Clean tickets
python clean_db.py --tickets              # All tickets (including terminal states)
python clean_db.py --tickets --active-only # Only active tickets

# Clean orchestration plans
python clean_db.py --orchestration
python clean_db.py --orchestration --active-only

# Clean chat & conversations
python clean_db.py --chat                 # All chat messages
python clean_db.py --conversations        # All conversation threads (cascade delete messages)

# Clean short-term memory (LangGraph checkpoints)
python clean_db.py --checkpoints
python clean_db.py --checkpoints --thread "thread_id"  # Specific thread

# Clean by agent (chat + checkpoints + conversations + linked tickets + subtasks + reminders)
python clean_db.py --agent agent_main

# Clean reminders
python clean_db.py --reminders

# Clean group chat rooms
python clean_db.py --rooms               # All rooms
python clean_db.py --room room_xxx       # Specific room

# Clean temp files
python clean_db.py --screenshots          # Browser screenshots
python clean_db.py --tool-outputs         # Tool output logs
python clean_db.py --temp                 # All temp files

# Full reset (dangerous, confirmation required)
python clean_db.py --all
python clean_db.py --all --force          # Skip confirmation
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Follow PEP 8 coding standards.

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Acknowledgements

- [LangChain](https://www.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/) teams
- [deepagents](https://github.com/hwchase17/deepagents) skill system
- [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) Computer Use paradigm
- [Zhipu AI](https://open.bigmodel.cn/) and [DeepSeek](https://www.deepseek.com/) model APIs
