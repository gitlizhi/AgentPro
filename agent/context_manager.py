"""
    上下文管理器——令牌预算跟踪、工具输出压缩和系统/用户消息角色分离实用程序。
    受Claude Code分层上下文模型的启发：
        -系统提示（静态指令）
        -CLAUDE.md文件（特定于项目/代理的上下文）
        -内存文件（持久，根据需要加载）
        -当前对话（通过压缩管理）
    本模块提供了保持角色界限清晰的机制
    并主动将上下文窗口控制在预算范围内。
    Context Manager — token budget tracking, tool output compaction, and
    system/user message role separation utilities.
    
    Inspired by Claude Code's layered context model:
      - System prompt (static instructions)
      - CLAUDE.md files (project/agent-specific context)
      - Memory files (persistent, loaded as needed)
      - Current conversation (managed by compaction)
    
    This module provides the machinery to maintain clean role boundaries
    and keep the context window within budget proactively.
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.agents.middleware.types import AgentMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Character-to-token ratio for token estimation.
# DeepSeek / OpenAI models: ~3.3 chars per token for mixed CN/EN text.
# Conservative estimate of 2.8 helps us under-count rather than over-count,
# i.e. we'll compact sooner, which is safer.
CHARS_PER_TOKEN = 2.8

# Default token budget (total for the entire message history).
# DeepSeek V4 context window is 128K, but we budget conservatively to
# leave room for the model response and the system prompt.
DEFAULT_TOKEN_BUDGET = 625000

# SummarizationMiddleware is configured to trigger at 20K tokens.
# We add a proactive warn/compact threshold as percentages of budget.
WARN_THRESHOLD = 0.80   # 80% — notify user via _safe_send
COMPACT_THRESHOLD = 0.85  # 85% — compact aggressively

# Tool output compaction: if a ToolMessage content exceeds this many
# characters, truncate and store the full output in a side file.
MAX_TOOL_OUTPUT_CHARS = 2000

# Directory for storing full tool outputs (created lazily).
TOOL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "agent_temp", "tool_outputs")

_cleaned_up = False


def cleanup_old_tool_outputs(days: int = 7):
    """删除超过指定天数的工具输出文件（会话内仅执行一次）。"""
    global _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    dir_path = Path(TOOL_OUTPUT_DIR)
    if not dir_path.exists():
        return
    cutoff = time.time() - days * 86400
    deleted = 0
    for f in dir_path.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass
    if deleted > 0:
        logger.info(f"已清理 {deleted} 个过期工具输出文件（>{days}天）")


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------

class TokenBudget:
    """Track estimated token usage of a message list.

    Uses a simple character-count heuristic.  Does NOT call any tokenizer;
    all estimates are approximate and conservative (under-count slightly
    so we compact before hitting real limits).
    跟踪消息列表的估计令牌使用情况。
    使用简单的字符计数启发式方法。不调用任何标记器；
    所有的估计都是近似的和保守的（略微低估了
    因此，我们在达到实际极限之前进行压缩）。
    """

    def __init__(self, max_tokens: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.max_tokens = max_tokens
        self._current_estimate: int = 0

    # -- public API --------------------------------------------------------

    def estimate_tokens(self, messages: list) -> int:
        """Return an estimated token count for *messages*.

        Walks each message, summing ``estimate_message(msg)``.
        """
        total = 0
        for msg in messages:
            total += self._estimate_one(msg)
        return total

    def update(self, messages: list) -> int:
        """Recompute and store the current estimate.  Returns the count."""
        self._current_estimate = self.estimate_tokens(messages)
        return self._current_estimate

    @property
    def current(self) -> int:
        return self._current_estimate

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self._current_estimate)

    @property
    def usage_ratio(self) -> float:
        if self.max_tokens == 0:
            return 1.0
        return self._current_estimate / self.max_tokens

    def should_warn(self) -> bool:
        """Return True if the token usage has crossed the warning threshold."""
        return self.usage_ratio >= WARN_THRESHOLD

    def should_compact(self) -> bool:
        """Return True if the token usage has crossed the proactive-compact threshold."""
        return self.usage_ratio >= COMPACT_THRESHOLD

    def reset(self) -> None:
        self._current_estimate = 0

    # -- internal ---------------------------------------------------------

    @staticmethod
    def _estimate_one(msg) -> int:
        """Estimate tokens for a single message object.

        Handles LangChain message types (HumanMessage, AIMessage, SystemMessage,
        ToolMessage) and plain dicts ({"role": ..., "content": ...}).
        """
        content = ""
        if hasattr(msg, "content"):
            content = msg.content or ""
        elif isinstance(msg, dict):
            content = msg.get("content", "") or ""

        # content may be a string or a list-of-dicts (e.g. vision content blocks)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = " ".join(text_parts)

        if not content:
            # Even empty messages cost a few tokens for role markers
            return 4

        # Tool messages cost extra role-tag overhead
        overhead = 8
        return overhead + int(len(content) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Tool output compaction
# ---------------------------------------------------------------------------

def compact_tool_output(content: str, tool_name: str = "",
                        max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate a tool result that is too long so it doesn't bloat the context.

    If *content* is under *max_chars* it is returned unchanged.
    
    Otherwise the first *max_chars* characters are kept, the remainder is
    written to a side file under ``agent/agent_temp/tool_outputs/``, and a
    reference + short tail-snippet is appended so the LLM can still see the
    very end of the output (which often contains status / error messages).

    Returns the compacted string.
    
    截断过长的工具结果，这样就不会使上下文膨胀。
    如果*content*低于*max_chars*，则返回不变。
    否则，保留前*max_chars*个字符，其余字符为
    写入“agent/agent_temp/tool_outputs/”下的侧文件，以及
    附加了引用+短尾代码段，这样LLM仍然可以看到
    输出的末尾（通常包含状态/错误消息）。
    
    返回压缩后的字符串。
    """
    if len(content) <= max_chars:
        return content

    cleanup_old_tool_outputs()
    os.makedirs(TOOL_OUTPUT_DIR, exist_ok=True)

    # Unique filename based on hash of content
    content_hash = str(hash(content))
    fname = f"{tool_name or 'tool'}_{content_hash[-12:]}.txt"
    fpath = os.path.join(TOOL_OUTPUT_DIR, fname)

    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        logger.warning(f"Failed to write tool output to {fpath}: {e}")
        # Fallback: just truncate brutally
        return content[:max_chars] + f"\n\n[... 输出过长，共 {len(content)} 字符，已截断]"

    # Keep first max_chars, plus the last 300 chars (often status/errors at tail)
    head = content[:max_chars]
    tail = content[-300:] if len(content) > max_chars + 300 else ""

    compacted = (
        f"{head}\n\n"
        f"[... 中间省略 {len(content) - max_chars - (300 if tail else 0)} 字符 ...]"
    )
    if tail:
        compacted += f"\n\n{tail}"
    compacted += (
        f"\n\n[完整输出已保存至 {fpath}，共 {len(content)} 字符]"
    )
    logger.debug(f"Compacted tool output: {len(content)} → {len(compacted)} chars ({tool_name})")
    return compacted


def should_compact(content: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> bool:
    """Convenience: return True if *content* would be compacted."""
    return len(content) > max_chars


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """Orchestrates context-window hygiene for a Brain instance.

    Responsibilities:
      - Estimate and track token usage across the message history
      - Compact tool outputs before they enter the message list
      - Provide proactive warnings when approaching the token budget
      
    为Brain实例编排上下文窗口
    
    责任：
        -估计并跟踪整个消息历史中的令牌使用情况
        -在进入消息列表之前压缩工具输出
        -在接近代币预算时提供主动警告
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_TOKEN_BUDGET,
        max_tool_output_chars: int = MAX_TOOL_OUTPUT_CHARS,
    ) -> None:
        self.budget = TokenBudget(max_tokens=max_tokens)
        self.max_tool_output_chars = max_tool_output_chars

    # -- tool output ------------------------------------------------------

    def compact_tool_result(self, content: str, tool_name: str = "") -> str:
        """Compact a single tool result, respecting the configured char limit."""
        return compact_tool_output(content, tool_name=tool_name,
                                   max_chars=self.max_tool_output_chars)

    # -- token budget -----------------------------------------------------

    def check_budget(self, messages: list) -> Optional[str]:
        """Check token budget against *messages*.

        Returns a warning string if the budget is running low, or None if
        everything is fine.  The caller (Brain) should relay the warning to
        the user via ``_safe_send``.
        
        根据*条消息*检查令牌预算。
        
        如果预算不足，则返回警告字符串，如果预算不足则返回None
        一切都很好。调用者（Brain）应将警告转达给
        用户通过“safe_send”发送。
        
        """
        self.budget.update(messages)
        if self.budget.should_warn():
            ratio = self.budget.usage_ratio
            return (
                f"上下文窗口已使用 {ratio:.0%}（约 {self.budget.current:,} / "
                f"{self.budget.max_tokens:,} tokens），接近上限时将自动压缩历史消息。"
            )
        return None

    def is_near_limit(self) -> bool:
        return self.budget.should_compact()

    # -- message helpers --------------------------------------------------

    @staticmethod
    def estimate_message(msg) -> int:
        """Estimate tokens for a single message (delegates to TokenBudget)."""
        return TokenBudget._estimate_one(msg)


# ---------------------------------------------------------------------------
# ToolOutputCompactionMiddleware
# ---------------------------------------------------------------------------

class ToolOutputCompactionMiddleware(AgentMiddleware):
    """LangChain AgentMiddleware that compacts long tool outputs before the
    model sees them.

    Wraps ``awrap_tool_call``: after every tool execution, if the result
    content exceeds the configured character limit it is compacted via
    :func:`compact_tool_output`.

    Usage inside Brain.__init__::

        ctx = ContextManager(max_tool_output_chars=2000)
        middleware_list.append(ToolOutputCompactionMiddleware(ctx))
    """

    tools = []            # no extra tools registered

    def __init__(self, context_manager: ContextManager) -> None:
        super().__init__()
        self._ctx = context_manager

    async def awrap_tool_call(self, request, handler):
        """Execute the tool, then compact the result if it is too long."""
        from langchain_core.messages import ToolMessage

        result = await handler(request)
        if isinstance(result, ToolMessage) and result.content:
            tool_name = ""
            if hasattr(request, 'tool_call') and isinstance(request.tool_call, dict):
                tool_name = request.tool_call.get('name', '')
            compacted = self._ctx.compact_tool_result(
                str(result.content), tool_name=tool_name
            )
            if compacted != result.content:
                result = ToolMessage(
                    content=compacted,
                    tool_call_id=result.tool_call_id,
                    name=getattr(result, 'name', None),
                )
        return result

    # Sync fallback for any sync-context usage
    def wrap_tool_call(self, request, handler):
        from langchain_core.messages import ToolMessage

        result = handler(request)
        if isinstance(result, ToolMessage) and result.content:
            tool_name = ""
            if hasattr(request, 'tool_call') and isinstance(request.tool_call, dict):
                tool_name = request.tool_call.get('name', '')
            compacted = self._ctx.compact_tool_result(
                str(result.content), tool_name=tool_name
            )
            if compacted != result.content:
                result = ToolMessage(
                    content=compacted,
                    tool_call_id=result.tool_call_id,
                    name=getattr(result, 'name', None),
                )
        return result

    @property
    def name(self) -> str:
        return self.__class__.__name__
