"""
智能体对话追踪器：按轮次计数，逐级降级警告，硬上限 + 用户通知。
管理每个对话会话的 thread_id，确保每次新会话有独立的 LangGraph checkpoint。

核心思路：不靠 LLM 事后判断，而用结构化规则在通信层面控制对话范围。
"""

import time
import uuid
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_MAX_ROUNDS = 8          # 硬上限轮次
DEFAULT_COOLDOWN_MINUTES = 30   # 达到上限后的冷却时间
WARN_THRESHOLD_1 = 4            # 第一级警告
WARN_THRESHOLD_2 = 6            # 第二级警告（最后通牒）


class ConversationState:
    """单对智能体之间的一个对话会话状态"""

    __slots__ = (
        "round_count", "max_rounds", "is_capped", "capped_at",
        "cooldown_until", "last_message_time", "task_description",
        "agent_a", "agent_b", "thread_id",
    )

    def __init__(self, agent_a: str, agent_b: str, max_rounds: int = DEFAULT_MAX_ROUNDS):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.round_count = 0
        self.max_rounds = max_rounds
        self.is_capped = False
        self.capped_at: float = 0.0
        self.cooldown_until: float = 0.0
        self.last_message_time: float = 0.0
        self.task_description: str = ""
        # 本次会话专用的 thread_id，隔离不同会话的 checkpoint
        self.thread_id: str = ""

    def increment(self) -> str:
        """
        递增轮次计数。
        返回: 'ok' | 'warning' | 'final_warning' | 'capped'
        """
        self.round_count += 1
        self.last_message_time = time.time()

        if self.round_count >= self.max_rounds:
            self.is_capped = True
            self.capped_at = time.time()
            return "capped"
        if self.round_count >= WARN_THRESHOLD_2:
            return "final_warning"
        if self.round_count >= WARN_THRESHOLD_1:
            return "warning"
        return "ok"

    @property
    def remaining(self) -> int:
        return max(0, self.max_rounds - self.round_count)

    def get_warning_hint(self) -> Optional[str]:
        """获取当前轮次应注入的警告提示，None 表示无需警告。"""
        if self.is_capped:
            return None  # 硬截断不走提示注入
        r = self.remaining
        if r == 1:
            return (
                f"⚠️ 这是你与 {self.agent_b} 对话的**最后一轮**。"
                f"请在本轮内完成所有必要的任务交接并输出最终结论。"
                f"不要发起新的子任务或展开新话题。"
            )
        if r == 2:
            return (
                f"⚠️ 你与 {self.agent_b} 的对话还剩 **2 轮**。"
                f"请开始收尾，汇总已完成的工作，准备结束对话。"
            )
        if self.round_count >= WARN_THRESHOLD_1:
            return (
                f"💡 你与 {self.agent_b} 的对话已进行 {self.round_count} 轮。"
                f"请在 {r} 轮内完成交流。避免展开新话题。"
            )
        return None


class ConversationTracker:
    """管理所有智能体对之间的对话状态和会话 thread_id"""

    def __init__(
        self,
        default_max_rounds: int = DEFAULT_MAX_ROUNDS,
        cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES,
    ):
        self._pairs: Dict[str, ConversationState] = {}
        self.default_max_rounds = default_max_rounds
        self.cooldown_minutes = cooldown_minutes

    @staticmethod
    def _pair_key(agent_a: str, agent_b: str) -> str:
        """生成无序的配对键"""
        return "|".join(sorted([agent_a, agent_b]))

    # ---- thread_id 管理 ----

    def get_thread_id(self, agent_a: str, agent_b: str) -> str:
        """
        获取本次会话的 thread_id。
        如果是新会话（无状态），自动生成新的 thread_id；
        如果会话进行中，返回同一个 thread_id 保持上下文连续。
        """
        key = self._pair_key(agent_a, agent_b)
        state = self._pairs.get(key)
        if state is None or not state.thread_id:
            return ""
        return state.thread_id

    def get_or_create_thread_id(self, agent_a: str, agent_b: str) -> str:
        """
        获取或创建本次会话的 thread_id。
        会话不存在时自动创建状态并生成新 thread_id，
        会话进行中返回已有 thread_id 保证上下文连续。
        """
        key = self._pair_key(agent_a, agent_b)
        if key not in self._pairs:
            self._pairs[key] = ConversationState(
                agent_a=agent_a,
                agent_b=agent_b,
                max_rounds=self.default_max_rounds,
            )
        state = self._pairs[key]
        if not state.thread_id:
            # 新会话：生成独立的 thread_id
            state.thread_id = f"private_{key}_{uuid.uuid4()}"
            logger.info(f"新会话 thread_id: {state.thread_id} ({agent_a} <-> {agent_b})")
        return state.thread_id

    # ---- 查询接口 ----

    def can_send(self, agent_from: str, agent_to: str) -> Tuple[bool, str]:
        """
        检查 agent_from 是否可以向 agent_to 发送消息。
        返回 (允许, 拒绝原因)。
        """
        key = self._pair_key(agent_from, agent_to)
        state = self._pairs.get(key)
        if state is None:
            return True, ""

        if state.is_capped:
            elapsed = time.time() - state.capped_at
            cooldown_sec = self.cooldown_minutes * 60
            if elapsed < cooldown_sec:
                remaining = int(cooldown_sec - elapsed)
                return False, (
                    f"你与 {agent_to} 的对话已达到 {state.max_rounds} 轮上限，"
                    f"已被限制。请在 {remaining} 秒后重试，或请人类用户介入解除限制。"
                )
            else:
                # 冷却到期，自动重置
                del self._pairs[key]
                logger.info(f"对话对 {key} 冷却到期，自动重置")
                return True, ""

        return True, ""

    def get_warning(self, agent_a: str, agent_b: str) -> Optional[str]:
        """获取当前对话的警告提示，用于注入到 system prompt。"""
        key = self._pair_key(agent_a, agent_b)
        state = self._pairs.get(key)
        if state is None:
            return None
        return state.get_warning_hint()

    def get_state(self, agent_a: str, agent_b: str) -> Optional[ConversationState]:
        """获取对话状态（只读），用于监控。"""
        key = self._pair_key(agent_a, agent_b)
        return self._pairs.get(key)

    def is_capped(self, agent_a: str, agent_b: str) -> bool:
        """检查对话对是否已被硬截断。"""
        key = self._pair_key(agent_a, agent_b)
        state = self._pairs.get(key)
        if state is None:
            return False
        return state.is_capped

    # ---- 记录接口 ----

    def record_send(
        self, agent_from: str, agent_to: str, task_description: str = ""
    ) -> Tuple[str, Optional[str]]:
        """
        记录一次发送，递增轮次。首次调用自动创建会话状态和 thread_id。
        返回 (等级, 警告文本)。

        等级: 'ok' | 'warning' | 'final_warning' | 'capped'
        """
        key = self._pair_key(agent_from, agent_to)
        if key not in self._pairs:
            self._pairs[key] = ConversationState(
                agent_a=agent_from,
                agent_b=agent_to,
                max_rounds=self.default_max_rounds,
            )
        state = self._pairs[key]
        # 首次发送时生成 thread_id
        if not state.thread_id:
            state.thread_id = f"private_{key}_{uuid.uuid4()}"
            logger.info(f"新会话 thread_id: {state.thread_id} ({agent_from} -> {agent_to})")

        if task_description and not state.task_description:
            state.task_description = task_description

        level = state.increment()
        warning = state.get_warning_hint()
        return level, warning

    # ---- 管理接口 ----

    def reset(self, agent_a: str, agent_b: str):
        """手动重置一对智能体的对话状态。下次通信将生成新 thread_id。"""
        key = self._pair_key(agent_a, agent_b)
        if key in self._pairs:
            old_thread = self._pairs[key].thread_id
            del self._pairs[key]
            logger.info(f"对话对 {key} 已被手动重置 (旧 thread_id={old_thread})")

    def reset_all_for(self, agent_id: str):
        """重置某个智能体参与的所有对话。"""
        to_remove = [k for k in self._pairs if agent_id in k.split("|")]
        for k in to_remove:
            logger.info(f"已重置 {agent_id} 的对话对 {k} (thread_id={self._pairs[k].thread_id})")
            del self._pairs[k]
        if to_remove:
            logger.info(f"共重置 {agent_id} 的 {len(to_remove)} 个对话对")

    def get_summary(self) -> list:
        """获取所有活跃对话的摘要，用于调试/监控。"""
        result = []
        for key, state in self._pairs.items():
            result.append({
                "agents": key.split("|"),
                "rounds": f"{state.round_count}/{state.max_rounds}",
                "capped": state.is_capped,
                "thread_id": state.thread_id[:60] if state.thread_id else "-",
                "task": state.task_description[:80] if state.task_description else "-",
            })
        return result
