"""
Task Delegation Protocol (TDP) — 任务委托协议核心模块。

将智能体之间的私聊从"自由文本聊天"升级为"结构化任务工单"：
- 每段私聊必须绑定一个 TaskTicket
- 工单有明确的生命周期状态机
- 多重自动终止条件（轮次耗尽、空闲超时、显式交付）
- 所有状态变更对 super_user 可见

替代了旧有的纯轮次计数 ConversationTracker 作为智能体间通信的主要控制机制。
"""

import asyncio
import enum
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

# ── 默认配置 ──────────────────────────────────────────────────────────
DEFAULT_MAX_ROUNDS = 8
DEFAULT_MAX_CLARIFICATIONS = 2
DEFAULT_IDLE_TIMEOUT_MINUTES = 10
WARN_THRESHOLD_1 = 4       # 第4轮开始提示
WARN_THRESHOLD_2 = 6       # 第6轮最后通牒


class TicketState(enum.Enum):
    """工单生命周期状态"""
    # 活跃状态
    PENDING = "pending"            # 委托已发出，等待 assignee 接受/拒绝
    ACCEPTED = "accepted"          # assignee 已接受，尚未开始工作
    NEGOTIATING = "negotiating"    # 澄清需求中（最多2轮）
    IN_PROGRESS = "in_progress"    # 工作中
    # 终态
    CLOSED = "closed"              # 正常完成（deliver_result）
    DECLINED = "declined"          # assignee 拒绝
    TIMED_OUT = "timed_out"        # 空闲超时或轮次耗尽
    CANCELLED = "cancelled"        # 任一方主动取消


# 状态机：每个状态允许转换到的目标状态集合
VALID_TRANSITIONS: dict[TicketState, set[TicketState]] = {
    TicketState.PENDING:      {TicketState.ACCEPTED, TicketState.DECLINED, TicketState.CANCELLED},
    TicketState.ACCEPTED:     {TicketState.IN_PROGRESS, TicketState.NEGOTIATING, TicketState.CANCELLED},
    TicketState.NEGOTIATING:  {TicketState.IN_PROGRESS, TicketState.CANCELLED},
    TicketState.IN_PROGRESS:  {TicketState.CLOSED, TicketState.NEGOTIATING,
                               TicketState.CANCELLED, TicketState.TIMED_OUT},
    # 终态不允许任何转换
    TicketState.CLOSED:     set(),
    TicketState.DECLINED:   set(),
    TicketState.TIMED_OUT:  set(),
    TicketState.CANCELLED:  set(),
}

ACTIVE_STATES = frozenset({
    TicketState.PENDING, TicketState.ACCEPTED,
    TicketState.NEGOTIATING, TicketState.IN_PROGRESS,
})

TERMINAL_STATES = frozenset({
    TicketState.CLOSED, TicketState.DECLINED,
    TicketState.TIMED_OUT, TicketState.CANCELLED,
})


class TicketError(Exception):
    """工单状态机非法操作异常"""
    pass


@dataclass
class TaskTicket:
    """任务工单 — 智能体之间一次委托的完整合约"""

    ticket_id: str
    issuer: str               # 委托方 agent_id
    assignee: str             # 被委托方 agent_id
    description: str           # 任务描述
    expected_output: str       # 期望产出（"完成"的标准）
    max_rounds: int = DEFAULT_MAX_ROUNDS

    state: TicketState = TicketState.PENDING

    # 时序
    created_at: float = field(default_factory=time.time)
    accepted_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_activity: float = field(default_factory=time.time)

    # 轮次追踪
    round_count: int = 0
    clarification_count: int = 0
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS

    # 结果
    result_summary: Optional[str] = None
    cancel_reason: Optional[str] = None
    cancelled_by: Optional[str] = None

    # LangGraph checkpoint 隔离
    thread_id: str = ""

    # 进度报告历史
    progress_updates: list = field(default_factory=list)

    # 编排关联：若此工单属于某个编排计划，存储 plan_id
    orchestration_plan_id: Optional[str] = None

    def __post_init__(self):
        if not self.thread_id:
            self.thread_id = f"ticket_{self.ticket_id}_{uuid.uuid4().hex[:8]}"

    # ── 属性 ──────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def remaining_rounds(self) -> int:
        return max(0, self.max_rounds - self.round_count)

    # ── 警告提示 ──────────────────────────────────────────────────

    def get_warning_hint(self, other_agent: str) -> Optional[str]:
        """获取当前轮次应注入的警告提示。None = 无需警告。"""
        if self.is_terminal:
            return None
        r = self.remaining_rounds
        if r == 1:
            return (
                f"⚠️ 这是你处理工单 {self.ticket_id} 的**最后一轮**。"
                f"请在回复中交付最终结果。不要发起新的子任务或展开新话题。"
            )
        if r == 2:
            return (
                f"⚠️ 工单 {self.ticket_id} 还剩 **2 轮**。"
                f"请汇总已完成的工作，准备交付结果。"
            )
        if self.round_count >= WARN_THRESHOLD_1:
            return (
                f"💡 工单 {self.ticket_id} 已消耗 {self.round_count} 轮（共 {self.max_rounds} 轮）。"
                f"请在 {r} 轮内完成并交付。"
            )
        return None


class DelegationManager:
    """管理所有任务工单的生命周期。

    内部维护两个索引：
    - _by_id: ticket_id → TaskTicket（精确查找）
    - _by_pair: pair_key → ticket_id（同一对智能体只有一个活跃工单）
    """

    def __init__(self):
        self._by_id: Dict[str, TaskTicket] = {}
        self._by_pair: Dict[str, List[str]] = {}  # 支持同对智能体的多个并行工单
        self._pool = None  # 持久化连接池

    def set_pool(self, pool):
        """注入数据库连接池，启用持久化"""
        self._pool = pool

    # ── 持久化 helpers ──

    def _save_ticket(self, ticket: TaskTicket):
        """将工单状态异步持久化到数据库（fire-and-forget）。"""
        if not self._pool:
            return

        async def _do():
            try:
                async with self._pool.connection() as conn:
                    await conn.execute("""
                        INSERT INTO delegation_tickets
                            (ticket_id, issuer, assignee, description, expected_output,
                             max_rounds, state, round_count, clarification_count,
                             result_summary, cancel_reason, cancelled_by, thread_id,
                             created_at, accepted_at, completed_at, last_activity,
                             orchestration_plan_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticket_id) DO UPDATE SET
                            state = EXCLUDED.state,
                            round_count = EXCLUDED.round_count,
                            clarification_count = EXCLUDED.clarification_count,
                            result_summary = EXCLUDED.result_summary,
                            cancel_reason = EXCLUDED.cancel_reason,
                            cancelled_by = EXCLUDED.cancelled_by,
                            accepted_at = EXCLUDED.accepted_at,
                            completed_at = EXCLUDED.completed_at,
                            last_activity = EXCLUDED.last_activity,
                            orchestration_plan_id = EXCLUDED.orchestration_plan_id
                    """, (
                        ticket.ticket_id, ticket.issuer, ticket.assignee,
                        ticket.description, ticket.expected_output,
                        ticket.max_rounds, ticket.state.value,
                        ticket.round_count, ticket.clarification_count,
                        ticket.result_summary, ticket.cancel_reason,
                        ticket.cancelled_by, ticket.thread_id,
                        ticket.created_at, ticket.accepted_at,
                        ticket.completed_at, ticket.last_activity,
                        ticket.orchestration_plan_id,
                    ))
            except Exception:
                logger.warning(f"持久化工单 {ticket.ticket_id} 失败", exc_info=True)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do())
        except RuntimeError:
            pass  # 无运行中的事件循环，跳过持久化

    async def load_active(self):
        """从数据库恢复所有活跃（非终态）工单，重建内存状态。"""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                rows = await conn.execute("""
                    SELECT ticket_id, issuer, assignee, description, expected_output,
                           max_rounds, state, round_count, clarification_count,
                           result_summary, cancel_reason, cancelled_by, thread_id,
                           created_at, accepted_at, completed_at, last_activity,
                           orchestration_plan_id
                    FROM delegation_tickets
                    WHERE state NOT IN ('closed', 'declined', 'timed_out', 'cancelled')
                    ORDER BY created_at
                """)
                for row in await rows.fetchall():
                    ticket = TaskTicket(
                        ticket_id=row[0],
                        issuer=row[1],
                        assignee=row[2],
                        description=row[3],
                        expected_output=row[4],
                        max_rounds=row[5],
                    )
                    ticket.state = TicketState(row[6])
                    ticket.round_count = row[7]
                    ticket.clarification_count = row[8]
                    ticket.result_summary = row[9]
                    ticket.cancel_reason = row[10]
                    ticket.cancelled_by = row[11]
                    ticket.thread_id = row[12] or ""
                    ticket.created_at = row[13]
                    ticket.accepted_at = row[14]
                    ticket.completed_at = row[15]
                    ticket.last_activity = row[16]
                    ticket.orchestration_plan_id = row[17] or None

                    if not ticket.thread_id:
                        ticket.thread_id = f"ticket_{ticket.ticket_id}_{uuid.uuid4().hex[:8]}"

                    self._by_id[ticket.ticket_id] = ticket
                    key = self._pair_key(ticket.issuer, ticket.assignee)
                    if key not in self._by_pair:
                        self._by_pair[key] = []
                    self._by_pair[key].append(ticket.ticket_id)

                    logger.info(
                        f"恢复工单 {ticket.ticket_id}: {ticket.issuer}→{ticket.assignee} "
                        f"({ticket.state.value}, {ticket.round_count}/{ticket.max_rounds} 轮)"
                    )
        except Exception:
            logger.error("恢复委托工单失败", exc_info=True)

    @staticmethod
    def _pair_key(agent_a: str, agent_b: str) -> str:
        """无序配对键"""
        return "|".join(sorted([agent_a, agent_b]))

    # ── 工单创建与查询 ────────────────────────────────────────────

    def create_ticket(
        self,
        issuer: str,
        assignee: str,
        description: str,
        expected_output: str,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        allow_parallel: bool = False,
    ) -> TaskTicket:
        """创建新的委托工单。同一对智能体已有活跃工单时拒绝（除非 allow_parallel=True）。

        allow_parallel: 编排场景下允许多个子任务并行派发给同一智能体。
        """
        key = self._pair_key(issuer, assignee)

        # 检查是否已有活跃工单（遍历列表，同时清理已终止的工单）
        existing_ids = self._by_pair.get(key, [])
        # 过滤掉已终止的工单，保持列表干净
        active_ids = [tid for tid in existing_ids
                      if self._by_id.get(tid) and self._by_id[tid].is_active]
        self._by_pair[key] = active_ids

        if not allow_parallel and active_ids:
            existing = self._by_id[active_ids[0]]
            raise TicketError(
                f"已存在活跃工单 {existing.ticket_id}（{existing.description[:50]}...），"
                f"请先完成或取消该工单后再创建新的。"
            )

        ticket_id = str(uuid.uuid4())[:8]
        ticket = TaskTicket(
            ticket_id=ticket_id,
            issuer=issuer,
            assignee=assignee,
            description=description,
            expected_output=expected_output,
            max_rounds=max_rounds,
        )

        self._by_id[ticket_id] = ticket
        if key not in self._by_pair:
            self._by_pair[key] = []
        self._by_pair[key].append(ticket_id)
        self._save_ticket(ticket)

        logger.info(
            f"工单已创建: {ticket_id} ({issuer} -> {assignee}) "
            f"\"{description[:80]}\", max_rounds={max_rounds}"
        )
        return ticket

    def get_ticket(self, ticket_id: str) -> Optional[TaskTicket]:
        """按 ID 查找工单"""
        return self._by_id.get(ticket_id)

    def get_active_ticket(self, agent_a: str, agent_b: str) -> Optional[TaskTicket]:
        """查找两个智能体之间的活跃工单（返回第一个活跃的）"""
        key = self._pair_key(agent_a, agent_b)
        ticket_ids = self._by_pair.get(key, [])
        for tid in ticket_ids:
            ticket = self._by_id.get(tid)
            if ticket and ticket.is_active:
                return ticket
        return None

    def get_active_tickets_by_pair(self, agent_a: str, agent_b: str) -> List[TaskTicket]:
        """查找两个智能体之间的所有活跃工单（支持并行工单场景）"""
        key = self._pair_key(agent_a, agent_b)
        ticket_ids = self._by_pair.get(key, [])
        result = []
        for tid in ticket_ids:
            ticket = self._by_id.get(tid)
            if ticket and ticket.is_active:
                result.append(ticket)
        return result

    def get_active_tickets_for_agent(self, agent_id: str) -> List[TaskTicket]:
        """获取某个智能体参与的所有活跃工单"""
        result = []
        for ticket in self._by_id.values():
            if ticket.is_active and agent_id in (ticket.issuer, ticket.assignee):
                result.append(ticket)
        return result

    # ── 状态机 ────────────────────────────────────────────────────

    def transition(self, ticket_id: str, new_state: TicketState) -> TaskTicket:
        """执行工单状态转换。转换不合法时抛出 TicketError。"""
        ticket = self._by_id.get(ticket_id)
        if not ticket:
            raise TicketError(f"工单 {ticket_id} 不存在")

        allowed = VALID_TRANSITIONS.get(ticket.state, set())
        if new_state not in allowed:
            raise TicketError(
                f"不允许从 {ticket.state.value} 转换到 {new_state.value}。"
                f"允许的目标状态: {[s.value for s in allowed]}"
            )

        old_state = ticket.state
        ticket.state = new_state
        ticket.last_activity = time.time()

        if new_state == TicketState.ACCEPTED:
            ticket.accepted_at = time.time()
        elif new_state in TERMINAL_STATES:
            ticket.completed_at = time.time()
            # 终态从 pair 列表中移除该工单（支持同对并行工单）
            key = self._pair_key(ticket.issuer, ticket.assignee)
            if key in self._by_pair and ticket_id in self._by_pair[key]:
                self._by_pair[key].remove(ticket_id)
                if not self._by_pair[key]:
                    del self._by_pair[key]

        self._save_ticket(ticket)
        logger.info(
            f"工单 {ticket_id}: {old_state.value} → {new_state.value}"
        )
        return ticket

    # ── 轮次管理 ──────────────────────────────────────────────────

    def record_round(self, ticket_id: str) -> Tuple[str, Optional[str]]:
        """记录一轮通信。返回 (level, warning_text)。
        level: 'ok' | 'warning' | 'final_warning' | 'capped'
        """
        ticket = self._by_id.get(ticket_id)
        if not ticket:
            raise TicketError(f"工单 {ticket_id} 不存在")
        if ticket.is_terminal:
            raise TicketError(f"工单 {ticket_id} 已终止，无法记录轮次")

        ticket.round_count += 1
        ticket.last_activity = time.time()

        if ticket.round_count >= ticket.max_rounds:
            self.transition(ticket_id, TicketState.TIMED_OUT)
            return "capped", None

        self._save_ticket(ticket)
        if ticket.round_count >= WARN_THRESHOLD_2:
            return "final_warning", ticket.get_warning_hint(ticket.assignee)
        if ticket.round_count >= WARN_THRESHOLD_1:
            return "warning", ticket.get_warning_hint(ticket.assignee)
        return "ok", None

    def record_clarification(self, ticket_id: str) -> bool:
        """记录一轮澄清。返回 True 表示允许，False 表示超限。"""
        ticket = self._by_id.get(ticket_id)
        if not ticket:
            raise TicketError(f"工单 {ticket_id} 不存在")
        if ticket.is_terminal:
            return False

        if ticket.clarification_count >= ticket.max_clarifications:
            return False

        ticket.clarification_count += 1
        # 澄清也计入总轮次（占用工作预算）
        ticket.round_count += 1
        ticket.last_activity = time.time()
        self._save_ticket(ticket)
        return True

    # ── 交付与取消 ────────────────────────────────────────────────

    def deliver_result(self, ticket_id: str, summary: str) -> TaskTicket:
        """交付任务结果，工单正常关闭。

        自动处理 ACCEPTED 状态的工单：先推进到 IN_PROGRESS 再关闭，
        避免 ACCEPTED→CLOSED 的死锁问题（assignee 接受后无工具能推进到 IN_PROGRESS）。
        """
        ticket = self._by_id.get(ticket_id)
        if not ticket:
            raise TicketError(f"工单 {ticket_id} 不存在")
        if ticket.is_terminal:
            raise TicketError(f"工单 {ticket_id} 已终止")

        # 自动桥接 ACCEPTED → IN_PROGRESS（消除状态死锁）
        if ticket.state == TicketState.ACCEPTED:
            self.transition(ticket_id, TicketState.IN_PROGRESS)
            logger.info(f"工单 {ticket_id}: 自动从 accepted 推进到 in_progress")

        ticket.result_summary = summary
        self.transition(ticket_id, TicketState.CLOSED)
        logger.info(f"工单 {ticket_id} 已交付: {summary[:100]}")
        return ticket

    def cancel_ticket(self, ticket_id: str, reason: str, cancelled_by: str) -> TaskTicket:
        """取消工单（任一方或系统均可调用）。"""
        ticket = self._by_id.get(ticket_id)
        if not ticket:
            raise TicketError(f"工单 {ticket_id} 不存在")
        if ticket.is_terminal:
            # 已是终态，幂等返回
            return ticket

        ticket.cancel_reason = reason
        ticket.cancelled_by = cancelled_by
        self.transition(ticket_id, TicketState.CANCELLED)
        logger.info(f"工单 {ticket_id} 被 {cancelled_by} 取消: {reason}")
        return ticket

    # ── 超时检测 ──────────────────────────────────────────────────

    def check_timeouts(self, idle_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES) -> List[TaskTicket]:
        """检查并自动超时关闭空闲工单。返回被超时的工单列表。"""
        now = time.time()
        timeout_seconds = idle_minutes * 60
        timed_out = []

        for ticket in list(self._by_id.values()):
            if not ticket.is_active:
                continue
            if now - ticket.last_activity > timeout_seconds:
                try:
                    ticket.cancel_reason = f"空闲超过 {idle_minutes} 分钟，自动超时"
                    ticket.cancelled_by = "system"
                    self.transition(ticket.ticket_id, TicketState.TIMED_OUT)
                    timed_out.append(ticket)
                except TicketError:
                    pass  # 并发情况下状态已变更

        return timed_out

    # ── 警告注入 ──────────────────────────────────────────────────

    def get_warning(self, ticket_id: str) -> Optional[str]:
        """获取工单的轮次警告提示，用于注入到 system prompt。"""
        ticket = self._by_id.get(ticket_id)
        if not ticket:
            return None
        return ticket.get_warning_hint("")

    # ── 摘要 ──────────────────────────────────────────────────────

    def get_summary(self) -> List[dict]:
        """获取所有工单的摘要，用于监控和用户界面。"""
        result = []
        for ticket in self._by_id.values():
            result.append({
                "ticket_id": ticket.ticket_id,
                "issuer": ticket.issuer,
                "assignee": ticket.assignee,
                "description": ticket.description[:80],
                "state": ticket.state.value,
                "rounds": f"{ticket.round_count}/{ticket.max_rounds}",
                "clarifications": f"{ticket.clarification_count}/{ticket.max_clarifications}",
                "thread_id": ticket.thread_id[:60] if ticket.thread_id else "-",
                "result": (ticket.result_summary or ticket.cancel_reason or "-")[:80],
            })
        return result

    # ── 清理 ──────────────────────────────────────────────────────

    def cancel_all_for_agent(self, agent_id: str, reason: str = "Agent shutting down"):
        """取消某个智能体参与的所有活跃工单（用于 Agent 下线时）。"""
        for ticket in list(self._by_id.values()):
            if ticket.is_active and agent_id in (ticket.issuer, ticket.assignee):
                try:
                    self.cancel_ticket(ticket.ticket_id, reason, agent_id)
                except TicketError:
                    pass
