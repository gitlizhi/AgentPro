"""
多智能体任务编排模块
在 TDP 之上提供任务分解、并行派发、进度聚合能力。
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SubTaskStatus(str, Enum):
    PENDING = "pending"          # 尚未派发
    DISPATCHED = "dispatched"    # 已派发，等待对方接受
    IN_PROGRESS = "in_progress"  # 对方已接受，正在执行
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 执行失败


class PlanState(str, Enum):
    PLANNING = "planning"                   # 空计划，subtasks 未填充
    READY = "ready"                         # 已填充子任务，等待派发
    EXECUTING = "executing"                 # 至少一个子任务已派发
    COMPLETED = "completed"                 # 全部成功
    PARTIALLY_COMPLETED = "partially_completed"  # 部分成功
    FAILED = "failed"                       # 全部失败
    CANCELLED = "cancelled"                 # 用户取消


@dataclass
class SubTask:
    id: str
    description: str
    assigned_to: Optional[str] = None
    status: SubTaskStatus = SubTaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    result: Optional[str] = None
    ticket_id: Optional[str] = None
    suggested_role: Optional[str] = None


@dataclass
class OrchestrationPlan:
    plan_id: str
    description: str
    subtasks: list[SubTask] = field(default_factory=list)
    state: PlanState = PlanState.PLANNING
    issuer: str = ""
    created_at: float = 0.0
    completed_count: int = 0
    failed_count: int = 0

    @property
    def total_count(self) -> int:
        return len(self.subtasks)

    @property
    def progress_pct(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.completed_count + self.failed_count) / self.total_count * 100

    def get_ready_subtasks(self) -> list[SubTask]:
        """获取所有依赖已满足且尚未派发的子任务。"""
        completed_ids = {st.id for st in self.subtasks if st.status == SubTaskStatus.COMPLETED}
        ready = []
        for st in self.subtasks:
            if st.status != SubTaskStatus.PENDING:
                continue
            if all(dep_id in completed_ids for dep_id in st.depends_on):
                ready.append(st)
        return ready

    def is_complete(self) -> bool:
        return all(
            st.status in (SubTaskStatus.COMPLETED, SubTaskStatus.FAILED)
            for st in self.subtasks
        )


class OrchestrationManager:
    """管理所有编排计划的生命周期。"""

    def __init__(self):
        self._plans: dict[str, OrchestrationPlan] = {}
        # ticket_id → (plan_id, subtask_id) 反查索引
        self._ticket_index: dict[str, tuple[str, str]] = {}
        self._pool = None  # 可选: 持久化连接池
        # plan_id → (timestamp, progress_snapshot) 用于防抖检测
        self._last_check: dict[str, tuple[float, str]] = {}
        self._DEBOUNCE_SECONDS = 20  # 同一 plan 最小查询间隔

    def set_pool(self, pool):
        """注入数据库连接池，启用持久化"""
        self._pool = pool

    # ── 持久化 helpers ──

    async def _save_plan(self, plan: OrchestrationPlan):
        """将计划及其所有子任务持久化到数据库。"""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                await conn.execute("""
                    INSERT INTO orchestration_plans
                        (plan_id, description, issuer, state, completed_count,
                         failed_count, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (plan_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        completed_count = EXCLUDED.completed_count,
                        failed_count = EXCLUDED.failed_count,
                        updated_at = EXCLUDED.updated_at
                """, (
                    plan.plan_id, plan.description, plan.issuer,
                    plan.state.value, plan.completed_count, plan.failed_count,
                    plan.created_at, time.time(),
                ))
                # 先删后插子任务（简单粗暴但可靠）
                await conn.execute(
                    "DELETE FROM orchestration_subtasks WHERE plan_id = %s",
                    (plan.plan_id,),
                )
                for st in plan.subtasks:
                    await conn.execute("""
                        INSERT INTO orchestration_subtasks
                            (plan_id, subtask_id, description, assigned_to,
                             status, depends_on, result, ticket_id, suggested_role)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        plan.plan_id, st.id, st.description, st.assigned_to,
                        st.status.value, json.dumps(st.depends_on),
                        st.result, st.ticket_id, st.suggested_role,
                    ))
        except Exception:
            logger.warning(f"持久化计划 {plan.plan_id} 失败", exc_info=True)

    async def load_active(self):
        """从数据库恢复所有非终态计划，重建内存状态。"""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                # 恢复非终态计划
                rows = await conn.execute("""
                    SELECT plan_id, description, issuer, state, completed_count,
                           failed_count, created_at
                    FROM orchestration_plans
                    WHERE state NOT IN ('completed', 'partially_completed', 'failed', 'cancelled')
                    ORDER BY created_at
                """)
                plans_data = await rows.fetchall()
                if not plans_data:
                    logger.info("无活跃编排计划需要恢复")
                    return

                for pd_row in plans_data:
                    plan = OrchestrationPlan(
                        plan_id=pd_row[0],
                        description=pd_row[1],
                        issuer=pd_row[2],
                        state=PlanState(pd_row[3]),
                        completed_count=pd_row[4],
                        failed_count=pd_row[5],
                        created_at=pd_row[6],
                    )
                    # 恢复子任务
                    st_rows = await conn.execute("""
                        SELECT subtask_id, description, assigned_to, status,
                               depends_on, result, ticket_id, suggested_role
                        FROM orchestration_subtasks
                        WHERE plan_id = %s
                        ORDER BY subtask_id
                    """, (plan.plan_id,))
                    for st_row in await st_rows.fetchall():
                        depends = json.loads(st_row[4]) if isinstance(st_row[4], str) else (st_row[4] or [])
                        # 规范化：DB 中可能存有整数，统一转字符串
                        depends = [f"st_{d}" if isinstance(d, int) else str(d) for d in (depends or [])]
                        st = SubTask(
                            id=st_row[0],
                            description=st_row[1],
                            assigned_to=st_row[2],
                            status=SubTaskStatus(st_row[3]),
                            depends_on=depends,
                            result=st_row[5],
                            ticket_id=st_row[6],
                            suggested_role=st_row[7],
                        )
                        plan.subtasks.append(st)
                        # 重建 ticket 反查索引
                        if st.ticket_id:
                            self._ticket_index[st.ticket_id] = (plan.plan_id, st.id)

                    self._plans[plan.plan_id] = plan
                    logger.info(
                        f"恢复编排计划 {plan.plan_id}: {plan.description[:60]} "
                        f"({len(plan.subtasks)} 子任务, 状态={plan.state.value})"
                    )

        except Exception:
            logger.error("恢复编排计划失败", exc_info=True)

    # ── 计划生命周期 ──

    async def create_plan(self, description: str, issuer: str) -> OrchestrationPlan:
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        plan = OrchestrationPlan(
            plan_id=plan_id,
            description=description,
            issuer=issuer,
            created_at=time.time(),
        )
        self._plans[plan_id] = plan
        await self._save_plan(plan)
        logger.info(f"Created orchestration plan {plan_id}: {description[:80]}")
        return plan

    async def set_subtasks(self, plan_id: str, subtask_specs: list[dict]) -> OrchestrationPlan:
        """用 LLM 分解结果填充计划的子任务列表。"""
        plan = self._plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        plan.subtasks = []
        for i, spec in enumerate(subtask_specs):
            # 规范化 depends_on：LLM 可能返回整数 [1, 2] 或字符串 ["1", "st_1"]
            raw_deps = spec.get("depends_on", [])
            normalized_deps = []
            for d in (raw_deps or []):
                if isinstance(d, int):
                    normalized_deps.append(f"st_{d}")
                else:
                    s = str(d)
                    normalized_deps.append(f"st_{s}" if s.isdigit() else s)
            plan.subtasks.append(SubTask(
                id=f"st_{i + 1}",
                description=spec["description"],
                depends_on=normalized_deps,
                suggested_role=spec.get("suggested_role", ""),
            ))

        # ── 验证依赖：移除对不存在子任务的引用（防止 LLM 幻觉如 st_0）──
        valid_ids = {st.id for st in plan.subtasks}
        for st in plan.subtasks:
            invalid = [d for d in st.depends_on if d not in valid_ids]
            if invalid:
                logger.warning(
                    f"子任务 {st.id} 引用了不存在的依赖 {invalid}，已自动移除。"
                    f"有效 ID: {sorted(valid_ids)}"
                )
                st.depends_on = [d for d in st.depends_on if d in valid_ids]

        plan.state = PlanState.READY
        await self._save_plan(plan)
        return plan

    def get_plan(self, plan_id: str) -> Optional[OrchestrationPlan]:
        return self._plans.get(plan_id)

    def get_plan_by_ticket(self, ticket_id: str) -> Optional[OrchestrationPlan]:
        """通过 TDP ticket_id 反查所属编排计划。"""
        entry = self._ticket_index.get(ticket_id)
        if not entry:
            return None
        plan_id, _ = entry
        return self._plans.get(plan_id)

    def get_subtask_by_ticket(self, ticket_id: str) -> Optional[tuple[OrchestrationPlan, "SubTask"]]:
        """通过 ticket_id 反查 (plan, subtask)。"""
        entry = self._ticket_index.get(ticket_id)
        if not entry:
            return None
        plan_id, subtask_id = entry
        plan = self._plans.get(plan_id)
        if not plan:
            return None
        st = next((s for s in plan.subtasks if s.id == subtask_id), None)
        return (plan, st) if st else None

    async def dispatch_subtask(self, plan_id: str, subtask_id: str,
                                agent_id: str, ticket_id: str) -> SubTask:
        """标记子任务已派发，建立 ticket ↔ subtask 映射。"""
        plan = self._plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        st = next((s for s in plan.subtasks if s.id == subtask_id), None)
        if not st:
            raise ValueError(f"Subtask {subtask_id} not found in plan {plan_id}")
        st.assigned_to = agent_id
        st.ticket_id = ticket_id
        st.status = SubTaskStatus.DISPATCHED
        self._ticket_index[ticket_id] = (plan_id, subtask_id)
        if plan.state == PlanState.READY:
            plan.state = PlanState.EXECUTING
        await self._save_plan(plan)
        return st

    # ── 状态变更 ──

    async def mark_accepted(self, ticket_id: str):
        """TDP 接受回调：标记子任务为执行中。"""
        result = self.get_subtask_by_ticket(ticket_id)
        if result:
            plan, st = result
            st.status = SubTaskStatus.IN_PROGRESS
            await self._save_plan(plan)

    async def mark_completed(self, ticket_id: str, result_text: str):
        """TDP 交付回调：标记子任务完成。"""
        result = self.get_subtask_by_ticket(ticket_id)
        if not result:
            return
        plan, st = result
        st.status = SubTaskStatus.COMPLETED
        st.result = result_text[:500]
        plan.completed_count += 1
        if plan.is_complete():
            plan.state = PlanState.COMPLETED if plan.failed_count == 0 else PlanState.PARTIALLY_COMPLETED
        await self._save_plan(plan)

    async def mark_failed(self, ticket_id: str, error: str):
        """标记子任务失败。"""
        result = self.get_subtask_by_ticket(ticket_id)
        if not result:
            return
        plan, st = result
        st.status = SubTaskStatus.FAILED
        st.result = error[:500]
        plan.failed_count += 1
        if plan.is_complete():
            plan.state = PlanState.PARTIALLY_COMPLETED if plan.completed_count > 0 else PlanState.FAILED
        await self._save_plan(plan)

    async def cancel_plan(self, plan_id: str):
        plan = self._plans.get(plan_id)
        if plan:
            plan.state = PlanState.CANCELLED
            await self._save_plan(plan)

    # ── 格式化 ──

    def get_progress_summary(self, plan_id: str) -> str:
        plan = self._plans.get(plan_id)
        if not plan:
            return f"未找到计划 {plan_id}"

        # ── 防抖：同一 plan 在 DEBOUNCE_SECONDS 内重复查询且进度无变化时拒绝 ──
        now = time.time()
        snapshot = f"{plan.completed_count}/{plan.total_count}/{plan.failed_count}/{plan.state.value}"
        if plan_id in self._last_check:
            last_ts, last_snapshot = self._last_check[plan_id]
            elapsed = now - last_ts
            if elapsed < self._DEBOUNCE_SECONDS and snapshot == last_snapshot:
                logger.info(
                    f"check_plan_progress 防抖拦截至 {plan_id}: "
                    f"距上次查询仅 {elapsed:.1f}s，进度无变化"
                )
                return (
                    f"⚠️ 距上次查询仅 {elapsed:.0f} 秒，计划进度与上次相同（{plan.completed_count}/{plan.total_count} 完成），"
                    f"无新变化。子任务完成后系统会自动通知你，**请勿频繁轮询**。"
                    f"只有收到通知或用户主动询问时才需要再次查询。"
                )
        self._last_check[plan_id] = (now, snapshot)

        icon_map = {
            "pending": "⏳", "dispatched": "📤", "in_progress": "🔄",
            "completed": "✅", "failed": "❌",
        }
        lines = [
            f"## 任务计划: {plan.description}",
            f"状态: {plan.state.value} | 进度: {plan.completed_count}/{plan.total_count} 完成 ({plan.progress_pct:.0f}%)",
            "---",
        ]
        for st in plan.subtasks:
            icon = icon_map.get(st.status.value, "❓")
            dep_str = f" (依赖: {', '.join(str(d) for d in st.depends_on)})" if st.depends_on else ""
            agent_str = f" → {st.assigned_to}" if st.assigned_to else ""
            result_str = f" — {st.result[:80]}" if st.result else ""
            lines.append(f"{icon} **{st.id}**: {st.description}{dep_str}{agent_str}{result_str}")
        return "\n".join(lines)
