"""
多智能体任务编排模块
在 TDP 之上提供任务分解、并行派发、进度聚合能力。
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SubTaskStatus(str, Enum):
    PENDING = "pending"          # 尚未派发
    DISPATCHED = "dispatched"    # 已派发，等待对方接受
    IN_PROGRESS = "in_progress"  # 对方已接受，正在执行
    REVIEWING = "reviewing"      # 审核中（交付后，待审核 Agent 审核）
    APPROVED = "approved"        # 审核通过
    BLOCKED = "blocked"          # 遇到困境，已触发升级
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 执行失败
    SKIPPED = "skipped"          # 被跳过（降级策略）


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

    # ===== Loop Engineering 新增字段 =====
    reviewer_agent: Optional[str] = None
    review_feedback: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    blocked_reason: Optional[str] = None
    attempts: list[dict] = field(default_factory=list)
    worker_system_prompt: Optional[str] = None
    reviewer_system_prompt: Optional[str] = None
    escalated_at: Optional[datetime] = None
    skipped: bool = False


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

    # ===== Loop Engineering 新增字段 =====
    project_overview: str = ""
    critical_decisions: list[str] = field(default_factory=list)
    escalation_log: list[dict] = field(default_factory=list)
    created_from_clarification: bool = False
    agent_pool: dict[str, dict] = field(default_factory=dict)
    degradation_level: int = 0  # 0=正常, 1=LLM降级, 2=网络降级, 3=安全模式

    @property
    def total_count(self) -> int:
        return len(self.subtasks)

    @property
    def progress_pct(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.completed_count + self.failed_count) / self.total_count * 100

    def get_subtask(self, subtask_id: str) -> Optional[SubTask]:
        """按 ID 获取子任务。"""
        for st in self.subtasks:
            if st.id == subtask_id:
                return st
        return None

    def get_ready_subtasks(self) -> list[SubTask]:
        """获取所有依赖已满足且尚未派发的子任务。"""
        terminal_ids = {st.id for st in self.subtasks
                        if st.status in (SubTaskStatus.COMPLETED, SubTaskStatus.APPROVED)}
        ready = []
        for st in self.subtasks:
            if st.status != SubTaskStatus.PENDING:
                continue
            if all(dep_id in terminal_ids for dep_id in st.depends_on):
                ready.append(st)
        return ready

    def is_complete(self) -> bool:
        """检查是否所有子任务都处于终态。"""
        terminal = {SubTaskStatus.COMPLETED, SubTaskStatus.FAILED,
                    SubTaskStatus.APPROVED, SubTaskStatus.SKIPPED}
        return all(st.status in terminal for st in self.subtasks)


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
                         failed_count, created_at, updated_at,
                         project_overview, critical_decisions, escalation_log,
                         created_from_clarification, agent_pool, degradation_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (plan_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        completed_count = EXCLUDED.completed_count,
                        failed_count = EXCLUDED.failed_count,
                        updated_at = EXCLUDED.updated_at,
                        project_overview = EXCLUDED.project_overview,
                        critical_decisions = EXCLUDED.critical_decisions,
                        escalation_log = EXCLUDED.escalation_log,
                        created_from_clarification = EXCLUDED.created_from_clarification,
                        agent_pool = EXCLUDED.agent_pool,
                        degradation_level = EXCLUDED.degradation_level
                """, (
                    plan.plan_id, plan.description, plan.issuer,
                    plan.state.value, plan.completed_count, plan.failed_count,
                    plan.created_at, time.time(),
                    plan.project_overview,
                    json.dumps(plan.critical_decisions),
                    json.dumps(plan.escalation_log),
                    plan.created_from_clarification,
                    json.dumps(plan.agent_pool),
                    plan.degradation_level,
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
                             status, depends_on, result, ticket_id, suggested_role,
                             reviewer_agent, review_feedback, retry_count,
                             max_retries, blocked_reason, attempts,
                             worker_system_prompt, reviewer_system_prompt,
                             escalated_at, skipped)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        plan.plan_id, st.id, st.description, st.assigned_to,
                        st.status.value, json.dumps(st.depends_on),
                        st.result, st.ticket_id, st.suggested_role,
                        st.reviewer_agent, st.review_feedback, st.retry_count,
                        st.max_retries, st.blocked_reason,
                        json.dumps(st.attempts),
                        st.worker_system_prompt, st.reviewer_system_prompt,
                        st.escalated_at.isoformat() if st.escalated_at else None,
                        st.skipped,
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
                           failed_count, created_at,
                           project_overview, critical_decisions, escalation_log,
                           created_from_clarification, agent_pool, degradation_level
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
                        project_overview=pd_row[7] or "",
                        critical_decisions=json.loads(pd_row[8]) if pd_row[8] and isinstance(pd_row[8], str) else (pd_row[8] or []),
                        escalation_log=json.loads(pd_row[9]) if pd_row[9] and isinstance(pd_row[9], str) else (pd_row[9] or []),
                        created_from_clarification=bool(pd_row[10]) if pd_row[10] is not None else False,
                        agent_pool=json.loads(pd_row[11]) if pd_row[11] and isinstance(pd_row[11], str) else (pd_row[11] or {}),
                        degradation_level=pd_row[12] or 0,
                    )
                    # 恢复子任务
                    st_rows = await conn.execute("""
                        SELECT subtask_id, description, assigned_to, status,
                               depends_on, result, ticket_id, suggested_role,
                               reviewer_agent, review_feedback, retry_count,
                               max_retries, blocked_reason, attempts,
                               worker_system_prompt, reviewer_system_prompt,
                               escalated_at, skipped
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
                            reviewer_agent=st_row[8],
                            review_feedback=st_row[9],
                            retry_count=st_row[10] or 0,
                            max_retries=st_row[11] or 3,
                            blocked_reason=st_row[12],
                            attempts=json.loads(st_row[13]) if st_row[13] and isinstance(st_row[13], str) else (st_row[13] or []),
                            worker_system_prompt=st_row[14],
                            reviewer_system_prompt=st_row[15],
                            escalated_at=datetime.fromisoformat(st_row[16]) if st_row[16] else None,
                            skipped=bool(st_row[17]) if st_row[17] is not None else False,
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
                worker_system_prompt=spec.get("worker_prompt", ""),
                reviewer_system_prompt=spec.get("reviewer_prompt", ""),
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

    # ── Agent 生命周期管理 ──

    def get_idle_agent(self, plan_id: str, role: str) -> Optional[str]:
        """获取同 plan 下空闲的同角色 Agent。"""
        plan = self._plans.get(plan_id)
        if not plan:
            return None
        pool = plan.agent_pool
        for agent_id, info in pool.items():
            if info.get("role") == role and info.get("status") == "idle":
                return agent_id
        return None

    def mark_agent_busy(self, plan_id: str, agent_id: str):
        """标记 Agent 为忙碌。"""
        plan = self._plans.get(plan_id)
        if plan and agent_id in plan.agent_pool:
            plan.agent_pool[agent_id]["status"] = "busy"

    def mark_agent_idle(self, plan_id: str, agent_id: str):
        """标记 Agent 为空闲。"""
        plan = self._plans.get(plan_id)
        if plan and agent_id in plan.agent_pool:
            plan.agent_pool[agent_id]["status"] = "idle"

    async def ensure_agent(self, plan_id: str, role: str, system_prompt: str,
                           brain=None, timeout: float = 30.0) -> str:
        """确保某个角色有可用的 Agent。返回 agent_id。

        Args:
            plan_id: 编排计划 ID
            role: 角色名（worker / reviewer）
            system_prompt: Agent 启动时使用的系统提示词
            brain: Brain 实例（用于 launch_agent 和 获取在线列表）
            timeout: 等待 Agent 在线的最大秒数
        """
        # 1. 尝试复用空闲 Agent
        idle = self.get_idle_agent(plan_id, role)
        if idle:
            self.mark_agent_busy(plan_id, idle)
            await self._save_plan(self._plans[plan_id])
            logger.info(f"复用空闲 Agent {idle} (role={role}) for plan {plan_id}")
            return idle

        # 2. 启动新 Agent
        import time as _time
        agent_name = f"{role}_{plan_id[:6]}_{int(_time.time())}"

        if brain is None:
            raise RuntimeError(f"无法启动 Agent {agent_name}：brain 引用不可用")

        try:
            await brain.launch_agent.afunc(agent_name, system_prompt)
        except Exception as e:
            raise RuntimeError(f"启动 Agent {agent_name} 失败: {e}")

        # 3. 等待 Agent 上线
        for i in range(int(timeout)):
            try:
                # 通过发送 get_agents 请求来获取最新在线列表
                await brain.comm.send({"type": "get_agents"})
            except Exception:
                pass
            await asyncio.sleep(1)
            if brain.online_agents and agent_name in brain.online_agents:
                plan = self._plans[plan_id]
                plan.agent_pool[agent_name] = {
                    "role": role,
                    "status": "busy",
                    "started_at": datetime.now().isoformat(),
                }
                await self._save_plan(plan)
                logger.info(f"Agent {agent_name} (role={role}) 已上线 for plan {plan_id}")
                return agent_name

        raise TimeoutError(f"Agent {agent_name} 启动超时 ({timeout}s)")

    async def recover_crashed_agent(self, plan_id: str, subtask_id: str, brain=None) -> bool:
        """检测并恢复崩溃的 Agent。返回 True 表示执行了恢复。"""
        plan = self._plans.get(plan_id)
        if not plan:
            return False
        st = plan.get_subtask(subtask_id)
        if not st or st.status in (SubTaskStatus.COMPLETED, SubTaskStatus.APPROVED,
                                     SubTaskStatus.SKIPPED, SubTaskStatus.FAILED):
            return False

        # 检查 Agent 是否在线
        if brain and brain.online_agents and st.assigned_to in brain.online_agents:
            return False  # 在线，不需要恢复

        # Agent 已离线，重新启动
        if brain:
            await brain._safe_send(
                f"🔄 Agent {st.assigned_to} 已离线，正在重启..."
            )

        try:
            new_agent = await self.ensure_agent(
                plan_id, "worker",
                st.worker_system_prompt or f"执行任务: {st.description}",
                brain=brain,
            )
            st.assigned_to = new_agent
            st.status = SubTaskStatus.PENDING
            await self._save_plan(plan)
            logger.info(f"已恢复 Agent: {new_agent} for subtask {subtask_id}")
            return True
        except Exception as e:
            logger.error(f"恢复 Agent 失败 for {subtask_id}: {e}")
            return False

    async def cleanup_agents(self, plan_id: str, brain=None):
        """清理计划关联的所有子 Agent。"""
        plan = self._plans.get(plan_id)
        if not plan:
            return
        pool = plan.agent_pool
        for agent_id in list(pool.keys()):
            if brain:
                try:
                    await brain.stop_agent.afunc(agent_id)
                except Exception as e:
                    logger.warning(f"清理 Agent {agent_id} 失败: {e}")
            else:
                logger.warning(f"无法清理 Agent {agent_id}：brain 引用不可用")
        plan.agent_pool.clear()
        await self._save_plan(plan)
        logger.info(f"已清理计划 {plan_id} 的所有子 Agent")

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
            "pending": "\u23f3", "dispatched": "\U0001f4e4", "in_progress": "\U0001f504",
            "reviewing": "\U0001f50d", "approved": "\u2705", "blocked": "\U0001f6a8",
            "completed": "\u2705", "failed": "\u274c", "skipped": "\u23ed\ufe0f",
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
