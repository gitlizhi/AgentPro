"""
Standalone tool-creation functions for the Brain agent.
Extracted from brain.py to keep the file size manageable.
Each function receives a Brain instance and returns @tool-decorated callables
that capture `brain_ref` for accessing Brain state at call time.
"""
import asyncio
import json as _json
import logging
import uuid
from datetime import datetime

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from agent.delegation import (
    DelegationManager, TaskTicket, TicketState, TicketError,
    DEFAULT_MAX_ROUNDS,
)
from agent.orchestration import OrchestrationManager, SubTaskStatus, PlanState
from agent.prompts import build_task_decomposition_prompt
from agent.utils import call_big_model_chat

logger = logging.getLogger(__name__)


def create_get_current_time_tool(brain):
    """获取当前时间的工具（无状态，不需要 brain 引用）"""

    @tool
    async def get_current_time() -> str:
        """获取当前的日期和时间。
        当需要知道现在是什么时间、日期、星期几，或涉及时间计算、时效性判断时调用此工具。
        返回当前时间的完整信息，包括日期、时间、星期和时区。"""
        from datetime import datetime, timezone, timedelta
        now = datetime.now()
        return (
            f"当前时间：\n"
            f"日期：{now.strftime('%Y年%m月%d日')}\n"
            f"时间：{now.strftime('%H:%M:%S')}\n"
            f"星期：{['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][now.weekday()]}\n"
            f"ISO 格式：{now.isoformat()}"
        )

    return get_current_time


def create_load_user_profile_tool(brain):
    """按需加载用户画像工具"""
    brain_ref = brain

    @tool
    async def load_user_profile() -> str:
        """加载人类用户（super_user）的个人画像信息，包括身份、偏好、习惯、长期约定等。
        当需要了解用户背景以提供个性化回复时调用此工具。
        返回格式化的用户画像文本，可直接用于个性化对话。"""
        if not brain_ref.memory:
            return "记忆系统未启用，无法获取用户画像。"
        return brain_ref.memory.get_user_profile("super_user")

    return load_user_profile


def create_list_online_agents_tool(brain):
    """查询在线智能体工具"""
    brain_ref = brain

    @tool
    def list_online_agents() -> str:
        """
        查询当前在线的智能体列表（不含用户和系统机器人）。
        用于在和其他 Agent 协作前判断对方是否在线，避免向离线的 Agent 发送消息。
        :return: JSON 格式的在线 Agent 列表
        """
        agents = brain_ref.peer_agents
        if agents is None:
            return '{"error": "在线列表未初始化", "agents": []}'
        return _json.dumps({"agents": sorted(list(agents)), "count": len(agents)}, ensure_ascii=False)

    return list_online_agents


def create_send_to_agent_tool(brain):
    """向其他 Agent 发送消息工具。现已集成 TDP 工单门控：
    - 无活跃工单时允许发送（兼容旧行为），但建议使用 delegate_task 创建工单
    - 有活跃工单时在工单内通信，受工单轮次预算约束
    - [停止交流] 标记会取消当前工单
    """
    brain_ref = brain
    max_rounds = brain_ref.conversation_tracker.default_max_rounds
    _send_desc = (
        f"向指定的 Agent 发送消息。注意：推荐优先使用 delegate_task 创建正式的任务委托。"
        f"如果已有活跃工单，消息应在工单框架内发送。"
        f"参数 target_agent_id: 目标 Agent 的 ID。"
        f"参数 message: 要发送的消息内容。"
        f"返回: 发送结果"
    )

    @tool(description=_send_desc)
    async def send_to_agent(target_agent_id: str, message: str) -> str:
        # ── 禁止向自己发消息 ──
        if target_agent_id == brain_ref.agent_id:
            return f"❌ 不能向自己发送消息。如果你想记录信息，请使用 update_memory 工具。"

        # ── 编排前探测拦截：无工单 + 无计划 → 禁止向 worker 发消息 ──
        dm: DelegationManager = brain_ref.delegation_manager
        ticket = dm.get_active_ticket(brain_ref.agent_id, target_agent_id)
        if (
            ticket is None
            and target_agent_id != "super_user"
            and target_agent_id in (brain_ref.peer_agents or set())
            and not brain_ref.orchestration_manager.has_active_plans(brain_ref.agent_id)
        ):
            return (
                "❌ **禁止在创建编排计划前向 worker 发送消息。**\n"
                f"你尚未创建任何编排计划，也没有与 {target_agent_id} 的活跃工单。\n"
                "正确流程：\n"
                "1. 直接调用 `create_task_plan(description)` 分解任务\n"
                "2. 系统会根据 suggested_role 自动匹配合适的 worker\n"
                "3. 用 `dispatch_subtasks(plan_id)` 派发任务\n\n"
                "**不需要事先询问 worker 的能力**——系统已自动匹配。"
            )

        # ---- [停止交流] 映射到 cancel_ticket ----
        if message and '[停止交流]' in message:
            ticket = dm.get_active_ticket(brain_ref.agent_id, target_agent_id)
            if ticket:
                try:
                    dm.cancel_ticket(ticket.ticket_id, "智能体发出停止交流标记", brain_ref.agent_id)
                    await brain_ref.comm.send_to_agent("super_user", {
                        "text": f"🔒 [TDP] {brain_ref.agent_id} 取消了与 {target_agent_id} 的工单 {ticket.ticket_id}"
                    })
                except TicketError:
                    pass
            brain_ref.conversation_tracker.reset(brain_ref.agent_id, target_agent_id)
            return f'已和 {target_agent_id} 停止交流'

        # ---- TDP 工单门控 ----
        ticket = dm.get_active_ticket(brain_ref.agent_id, target_agent_id)
        if ticket and ticket.is_terminal:
            return (
                f"❌ 工单 {ticket.ticket_id} 已终止（{ticket.state.value}），"
                f"请使用 delegate_task 创建新的委托。"
            )

        # ---- P0: 轮次门控（TDP 优先，ConversationTracker 兜底）----
        if ticket and ticket.is_active:
            level, warning = dm.record_round(ticket.ticket_id)
            if level == "capped":
                try:
                    await brain_ref.comm.send_to_agent("super_user", {
                        "text": (
                            f"🔒 [TDP 轮次耗尽] {brain_ref.agent_id} 与 {target_agent_id} 的工单 "
                            f"{ticket.ticket_id}（{ticket.description[:50]}...）已达 {ticket.max_rounds} 轮上限。"
                        )
                    })
                except Exception:
                    pass
                return (
                    f"⚠️ 工单 {ticket.ticket_id} 轮次已耗尽（{ticket.max_rounds}轮）。"
                    f"如需继续，请创建新工单或等待用户介入。"
                )
        else:
            # 无活跃工单：走旧 ConversationTracker 兜底
            can_send, reject_reason = brain_ref.conversation_tracker.can_send(brain_ref.agent_id, target_agent_id)
            if not can_send:
                logger.warning(f"send_to_agent 被拦截: {brain_ref.agent_id} -> {target_agent_id}: {reject_reason}")
                try:
                    await brain_ref.comm.send_to_agent("super_user", {
                        "text": f"🔒 [对话限制] {brain_ref.agent_id} 尝试向 {target_agent_id} 发送消息被拦截：{reject_reason}"
                    })
                except Exception:
                    pass
                return f"❌ 发送失败：{reject_reason}"

        online = brain_ref.online_agents
        if online is not None and target_agent_id not in online:
            logger.warning(f"向离线 Agent 发送消息: {target_agent_id}")

        # 发送时携带 ticket_id（如有）
        payload = {"text": message}
        if ticket and ticket.is_active:
            payload["ticket_id"] = ticket.ticket_id
        await brain_ref.comm.send_to_agent(target_agent_id, payload)

        # ---- 记录轮次（无工单时走旧 tracker）----
        if not ticket or not ticket.is_active:
            brain_ref.conversation_tracker.record_send(brain_ref.agent_id, target_agent_id)

        hint = "" if (online is None or target_agent_id in online) else f"（注意：{target_agent_id} 当前可能离线，对方可能无法收到）"
        if hint:
            return f'消息已发送，但{hint}'

        result = f'消息已经发送给了 Agent : {target_agent_id}，请等待对方回复。'
        return result

    return send_to_agent


def create_room_tools(brain):
    """创建群组相关工具 [join_room, leave_room, send_group_message]"""
    brain_ref = brain

    @tool
    async def join_room(room_id: str) -> str:
        """加入一个已有群组。"""
        await brain_ref.comm.send({"type": "join_room", "room_id": room_id, "agent_id": brain_ref.agent_id})
        return f"已请求加入群组 {room_id}"

    @tool
    async def leave_room(room_id: str) -> str:
        """离开群组。"""
        await brain_ref.comm.send({"type": "leave_room", "room_id": room_id, "agent_id": brain_ref.agent_id})
        return f"已离开群组 {room_id}"

    @tool
    async def send_group_message(room_id: str, message: str) -> str:
        """向群组发送消息。"""
        await brain_ref.comm.send({
            "type": "group_message",
            "room_id": room_id,
            "from": brain_ref.agent_id,
            "payload": {"text": message}
        })
        return f"消息已发送到群组 {room_id}"

    return [join_room, leave_room, send_group_message]


def create_log_memory_tool(brain):
    """任务步骤记录与反思工具"""
    brain_ref = brain

    @tool
    async def log_memory(description: str, result: str, task_complete: bool = False,
                         config: RunnableConfig = None) -> str:
        """
        记录当前任务的执行步骤或最终总结。
        当 task_complete=True 时，将整个任务提交给反思模块进行离线分析。

        :param description: 步骤描述或任务总结
        :param result: 执行结果
        :param task_complete: 是否完成任务
        :return: 提示信息
        """
        thread_id = config.get("configurable", {}).get("thread_id") if config else None
        if not thread_id:
            return "错误：无法获取当前对话 ID。"

        brain_ref.task_buffer.add_step(thread_id, description, result)

        if task_complete:
            final_result = "success" if "成功" in result else "failure"
            task_data = brain_ref.task_buffer.finish_task(thread_id, final_result)
            if task_data:
                brain_ref.schedule_background_task(brain_ref._process_reflection(task_data))
            return f"步骤已记录，任务结束，将进行经验反思。"
        else:
            return "步骤已记录。"

    return log_memory


# ══════════════════════════════════════════════════════════════════════
# Task Delegation Protocol (TDP) 工具
# ══════════════════════════════════════════════════════════════════════

def create_delegation_tools(brain):
    """创建 TDP 任务委托协议工具集（6个工具）。
    替代自由文本 send_to_agent，为智能体间协作提供结构化工单生命周期管理。
    """
    brain_ref = brain

    @tool
    async def delegate_task(
        agent: str,
        description: str,
        expected_output: str,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> str:
        """向指定智能体发起正式的任务委托（创建工单）。
        这是推荐的多智能体协作方式，不要使用 send_to_agent 私聊。
        注意：只能委托给其他智能体，不能委托给 super_user（用户）或 reminder_bot（系统机器人）。

        :param agent: 目标智能体的 ID（不能是 super_user）
        :param description: 任务描述，清晰说明需要做什么
        :param expected_output: 期望产出，说明"完成"的标准是什么
        :param max_rounds: 轮次预算（默认8），用于限制本工单的最大通信轮次
        :return: 委托结果
        """
        # 硬拦截：不能向用户或系统机器人发布工单
        if agent in ("super_user", "reminder_bot"):
            return f"❌ 不能向 '{agent}' 发起任务委托。'super_user' 是人类用户，'reminder_bot' 是系统机器人，仅智能体之间可以互相委派任务。请使用 list_online_agents 查看可委派的智能体。"

        dm: DelegationManager = brain_ref.delegation_manager

        online = brain_ref.online_agents
        if online is not None and agent not in online:
            logger.warning(f"向离线 Agent 发起委托: {agent}")

        try:
            ticket = dm.create_ticket(
                issuer=brain_ref.agent_id,
                assignee=agent,
                description=description,
                expected_output=expected_output,
                max_rounds=max_rounds,
            )
        except TicketError as e:
            return f"❌ 创建委托失败：{e}"

        # 发送 TDP 通知给 assignee
        await brain_ref.comm.send_to_agent(agent, {
            "text": (
                f"[任务委托] {brain_ref.agent_id} 向你发起了一个任务委托：\n"
                f"工单ID: {ticket.ticket_id}\n"
                f"任务描述: {description}\n"
                f"期望产出: {expected_output}\n"
                f"轮次预算: {max_rounds} 轮\n\n"
                f"请调用 accept_task('{ticket.ticket_id}') 接受，"
                f"或 decline_task('{ticket.ticket_id}', '原因') 拒绝。"
            ),
            "_tdp": "delegation",
            "ticket_id": ticket.ticket_id,
        })

        # 通知 super_user
        try:
            await brain_ref.comm.send_to_agent("super_user", {
                "text": (
                    f"📋 [TDP] {brain_ref.agent_id} 向 {agent} 发起任务委托 "
                    f"#{ticket.ticket_id}：{description[:80]}"
                )
            })
        except Exception:
            pass

        return (
            f"✅ 委托已创建。\n"
            f"工单ID: {ticket.ticket_id}\n"
            f"状态: {ticket.state.value}\n"
            f"轮次预算: {max_rounds} 轮\n"
            f"正在等待 {agent} 接受或拒绝。"
        )

    @tool
    async def accept_task(ticket_id: str) -> str:
        """接受一个待处理的任务委托。只有工单的 assignee（被委托方）可以调用。

        :param ticket_id: 工单 ID
        :return: 接受结果
        """
        dm: DelegationManager = brain_ref.delegation_manager
        ticket = dm.get_ticket(ticket_id)
        if not ticket:
            return f"❌ 工单 {ticket_id} 不存在。"

        if ticket.assignee != brain_ref.agent_id:
            return f"❌ 只有被委托方（{ticket.assignee}）可以接受此工单。"

        try:
            dm.transition(ticket_id, TicketState.ACCEPTED)
        except TicketError as e:
            return f"❌ 无法接受工单：{e}"

        # 通知 issuer（携带 _orchestration 以便编排回调处理）
        acceptance_payload = {
            "text": f"[TDP] {brain_ref.agent_id} 已接受工单 {ticket_id}: {ticket.description[:80]}",
            "_tdp": "acceptance",
            "ticket_id": ticket_id,
        }
        if getattr(ticket, 'orchestration_plan_id', None):
            acceptance_payload["_orchestration"] = ticket.orchestration_plan_id
        await brain_ref.comm.send_to_agent(ticket.issuer, acceptance_payload)

        try:
            await brain_ref.comm.send_to_agent("super_user", {
                "text": f"✅ [TDP] {brain_ref.agent_id} 接受了 {ticket.issuer} 的工单 #{ticket_id}"
            })
        except Exception:
            pass

        return (
            f"✅ 已接受工单 {ticket_id}。\n"
            f"任务: {ticket.description}\n"
            f"期望产出: {ticket.expected_output}\n"
            f"轮次预算: {ticket.max_rounds} 轮（剩余 {ticket.remaining_rounds} 轮）\n"
            f"完成后请使用 deliver_result('{ticket_id}', '总结') 交付结果。"
        )

    @tool
    async def decline_task(ticket_id: str, reason: str) -> str:
        """拒绝一个待处理的任务委托。只有工单的 assignee（被委托方）可以调用。

        :param ticket_id: 工单 ID
        :param reason: 拒绝原因
        :return: 拒绝结果
        """
        dm: DelegationManager = brain_ref.delegation_manager
        ticket = dm.get_ticket(ticket_id)
        if not ticket:
            return f"❌ 工单 {ticket_id} 不存在。"

        if ticket.assignee != brain_ref.agent_id:
            return f"❌ 只有被委托方（{ticket.assignee}）可以拒绝此工单。"

        try:
            dm.cancel_ticket(ticket_id, reason, brain_ref.agent_id)
            # 覆盖状态为 DECLINED
            ticket.state = TicketState.DECLINED
        except TicketError as e:
            return f"❌ 无法拒绝工单：{e}"

        decline_payload = {
            "text": f"[TDP] {brain_ref.agent_id} 拒绝了工单 {ticket_id}。原因: {reason}",
            "_tdp": "decline",
            "ticket_id": ticket_id,
        }
        if getattr(ticket, 'orchestration_plan_id', None):
            decline_payload["_orchestration"] = ticket.orchestration_plan_id
        await brain_ref.comm.send_to_agent(ticket.issuer, decline_payload)

        try:
            await brain_ref.comm.send_to_agent("super_user", {
                "text": f"❌ [TDP] {brain_ref.agent_id} 拒绝了 {ticket.issuer} 的工单 #{ticket_id}: {reason}"
            })
        except Exception:
            pass

        return f"已拒绝工单 {ticket_id}。"

    @tool
    async def deliver_result(ticket_id: str, summary: str) -> str:
        """交付任务结果并关闭工单。任务完成时调用此工具。

        :param ticket_id: 工单 ID
        :param summary: 结果总结，简要描述完成的工作和产出
        :return: 交付结果
        """
        dm: DelegationManager = brain_ref.delegation_manager
        ticket = dm.get_ticket(ticket_id)
        if not ticket:
            return f"❌ 工单 {ticket_id} 不存在。"

        if ticket.assignee != brain_ref.agent_id:
            # issuer 也可能想"交付"——但语义上只有 assignee 能交付
            return f"❌ 只有被委托方（{ticket.assignee}）可以交付结果。你是委托方，如需关闭请使用 cancel_task。"

        if ticket.is_terminal:
            return f"❌ 工单 {ticket_id} 已处于终态（{ticket.state.value}），无法交付。"

        try:
            dm.deliver_result(ticket_id, summary)
        except TicketError as e:
            return f"❌ 交付失败：{e}"

        # 通知 issuer（携带 _orchestration 以便编排回调处理）
        delivery_payload = {
            "text": (
                f"[TDP 交付] {brain_ref.agent_id} 已完成工单 {ticket_id}。\n"
                f"结果: {summary}"
            ),
            "_tdp": "delivery",
            "ticket_id": ticket_id,
        }
        if getattr(ticket, 'orchestration_plan_id', None):
            delivery_payload["_orchestration"] = ticket.orchestration_plan_id
        await brain_ref.comm.send_to_agent(ticket.issuer, delivery_payload)

        try:
            await brain_ref.comm.send_to_agent("super_user", {
                "text": (
                    f"📦 [TDP 交付] {brain_ref.agent_id} 完成了 {ticket.issuer} 的工单 #{ticket_id}:\n"
                    f"{summary[:200]}"
                )
            })
        except Exception:
            pass

        # 检查是否还有其他待处理的工单（提醒 agent 逐一处理）
        pending_tickets = [
            t for t in dm.get_active_tickets_for_agent(brain_ref.agent_id)
            if t.state == TicketState.PENDING and t.assignee == brain_ref.agent_id
        ]
        if pending_tickets:
            lines = [
                f"\n📋 **你还有 {len(pending_tickets)} 个待处理的工单，"
                f"请立即 accept 并完成，不要遗漏：**"
            ]
            for pt in pending_tickets[:5]:
                lines.append(f"- 工单 `{pt.ticket_id}`: {pt.description[:120]}")
            pending_reminder = "\n".join(lines)
            return (
                f"✅ 工单 {ticket_id} 已交付并关闭。\n"
                f"耗时轮次: {ticket.round_count}/{ticket.max_rounds}\n"
                f"交付总结: {summary}\n"
                f"{pending_reminder}"
            )
        else:
            return (
                f"✅ 工单 {ticket_id} 已交付并关闭。\n"
                f"耗时轮次: {ticket.round_count}/{ticket.max_rounds}\n"
                f"交付总结: {summary}\n\n"
                f"⚠️ **工单已关闭，不要发送任何后续消息。你的任务已完全结束。** "
                f"不要向委托方发总结或补充说明，系统会自动通知对方。"
            )

    @tool
    async def request_clarification(ticket_id: str, question: str) -> str:
        """在工单内向委托方请求澄清。最多2轮澄清，计入总轮次预算。

        :param ticket_id: 工单 ID
        :param question: 需要澄清的问题
        :return: 发送结果
        """
        dm: DelegationManager = brain_ref.delegation_manager
        ticket = dm.get_ticket(ticket_id)
        if not ticket:
            return f"❌ 工单 {ticket_id} 不存在。"

        if ticket.is_terminal:
            return f"❌ 工单 {ticket_id} 已终止。"

        if not dm.record_clarification(ticket_id):
            return (
                f"❌ 澄清次数已达上限（{ticket.max_clarifications} 轮）。"
                f"请基于现有信息继续工作或交付结果。"
            )

        # 确定对方是谁
        other = ticket.issuer if brain_ref.agent_id == ticket.assignee else ticket.assignee
        # 如果当前不在 NEGOTIATING 状态，尝试转换
        if ticket.state not in (TicketState.NEGOTIATING,):
            try:
                dm.transition(ticket_id, TicketState.NEGOTIATING)
            except TicketError:
                pass  # 某些状态下可能不允许转换，继续发送消息

        await brain_ref.comm.send_to_agent(other, {
            "text": f"[TDP 澄清 #{ticket.clarification_count}/{ticket.max_clarifications}] {question}",
            "_tdp": "clarification",
            "ticket_id": ticket_id,
        })

        return (
            f"已发送澄清请求（{ticket.clarification_count}/{ticket.max_clarifications} 轮）。\n"
            f"剩余工作轮次: {ticket.remaining_rounds}"
        )

    @tool
    async def cancel_task(ticket_id: str, reason: str) -> str:
        """取消一个活跃的任务委托。委托方或被委托方均可调用。

        :param ticket_id: 工单 ID
        :param reason: 取消原因
        :return: 取消结果
        """
        dm: DelegationManager = brain_ref.delegation_manager
        ticket = dm.get_ticket(ticket_id)
        if not ticket:
            return f"❌ 工单 {ticket_id} 不存在。"

        if ticket.is_terminal:
            return f"工单 {ticket_id} 已处于终态（{ticket.state.value}），无需取消。"

        # 确定对方
        other = ticket.issuer if brain_ref.agent_id == ticket.assignee else ticket.assignee

        try:
            dm.cancel_ticket(ticket_id, reason, brain_ref.agent_id)
        except TicketError as e:
            return f"❌ 取消失败：{e}"

        # 通知对方
        try:
            await brain_ref.comm.send_to_agent(other, {
                "text": f"[TDP] {brain_ref.agent_id} 取消了工单 {ticket_id}。原因: {reason}",
                "_tdp": "cancel",
                "ticket_id": ticket_id,
            })
        except Exception:
            pass

        try:
            await brain_ref.comm.send_to_agent("super_user", {
                "text": f"🚫 [TDP] {brain_ref.agent_id} 取消了工单 #{ticket_id}: {reason}"
            })
        except Exception:
            pass

        return f"已取消工单 {ticket_id}。"

    @tool
    async def report_progress(ticket_id: str, status: str) -> str:
        """汇报工单进度（可选，不计入轮次）。用于在长时间任务中告知对方进展。

        :param ticket_id: 工单 ID
        :param status: 当前进度描述
        :return: 汇报结果
        """
        dm: DelegationManager = brain_ref.delegation_manager
        ticket = dm.get_ticket(ticket_id)
        if not ticket:
            return f"❌ 工单 {ticket_id} 不存在。"

        if ticket.is_terminal:
            return f"❌ 工单 {ticket_id} 已终止。"

        ticket.progress_updates.append({
            "status": status,
            "time": __import__("time").time(),
        })
        ticket.last_activity = __import__("time").time()

        other = ticket.issuer if brain_ref.agent_id == ticket.assignee else ticket.assignee
        await brain_ref.comm.send_to_agent(other, {
            "text": f"[TDP 进度] {brain_ref.agent_id}: {status}",
            "_tdp": "progress",
            "ticket_id": ticket_id,
        })

        return f"进度已汇报。"

    return [
        delegate_task,
        accept_task,
        decline_task,
        deliver_result,
        request_clarification,
        cancel_task,
        report_progress,
    ]


def create_orchestration_tools(brain):
    """创建编排工具：任务分解、并行派发、进度跟踪、失败重分配。

    在 TDP 之上提供多智能体协同完成复杂任务的能力。
    """
    brain_ref = brain
    om: OrchestrationManager = brain_ref.orchestration_manager
    dm: DelegationManager = brain_ref.delegation_manager
    comm = brain_ref.comm

    @tool
    async def create_task_plan(description: str) -> str:
        """分析复杂任务并创建结构化执行计划，含子任务、依赖关系和建议智能体角色。

        适用场景：需要多个智能体协作完成的复杂任务。LLM 会自动分解为 2-5 个子任务。
        生成的计划需要用户审阅确认后，再使用 dispatch_subtasks 派发。

        Args:
            description: 任务的详细描述
        """
        peer_agents = brain_ref.peer_agents
        agents_info = "\n".join(f"- {a}" for a in sorted(peer_agents)) if peer_agents else ""
        online_count = len(peer_agents)

        prompt = build_task_decomposition_prompt(description, agents_info)
        try:
            response = await call_big_model_chat(
                prompt, temperature=0.3, is_json=True,
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return "任务分解失败：LLM 返回空响应"

            plan_data = _json.loads(content)
            subtasks = plan_data.get("subtasks", [])

            # ── 程序化校验并行度（代码层兜底，不依赖提示词）──
            if online_count >= 2 and len(subtasks) >= 2:
                roles = [s.get("suggested_role", "") for s in subtasks]
                unique_roles = set(r for r in roles if r)

                # 校验1：全部串行（每个子任务都依赖前一个）
                all_serial = all(
                    subtasks[i].get("depends_on", []) == [i]
                    for i in range(1, len(subtasks))
                )
                # 校验2：角色完全相同（无法分散到不同 agent）
                all_same_role = len(unique_roles) <= 1 and len(subtasks) >= 2

                if all_serial or all_same_role:
                    reason = (
                        "全部子任务串行依赖" if all_serial else "所有子任务角色相同"
                    )
                    logger.warning(
                        f"计划并行度校验失败（{reason}），自动重试。"
                        f"在线智能体: {online_count}, 子任务: {len(subtasks)}"
                    )
                    # 带修正指令重试一次
                    correction = (
                        f"\n\n【重要修正指令】上一次生成的计划存在严重问题：{reason}。"
                        f"当前有 {online_count} 个在线智能体，必须充分利用：\n"
                    )
                    if all_serial:
                        correction += (
                            "- **必须创建至少 2 个无依赖（depends_on=[]）的子任务**，使其可并行派发。\n"
                            "- 仅当后续任务确实需要前置输出时才建立依赖。\n"
                        )
                    if all_same_role:
                        agents_str = ", ".join(sorted(peer_agents))
                        correction += (
                            f"- 在线智能体: {agents_str}\n"
                            "- **必须为不同子任务分配不同的 suggested_role**，"
                            f"确保工作分散到所有可用智能体。\n"
                        )
                    correction += "- 重新生成完整的 JSON 计划。"
                    retry_prompt = prompt + correction
                    response = await call_big_model_chat(
                        retry_prompt, temperature=0.4, is_json=True,
                    )
                    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        try:
                            plan_data = _json.loads(content)
                            subtasks = plan_data.get("subtasks", [])
                            logger.info("并行度重试成功")
                        except _json.JSONDecodeError:
                            logger.warning("并行度重试返回格式异常，使用原始计划")
                            # 回退到原始计划
                            subtasks = plan_data.get("subtasks", [])

            if not subtasks:
                return "任务分解失败：未生成子任务"

            plan = await om.create_plan(description, brain_ref.agent_id)
            # 记录创建时的 thread_id，确保合成报告路由到正确会话
            plan.thread_id = brain_ref.thread_id or ""
            # 存储项目概览
            plan.project_overview = plan_data.get("project_overview", "")
            await om.set_subtasks(plan.plan_id, subtasks)

            result = om.get_progress_summary(plan.plan_id)
            result += f"\n\n计划 ID: `{plan.plan_id}`"
            result += f"\n可用智能体: {', '.join(sorted(peer_agents)) if peer_agents else '（无在线智能体，启动后再派发）'}"
            result += "\n请审阅后使用 `dispatch_subtasks` 派发。"
            return result
        except _json.JSONDecodeError as e:
            logger.error(f"Failed to parse task decomposition JSON: {e}")
            return f"任务分解失败：LLM 返回格式异常。请重试或简化任务描述。"
        except Exception as e:
            logger.error(f"Failed to create task plan: {e}", exc_info=True)
            return f"创建任务计划失败: {e}"

    @tool
    async def dispatch_subtasks(plan_id: str) -> str:
        """派发编排计划中所有就绪的子任务给合适的在线智能体。

        子任务的依赖满足后（前序任务已完成）即可派发。
        每个子任务会创建一个 TDP 工单并通知目标智能体。

        Args:
            plan_id: create_task_plan 返回的计划 ID
        """
        plan = om.get_plan(plan_id)
        if not plan:
            return f"未找到计划 {plan_id}。请先使用 create_task_plan 创建计划。"

        ready = plan.get_ready_subtasks()
        if not ready:
            # 检查是否有未完成的依赖
            pending = [s for s in plan.subtasks if s.status == SubTaskStatus.PENDING]
            if pending:
                blocked = [f"{s.id} (依赖: {', '.join(s.depends_on)})" for s in pending]
                return f"没有可派发的子任务。以下子任务被依赖阻塞:\n" + "\n".join(blocked)
            dispatched = [s for s in plan.subtasks if s.status == SubTaskStatus.DISPATCHED]
            in_progress = [s for s in plan.subtasks if s.status == SubTaskStatus.IN_PROGRESS]
            return f"没有待派发的子任务。已派发: {len(dispatched)}, 执行中: {len(in_progress)}"

        online = list(brain_ref.peer_agents)
        if not online:
            return "当前没有在线智能体可供派发任务。请等待智能体上线后重试。"

        results = []
        # 使用 brain 级别的持久缓存，确保跨批次（DAG 阶段）不会重复分配同一智能体
        used_agents = brain_ref._plan_used_agents.setdefault(plan_id, set())
        for st in ready:
            try:
                msg, _ = await brain_ref._dispatch_single_subtask(
                    plan_id, st, online, allow_parallel=True,
                    used_agents=used_agents)
                results.append(msg)
            except Exception as e:
                results.append(f"❌ {st.id} 派发失败: {e}")
                logger.error(f"Failed to dispatch subtask {st.id}: {e}")

        if plan.state == PlanState.READY:
            plan.state = PlanState.EXECUTING

        return (
            f"计划 {plan_id} 已派发 {len(results)} 个子任务:\n"
            + "\n".join(results)
            + f"\n\n**重要：你无需轮询进度。** 每个子任务完成后系统会自动通知你，"
            f"届时如果你需要回复用户，直接汇总结果即可。"
            f"仅当用户主动询问进度时才调用 `check_plan_progress('{plan_id}')`。"
        )

    @tool
    async def check_plan_progress(plan_id: str) -> str:
        """查看编排计划的整体进度。

        此工具仅应在以下情况使用：
        1. 用户明确询问任务进度
        2. 你收到了子任务完成的通知，需要确认最终状态
        **严禁反复轮询！** 子任务完成后系统会自动通知你，无需主动查询。

        Args:
            plan_id: 要查询的计划 ID
        """
        return om.get_progress_summary(plan_id)

    @tool
    async def reassign_subtask(plan_id: str, subtask_id: str, new_agent: str) -> str:
        """将失败的子任务重新分配给另一个智能体。

        Args:
            plan_id: 计划 ID
            subtask_id: 要重新分配的子任务 ID（如 st_1）
            new_agent: 新分配的智能体 ID
        """
        plan = om.get_plan(plan_id)
        if not plan:
            return f"未找到计划 {plan_id}"

        st = next((s for s in plan.subtasks if s.id == subtask_id), None)
        if not st:
            return f"未找到子任务 {subtask_id}（可用: {', '.join(s.id for s in plan.subtasks)}）"

        if st.status not in (SubTaskStatus.FAILED, SubTaskStatus.PENDING):
            return f"子任务 {subtask_id} 当前状态为 {st.status.value}，只能重新分配失败或待派发的子任务"

        # 重置状态
        old_ticket = st.ticket_id
        if old_ticket and old_ticket in om._ticket_index:
            del om._ticket_index[old_ticket]
        st.status = SubTaskStatus.PENDING
        st.assigned_to = None
        st.ticket_id = None
        st.result = None

        online = list(brain_ref.peer_agents)
        _, ticket_id = await brain_ref._dispatch_single_subtask(
            plan_id, st, online,
            allow_parallel=False, reassign=True,
            assigned_to=new_agent,
        )

        return f"已重新派发 {subtask_id} → **{new_agent}** (工单 `{ticket_id}`)"

    return [
        create_task_plan,
        dispatch_subtasks,
        check_plan_progress,
        reassign_subtask,
    ]


def create_board_tools(brain):
    """创建 Loop Engineering 全局任务看板工具。

    提供 read_board（查看看板快照）和 update_task_status（更新子任务状态并触发事件）。
    """
    brain_ref = brain
    om: OrchestrationManager = brain_ref.orchestration_manager

    @tool
    def read_board(plan_id: str) -> str:
        """获取当前计划的完整看板快照（Markdown 格式）。

        所有 Agent 都可以通过此工具查看任务团队的全局状态，
        包括每个子任务的状态、执行者、审核者和重试次数。

        Args:
            plan_id: 计划 ID
        """
        plan = om.get_plan(plan_id)
        if not plan:
            return "❌ 计划不存在"

        lines = [
            f"# 📊 任务看板: {plan_id}",
            f"**项目**: {plan.project_overview or plan.description[:80]}",
            f"**状态**: {plan.state.value}",
            f"**进度**: {plan.completed_count}/{plan.total_count} 已完成",
            f"**降级模式**: {['正常', 'LLM降级', '网络降级', '安全模式'][plan.degradation_level]}",
            "",
            "## 子任务列表",
            "| ID | 描述 | 状态 | 执行者 | 审核者 | 重试 |",
            "|----|------|------|--------|--------|------|",
        ]

        for st in plan.subtasks:
            desc = st.description[:30] + "..." if len(st.description) > 30 else st.description
            reviewer = st.reviewer_agent or "无"
            lines.append(
                f"| {st.id} | {desc} | {st.status.value} | "
                f"{st.assigned_to or '未分配'} | {reviewer} | "
                f"{st.retry_count}/{st.max_retries} |"
            )

        if plan.critical_decisions:
            lines.append("")
            lines.append("## 📝 关键决策")
            for decision in plan.critical_decisions[-5:]:
                lines.append(f"- {decision}")

        if plan.escalation_log:
            lines.append("")
            lines.append("## 🚨 升级记录")
            for entry in plan.escalation_log[-3:]:
                lines.append(f"- {entry.get('time', '?')}: {entry.get('reason', '?')[:80]}")

        return "\n".join(lines)

    @tool
    async def update_task_status(plan_id: str, subtask_id: str, status: str,
                                  message: str = "") -> str:
        """更新子任务状态，触发级联事件。

        可用状态: pending, in_progress, reviewing, approved, blocked, completed, failed, skipped
        特殊行为:
        - BLOCKED: 自动触发升级研讨
        - APPROVED: 自动触发后续派发

        Args:
            plan_id: 计划 ID
            subtask_id: 子任务 ID
            status: 新状态
            message: 附加说明（阻塞原因或审核意见）
        """
        plan = om.get_plan(plan_id)
        if not plan:
            return f"❌ 计划 {plan_id} 不存在"

        st = plan.get_subtask(subtask_id)
        if not st:
            return f"❌ 子任务 {subtask_id} 不存在"

        old_status = st.status
        try:
            new_status = SubTaskStatus(status)
        except ValueError:
            valid = [s.value for s in SubTaskStatus]
            return f"❌ 无效状态 '{status}'。可用: {', '.join(valid)}"

        st.status = new_status

        # BLOCKED 触发升级
        if new_status == SubTaskStatus.BLOCKED:
            st.blocked_reason = message
            plan.escalation_log.append({
                "time": datetime.now().isoformat(),
                "subtask_id": subtask_id,
                "reason": message,
                "resolution": "待处理",
            })
            asyncio.create_task(brain_ref._handle_escalation(plan_id, subtask_id))

        # APPROVED 触发后续派发
        if new_status == SubTaskStatus.APPROVED:
            plan.completed_count += 1
            if plan.is_complete():
                asyncio.create_task(brain_ref._synthesize_plan_results(plan_id))
            else:
                ready = plan.get_ready_subtasks()
                if ready:
                    asyncio.create_task(
                        brain_ref._dispatch_ready_subtasks(plan, ready)
                    )

        await om._save_plan(plan)
        return f"✅ 子任务 {subtask_id} 状态已更新: {old_status.value} → {new_status.value}"

    return [read_board, update_task_status]
