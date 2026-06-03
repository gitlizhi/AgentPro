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

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from agent.delegation import (
    DelegationManager, TaskTicket, TicketState, TicketError,
    DEFAULT_MAX_ROUNDS,
)

logger = logging.getLogger(__name__)


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
        查询当前在线的 Agent 列表。
        用于在和其他 Agent 协作前判断对方是否在线，避免向离线的 Agent 发送消息。
        :return: JSON 格式的在线 Agent 列表
        """
        agents = brain_ref.online_agents
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
        dm: DelegationManager = brain_ref.delegation_manager

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
        """向指定 Agent 发起正式的任务委托（创建工单）。
        这是推荐的多智能体协作方式，不要使用 send_to_agent 私聊。

        :param agent: 目标 Agent 的 ID
        :param description: 任务描述，清晰说明需要做什么
        :param expected_output: 期望产出，说明"完成"的标准是什么
        :param max_rounds: 轮次预算（默认8），用于限制本工单的最大通信轮次
        :return: 委托结果
        """
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

        # 通知 issuer
        await brain_ref.comm.send_to_agent(ticket.issuer, {
            "text": f"[TDP] {brain_ref.agent_id} 已接受工单 {ticket_id}: {ticket.description[:80]}",
            "_tdp": "acceptance",
            "ticket_id": ticket_id,
        })

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

        await brain_ref.comm.send_to_agent(ticket.issuer, {
            "text": f"[TDP] {brain_ref.agent_id} 拒绝了工单 {ticket_id}。原因: {reason}",
            "_tdp": "decline",
            "ticket_id": ticket_id,
        })

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

        # 通知 issuer
        await brain_ref.comm.send_to_agent(ticket.issuer, {
            "text": (
                f"[TDP 交付] {brain_ref.agent_id} 已完成工单 {ticket_id}。\n"
                f"结果: {summary}"
            ),
            "_tdp": "delivery",
            "ticket_id": ticket_id,
        })

        try:
            await brain_ref.comm.send_to_agent("super_user", {
                "text": (
                    f"📦 [TDP 交付] {brain_ref.agent_id} 完成了 {ticket.issuer} 的工单 #{ticket_id}:\n"
                    f"{summary[:200]}"
                )
            })
        except Exception:
            pass

        return (
            f"✅ 工单 {ticket_id} 已交付并关闭。\n"
            f"耗时轮次: {ticket.round_count}/{ticket.max_rounds}\n"
            f"交付总结: {summary}"
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
