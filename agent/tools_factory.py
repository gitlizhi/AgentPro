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
    """向其他 Agent 发送消息工具"""
    brain_ref = brain
    max_rounds = brain_ref.conversation_tracker.default_max_rounds
    _send_desc = (
        f"向指定的 Agent 发送消息，不等待对方回复。"
        f"注意：你与每个 Agent 的对话有轮次上限（默认 {max_rounds} 轮），请高效沟通。"
        f"建议在发送前先调用 list_online_agents 检查目标 Agent 是否在线。"
        f"参数 target_agent_id: 目标 Agent 的 ID。"
        f"参数 message: 要发送的消息内容。"
        f"返回: 发送结果，包含轮次提醒"
    )

    @tool(description=_send_desc)
    async def send_to_agent(target_agent_id: str, message: str) -> str:
        if message and '[停止交流]' in message:
            brain_ref.conversation_tracker.reset(brain_ref.agent_id, target_agent_id)
            return f'已和 {target_agent_id} 停止交流'

        # ---- P0: 轮次门控 ----
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
        await brain_ref.comm.send_to_agent(target_agent_id, {"text": message})

        # ---- P0: 记录轮次 ----
        level, warning = brain_ref.conversation_tracker.record_send(brain_ref.agent_id, target_agent_id)

        hint = "" if (online is None or target_agent_id in online) else f"（注意：{target_agent_id} 当前可能离线，对方可能无法收到）"
        if hint:
            return f'消息已发送，但{hint}'

        # ---- P0: 达到上限时通知用户 ----
        if level == "capped":
            state = brain_ref.conversation_tracker.get_state(brain_ref.agent_id, target_agent_id)
            try:
                await brain_ref.comm.send_to_agent("super_user", {
                    "text": (
                        f"🔒 [对话达上限] {brain_ref.agent_id} 与 {target_agent_id} 的对话已达到 "
                        f"{state.max_rounds} 轮上限，已自动限制。如需继续，请发送消息指示。"
                    )
                })
            except Exception:
                pass
            return (
                f"⚠️ 消息已发送，但你与 {target_agent_id} 的对话已达 {state.max_rounds} 轮上限，"
                f"后续消息将被限制。请等待用户指示或使用其他方式协作。"
            )

        result = f'消息已经发送给了 Agent : {target_agent_id}，请等待对方回复。'
        if warning:
            result += f"\n{warning}"
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
