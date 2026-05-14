"""
意图识别枚举
"""
from enum import Enum
from typing import Dict, Any

class IntentType(Enum):
    SET_REMINDER = 1

class IntentType(str, Enum):
    SET_REMINDER = "set_reminder"
    QUERY_REMINDER = "query_reminder"
    COMPLEX_TASKS = "complex_tasks"
    # 后续可继续添加，如 "CANCEL_REMINDER", "WEATHER_QUERY" 等

# 意图描述，用于提示词
INTENT_DESCRIPTIONS: Dict[IntentType, str] = {
    IntentType.SET_REMINDER: "设置一个定时提醒。用户希望在未来某个时间收到提醒，需要提取提醒的时间和内容。",
    IntentType.QUERY_REMINDER: "用户的意图是想知道当前有哪些待办事项或者设置的提醒列表内容。",
    IntentType.COMPLEX_TASKS: "用户的意图不是简单的聊天，而是要具体执行某项任务，执行复杂任务，或者需要复杂推理的事情",
}
