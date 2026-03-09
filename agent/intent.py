"""
意图识别枚举
"""
from enum import Enum
from typing import Dict, Any

class IntentType(Enum):
    SET_REMINDER = 1

class IntentType(str, Enum):
    CHAT = "chat"
    SET_REMINDER = "set_reminder"
    QUERY_REMINDER = "query_reminder"
    # 后续可继续添加，如 "CANCEL_REMINDER", "WEATHER_QUERY" 等

# 意图描述，用于提示词
INTENT_DESCRIPTIONS: Dict[IntentType, str] = {
    IntentType.CHAT: "普通对话，不需要特殊处理，直接由AI助手回答。",
    IntentType.SET_REMINDER: "设置一个定时提醒。用户希望在未来某个时间收到提醒，需要提取提醒的时间和内容。",
    IntentType.QUERY_REMINDER: "查询当前有哪些未到期的提醒。用户想知道自己设置的提醒列表。",
}

# 意图所需的参数说明（可选，用于提示词）
# INTENT_PARAMS: Dict[IntentType, Dict[str, str]] = {
#     IntentType.SET_REMINDER: {
#         "time": "提醒时间，需要你将自然语言转为代码可解析的时间，格式如：2026-03-09 10:10:19",
#         "message": "提醒内容"
#     },
#     # 其他意图无参数
# }