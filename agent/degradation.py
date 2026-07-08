"""
Loop Engineering 降级策略模块
定义降级级别和管理器，保证系统在任何异常下都有对应的降级方案。
"""
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    NORMAL = 0          # 全功能模式
    DEGRADED_LLM = 1    # LLM 不可用，使用规则/模板
    DEGRADED_NET = 2    # 网络不可用，离线模式
    DEGRADED_SAFE = 3   # 安全模式，仅读操作


class DegradationManager:
    """管理降级级别和降级事件日志。"""

    def __init__(self):
        self.level = DegradationLevel.NORMAL
        self.fallback_log: list[dict] = []
        self.enabled = True  # 降级策略开关

    def set_level(self, level: DegradationLevel, reason: str = ""):
        """设置降级级别并记录日志。"""
        old_level = self.level
        self.level = level
        entry = {
            "time": datetime.now().isoformat(),
            "from": old_level.name,
            "to": level.name,
            "reason": reason,
        }
        self.fallback_log.append(entry)
        logger.warning(f"降级级别变更: {old_level.name} → {level.name}，原因: {reason}")

    def is_normal(self) -> bool:
        return self.level == DegradationLevel.NORMAL

    def can_call_llm(self) -> bool:
        return self.level.value <= DegradationLevel.DEGRADED_LLM.value

    def can_call_network(self) -> bool:
        return self.level.value <= DegradationLevel.DEGRADED_NET.value

    def can_write(self) -> bool:
        return self.level != DegradationLevel.DEGRADED_SAFE

    def reset(self):
        """重置到正常模式。"""
        if self.level != DegradationLevel.NORMAL:
            logger.info(f"降级级别恢复: {self.level.name} → NORMAL")
            self.level = DegradationLevel.NORMAL

    def get_fallback_response(self, prompt_hint: str) -> dict:
        """基于规则生成兜底响应。"""
        if "分解" in prompt_hint or "decomposition" in prompt_hint:
            lines = [line.strip() for line in prompt_hint.split("\n")
                     if line.strip() and not line.startswith("{")]
            return {
                "subtasks": [
                    {"id": f"st_{i + 1}", "description": line, "depends_on": [],
                     "worker_prompt": f"执行: {line}", "reviewer_prompt": ""}
                    for i, line in enumerate(lines[:5])
                ],
                "project_overview": "",
            }
        elif "审核" in prompt_hint or "review" in prompt_hint:
            return {"passed": True, "feedback": "自动通过（LLM降级模式）", "score": 8}
        elif "升级" in prompt_hint or "escalation" in prompt_hint:
            return {"resolution": "retry", "reason": "LLM不可用，自动重试"}
        elif "澄清" in prompt_hint or "clarification" in prompt_hint:
            return {"clear": True, "questions": [], "assumption": ""}
        return {}
