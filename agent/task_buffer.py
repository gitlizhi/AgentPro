"""
任务缓冲区：按 thread_id 存储当前任务的步骤列表和描述
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

class TaskBuffer:
    def __init__(self):
        self.buffers: Dict[str, Dict] = {}  # thread_id -> 任务详情

    def start_task(self, thread_id: str, task_description: str):
        self.buffers[thread_id] = {
            "steps": [],
            "task_description": task_description,
            "start_time": datetime.now(),
            "last_active_time": time.time(),      # 新增：最后活动时间戳（秒）
            "status": "in_progress",             # 新增：状态 in_progress / completed / failed
        }

    def add_step(self, thread_id: str, step_description: str, result: str, tool_calls: List = None):
        if thread_id not in self.buffers:
            # 如果未显式开始，自动开始（使用空描述）
            self.start_task(thread_id, "")
        self.buffers[thread_id]["steps"].append({
            "step_description": step_description,
            "result": result,
            "tool_calls": tool_calls or [],
            "timestamp": datetime.now().isoformat()
        })
        # 更新最后活动时间
        self.buffers[thread_id]["last_active_time"] = time.time()

    def finish_task(self, thread_id: str, final_result: str, user_feedback: str = "") -> Dict:
        if thread_id not in self.buffers:
            return {}
        task = self.buffers[thread_id]
        task["status"] = "completed" if "成功" in final_result else "failed"
        task["last_active_time"] = time.time()
        task_data = {
            "task_id": thread_id,
            "task_description": task["task_description"],
            "steps": task["steps"],
            "final_result": final_result,
            "user_feedback": user_feedback,
            "duration_seconds": (datetime.now() - task["start_time"]).total_seconds()
        }
        del self.buffers[thread_id]
        return task_data

    def get_current_task(self, thread_id: str) -> Optional[Dict]:
        return self.buffers.get(thread_id)

    def has_active_task(self, thread_id: str, min_rounds: int = 8, max_idle_seconds: int = 300) -> bool:
        """
        判断指定 thread_id 是否有进行中的任务，且未超时空闲。
        - max_idle_seconds: 最大允许空闲秒数，超过则认为任务已“僵死”，不再阻止终止。
        """
        task = self.buffers.get(thread_id)
        if not task:
            return False
        if task.get("status") != "in_progress":
            return False
        # 检查是否还在初始保护期
        steps_count = len(task.get("steps", []))
        if steps_count <= min_rounds:
            # 前 min_rounds 步内，不终止
            return True
        last_active = task.get("last_active_time", 0)
        idle_seconds = time.time() - last_active
        if idle_seconds >= max_idle_seconds:
            return False
        return True