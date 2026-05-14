"""
任务缓冲区：按 thread_id 存储当前任务的步骤列表和描述
"""

from typing import Dict, List, Any
from datetime import datetime

class TaskBuffer:
    def __init__(self):
        self.buffers: Dict[str, Dict] = {}  # thread_id -> {steps, task_description, start_time}

    def start_task(self, thread_id: str, task_description: str):
        self.buffers[thread_id] = {
            "steps": [],
            "task_description": task_description,
            "start_time": datetime.now()
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

    def finish_task(self, thread_id: str, final_result: str, user_feedback: str = "") -> Dict:
        """完成并返回任务数据，同时清空缓冲区"""
        if thread_id not in self.buffers:
            return {}
        task_data = {
            "task_id": thread_id,
            "task_description": self.buffers[thread_id]["task_description"],
            "steps": self.buffers[thread_id]["steps"],
            "final_result": final_result,
            "user_feedback": user_feedback,
            "duration_seconds": (datetime.now() - self.buffers[thread_id]["start_time"]).total_seconds()
        }
        del self.buffers[thread_id]
        return task_data

    def get_current_task(self, thread_id: str):
        return self.buffers.get(thread_id)