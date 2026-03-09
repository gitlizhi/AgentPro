# agent/tasks.py
import logging

logger = logging.getLogger(__name__)

_reminder_comm = None

def set_reminder_comm(comm):
    global _reminder_comm
    _reminder_comm = comm

async def send_reminder(user_id: str, message: str):
    if _reminder_comm is None:
        raise RuntimeError("Reminder comm not set")
    await _reminder_comm.send_to_agent(user_id, {"text": f"提醒：{message}"})