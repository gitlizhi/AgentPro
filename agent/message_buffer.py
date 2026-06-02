"""
Agent-to-agent message buffer with delayed batching.
Accumulates rapid-fire messages, waits for a quiet period, then flushes.
"""
import asyncio
import logging
from typing import Callable, Coroutine, Optional, Dict

logger = logging.getLogger(__name__)


class MessageBuffer:
    """
    Buffers inbound messages from other agents, merging them into a single
    batched input after a configurable quiet-period delay.

    Lifecycle:
      enqueue()  -- accumulate text, (re)start delay timer
      cancel()   -- tear down timer and processing for a user_id
      cancel_all() -- tear down all timers (for agent shutdown)
      is_processing() -- check if a user_id is currently in the LLM call

    When the delay expires the buffer calls `on_process(full_text, saved_ctx)`
    where `saved_ctx` is any group_context that was saved alongside the messages.
    """

    def __init__(self, delay_seconds: float = 5.0) -> None:
        self._delay = delay_seconds
        self._cache: Dict[str, str] = {}           # user_id -> accumulated text
        self._group_ctx: Dict[str, dict] = {}      # user_id -> saved group_context
        self._pending: Dict[str, asyncio.Task] = {} # user_id -> sleep-timer Task
        self._processing: set[str] = set()          # user_ids inside LLM call

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def enqueue(
        self,
        user_id: str,
        user_input: str,
        group_context: Optional[dict],
        on_process: Callable[[str, Optional[dict]], Coroutine],
    ) -> None:
        """
        Accumulate *user_input* for *user_id*.  If a timer is already
        sleeping it is cancelled and replaced.  If the user_id is
        currently inside processing, the text is queued silently and will
        trigger a new timer after the current call completes.

        *on_process* is called as:  await on_process(full_text, saved_ctx)
        """
        # Accumulate text
        if user_id not in self._cache:
            self._cache[user_id] = user_input
        else:
            self._cache[user_id] += "\n" + user_input

        # Save group context alongside the message (may be overwritten after process() returns)
        if group_context:
            self._group_ctx[user_id] = group_context

        # If already processing, don't interrupt — message is in cache,
        # will be picked up by auto-restart at the end of current call
        if user_id in self._processing:
            return

        # Cancel any existing sleep timer (safe, hasn't entered LLM call yet)
        if user_id in self._pending:
            self._pending[user_id].cancel()

        # Start a new delay timer
        if user_input:
            t = asyncio.create_task(self._delayed_process(user_id, on_process))
            self._pending[user_id] = t

    def cancel(self, user_id: str) -> None:
        """Cancel any pending timer and remove from processing set."""
        if user_id in self._pending:
            self._pending[user_id].cancel()
            self._pending.pop(user_id, None)
        self._processing.discard(user_id)
        self._cache.pop(user_id, None)
        self._group_ctx.pop(user_id, None)

    def cancel_all(self) -> None:
        """Cancel all pending timers. Used during agent shutdown."""
        for task in self._pending.values():
            task.cancel()
        self._pending.clear()
        self._processing.clear()

    def is_processing(self, user_id: str) -> bool:
        """Return True if *user_id* is currently inside the LLM-processing phase."""
        return user_id in self._processing

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------
    async def _delayed_process(
        self,
        user_id: str,
        on_process: Callable[[str, Optional[dict]], Coroutine],
    ) -> None:
        """
        1. Sleep for _delay seconds.
        2. Pop the accumulated text and saved group context.
        3. Call on_process(full_text, saved_ctx).
        4. After on_process returns, if more text arrived during processing,
           auto-start a new timer.
        """
        await asyncio.sleep(self._delay)

        # Enter processing phase
        self._pending.pop(user_id, None)
        self._processing.add(user_id)

        # Restore saved group context
        saved_ctx = self._group_ctx.pop(user_id, None)

        try:
            full_input = self._cache.pop(user_id, "")
            if full_input:
                await on_process(full_input, saved_ctx)
        finally:
            self._processing.discard(user_id)
            # If new messages arrived during processing, auto-start a new timer
            if (user_id in self._cache
                    and self._cache[user_id]
                    and user_id not in self._pending):
                t = asyncio.create_task(self._delayed_process(user_id, on_process))
                self._pending[user_id] = t
