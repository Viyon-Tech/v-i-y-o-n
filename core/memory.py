"""Short-term conversational memory for VIYON CORE.

A small ring buffer of the most recent (role, content) turns, surfaced to the
router and merge prompts so VIYON has light conversational context.
"""

from __future__ import annotations

from collections import deque


class SessionMemory:
    """Ring buffer of the last ``max_turns`` (role, content) turns.

    Args:
        max_turns: How many turns to retain (oldest dropped first).
    """

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._turns: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> None:
        """Append a turn; the oldest is evicted once the buffer is full."""
        self._turns.append((role, content))

    def add_user(self, content: str) -> None:
        """Record a user turn."""
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        """Record an assistant (VIYON) turn."""
        self.add("assistant", content)

    def get_context(self) -> list[tuple[str, str]]:
        """Return the retained turns, oldest first, for prompt inclusion."""
        return list(self._turns)

    def clear(self) -> None:
        """Drop all retained turns."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
