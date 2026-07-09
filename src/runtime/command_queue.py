from collections import deque
from typing import Deque, List, Optional

from src.agent.commands import Command, CommandType


class CommandQueue:
    """
    FIFO queue of manager Command objects consumed by the build agent.
    Early curriculum levels typically keep one active macro command at a time.
    """

    def __init__(self) -> None:
        self._pending: Deque[Command] = deque()
        self._active: Optional[Command] = None

    def clear(self) -> None:
        self._pending.clear()
        self._active = None

    def enqueue(self, command: Command) -> None:
        if command.command_type == CommandType.NO_OP:
            return
        self._pending.append(command)

    def enqueue_many(self, commands: List[Command]) -> None:
        for cmd in commands:
            self.enqueue(cmd)

    @property
    def active(self) -> Optional[Command]:
        return self._active

    def has_active_macro(self) -> bool:
        return self._active is not None and self._active.is_macro

    def activate_next(self) -> Optional[Command]:
        """Promote the next pending macro command if none is active."""
        if self._active is not None:
            return self._active
        while self._pending:
            cmd = self._pending.popleft()
            if cmd.is_macro:
                self._active = cmd
                return self._active
        return None

    def complete_active(self) -> None:
        self._active = None

    def peek_pending(self) -> List[Command]:
        return list(self._pending)
