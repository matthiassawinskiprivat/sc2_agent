from collections import deque
from typing import Deque, List, Optional

from src.agent.commands import Command, CommandType


class CommandQueue:
    """
    FIFO queue of manager Command objects.

    Macro commands (BUILD/TRAIN/EXPAND) compete for a single ``active`` slot
    consumed by the build policy. ``MINE`` is a standing economy directive and
    does not block the macro queue.
    """

    def __init__(self) -> None:
        self._pending: Deque[Command] = deque()
        self._active: Optional[Command] = None
        self._standing_mine: Optional[Command] = None

    def clear(self) -> None:
        self._pending.clear()
        self._active = None
        self._standing_mine = None

    def enqueue(self, command: Command) -> None:
        if command.command_type == CommandType.NO_OP:
            return
        if command.command_type == CommandType.MINE:
            self._standing_mine = command
            return
        self._pending.append(command)

    def enqueue_many(self, commands: List[Command]) -> None:
        for cmd in commands:
            self.enqueue(cmd)

    @property
    def active(self) -> Optional[Command]:
        return self._active

    @property
    def standing_mine(self) -> Optional[Command]:
        return self._standing_mine

    def has_active_macro(self) -> bool:
        return self._active is not None and self._active.is_macro

    def activate_next(self) -> Optional[Command]:
        """Promote the next pending macro command if none is active."""
        if self._active is not None:
            return self._active
        while self._pending:
            cmd = self._pending.popleft()
            if cmd.command_type == CommandType.MINE:
                self._standing_mine = cmd
                continue
            if cmd.is_macro:
                self._active = cmd
                return self._active
        return None

    def complete_active(self) -> None:
        self._active = None

    def peek_pending(self) -> List[Command]:
        return list(self._pending)
