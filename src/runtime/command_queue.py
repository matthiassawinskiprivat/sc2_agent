from collections import deque
from typing import Deque, List, Optional

from src.agent.commands import Command, CommandType


class CommandQueue:
    """
    FIFO queue of manager Command objects.

    Macro commands (BUILD/TRAIN/EXPAND/INJECT) compete for a single ``active``
    slot consumed by the build policy. ``MINE`` / ``MINE_GAS`` are standing
    economy directives. ``ATTACK`` / ``DEFEND`` are standing combat directives
    consumed by stub (or later unit-control) combat — they do not block macro.
    """

    def __init__(self) -> None:
        self._pending: Deque[Command] = deque()
        self._active: Optional[Command] = None
        self._standing_mine: Optional[Command] = None
        self._standing_mine_gas: Optional[Command] = None
        self._standing_combat: Optional[Command] = None

    def clear(self) -> None:
        self._pending.clear()
        self._active = None
        self._standing_mine = None
        self._standing_mine_gas = None
        self._standing_combat = None

    def enqueue(self, command: Command) -> None:
        if command.command_type == CommandType.NO_OP:
            return
        if command.command_type == CommandType.MINE:
            self._standing_mine = command
            return
        if command.command_type == CommandType.MINE_GAS:
            self._standing_mine_gas = command
            return
        if command.command_type in {CommandType.ATTACK, CommandType.DEFEND}:
            self._standing_combat = command
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

    @property
    def standing_mine_gas(self) -> Optional[Command]:
        return self._standing_mine_gas

    @property
    def standing_combat(self) -> Optional[Command]:
        return self._standing_combat

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
            if cmd.command_type == CommandType.MINE_GAS:
                self._standing_mine_gas = cmd
                continue
            if cmd.command_type in {CommandType.ATTACK, CommandType.DEFEND}:
                self._standing_combat = cmd
                continue
            if cmd.is_macro:
                self._active = cmd
                return self._active
        return None

    def complete_active(self) -> None:
        self._active = None

    def peek_pending(self) -> List[Command]:
        return list(self._pending)
