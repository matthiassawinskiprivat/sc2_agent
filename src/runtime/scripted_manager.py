from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.commands import Command, CommandType
from src.runtime.command_queue import CommandQueue
from src.runtime.worker_registry import WorkerRegistry, WorkerRole


@dataclass
class ManagerScriptStep:
    """One scripted manager emission (macro tick or episode start)."""

    command: Command
    worker_roles: Dict[int, WorkerRole] = field(default_factory=dict)
    reassign_micro_tags: Set[int] = field(default_factory=set)


class ScriptedManager:
    """
    Level-driven manager with no neural net. Emits Command objects and worker role
    assignments on macro ticks only; sub-agents run every game step elsewhere.
    """

    def __init__(
        self,
        script_steps: List[ManagerScriptStep],
        macro_tick_interval: int = 8,
        worker_registry: Optional[WorkerRegistry] = None,
        command_queue: Optional[CommandQueue] = None,
    ) -> None:
        self.script_steps = list(script_steps)
        self.macro_tick_interval = max(1, int(macro_tick_interval))
        self.worker_registry = worker_registry or WorkerRegistry()
        self.command_queue = command_queue or CommandQueue()
        self._script_index = 0
        self._game_step = 0
        self._last_macro_step = -self.macro_tick_interval

    def reset(self) -> None:
        self._script_index = 0
        self._game_step = 0
        self._last_macro_step = -self.macro_tick_interval
        self.worker_registry.clear()
        self.command_queue.clear()

    def on_game_step(self) -> None:
        """Call once per SC2 game step."""
        self._game_step += 1
        if self._game_step - self._last_macro_step >= self.macro_tick_interval:
            self._last_macro_step = self._game_step
            self._emit_step_if_available()

    def _emit_step_if_available(self) -> None:
        if self._script_index >= len(self.script_steps):
            return
        step = self.script_steps[self._script_index]
        self._apply_step(step)
        self._script_index += 1

    def _apply_step(self, step: ManagerScriptStep) -> None:
        for tag, role in step.worker_roles.items():
            self.worker_registry.assign(tag, role)
        if step.reassign_micro_tags:
            self.worker_registry.reassign_to_micro(step.reassign_micro_tags)
        self.command_queue.enqueue(step.command)
        self.command_queue.activate_next()

    @staticmethod
    def build_command(
        command_type: CommandType,
        unit_type: Optional[UnitTypeId] = None,
        target_pos: Optional[Point2] = None,
        urgent: bool = True,
    ) -> Command:
        return Command(
            command_type=command_type,
            unit_type_id=unit_type,
            target_pos=target_pos,
            urgent=urgent,
        )

    @staticmethod
    def from_level_config(level_env: Dict[str, Any]) -> "ScriptedManager":
        """Parse manager_script from level YAML env section."""
        script_cfg = level_env.get("manager_script", [])
        macro_tick = int(level_env.get("macro_tick_interval", 8))
        steps: List[ManagerScriptStep] = []
        for entry in script_cfg:
            cmd_type = CommandType[entry["command_type"].upper()]
            unit_type = None
            if entry.get("unit_type"):
                unit_type = UnitTypeId[entry["unit_type"].upper()]
            target = None
            if entry.get("target_pos"):
                tp = entry["target_pos"]
                target = Point2((float(tp[0]), float(tp[1])))
            cmd = ScriptedManager.build_command(cmd_type, unit_type, target, entry.get("urgent", True))
            roles = {}
            for tag, role_name in (entry.get("worker_roles") or {}).items():
                roles[int(tag)] = WorkerRole(role_name.lower())
            micro_tags = {int(t) for t in entry.get("reassign_micro_tags", [])}
            steps.append(ManagerScriptStep(command=cmd, worker_roles=roles, reassign_micro_tags=micro_tags))
        return ScriptedManager(steps, macro_tick_interval=macro_tick)
