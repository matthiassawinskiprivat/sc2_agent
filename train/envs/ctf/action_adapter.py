"""
CTFActionAdapter — translates universal action tensors to SC2 commands.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from train.envs.ctf.config import CTFConfig
from train.envs.ctf.region_loader import Point2
from train.envs.ctf.squad_manager import SquadState


class ActionType(IntEnum):
    MOVE = 0
    ATTACK = 1
    HOLD = 2


@dataclass
class SC2MoveCommand:
    unit_tag: int
    target_position: Point2


@dataclass
class SC2AttackCommand:
    unit_tag: int
    target_tag: int
    target_position: Point2


@dataclass
class SC2HoldCommand:
    unit_tag: int


SC2Command = SC2MoveCommand | SC2AttackCommand | SC2HoldCommand


class CTFActionAdapter:
    def __init__(self, config: CTFConfig, move_scale_wu: float = 8.0) -> None:
        self.config = config
        self._max_units = config.max_units_per_squad
        self._max_enemies = config.max_enemies_visible
        self._move_scale = move_scale_wu

    def to_sc2_commands(
        self,
        actions: Dict[str, np.ndarray],
        squads: List[SquadState],
        unit_positions: Optional[Dict[int, Point2]] = None,
        enemy_tags: Optional[List[int]] = None,
        objective_resolver: Optional[Any] = None,
    ) -> List[SC2Command]:
        if unit_positions is None:
            unit_positions = {}
        if enemy_tags is None:
            enemy_tags = []

        commands: List[SC2Command] = []
        squad_map: Dict[str, SquadState] = {s.squad_id: s for s in squads}

        for squad_id, action_tensor in actions.items():
            state = squad_map.get(squad_id)
            if state is None or not state.is_active:
                continue
            squad_commands = self._squad_to_commands(
                action_tensor=action_tensor,
                squad=state,
                unit_positions=unit_positions,
                enemy_tags=enemy_tags,
            )
            commands.extend(squad_commands)

        return commands

    def validate_action(self, action_tensor: np.ndarray, unit_mask: np.ndarray) -> np.ndarray:
        assert action_tensor.shape == (self._max_units, 4)
        a = action_tensor.copy().astype(np.float32)
        a[:, 0] = np.clip(a[:, 0], -1.0, 1.0)
        a[:, 1] = np.clip(a[:, 1], -1.0, 1.0)
        a[:, 2] = np.clip(a[:, 2], 0.0, float(self._max_enemies - 1))
        a[:, 3] = np.clip(np.round(a[:, 3]), 0.0, 2.0)
        if unit_mask is not None:
            for i in range(self._max_units):
                if i >= len(unit_mask) or not unit_mask[i]:
                    a[i] = 0.0
        return a

    def _squad_to_commands(
        self,
        action_tensor: np.ndarray,
        squad: SquadState,
        unit_positions: Dict[int, Point2],
        enemy_tags: List[int],
    ) -> List[SC2Command]:
        commands: List[SC2Command] = []
        unit_tag_list = sorted(squad.unit_tags)[: self._max_units]

        for i, tag in enumerate(unit_tag_list):
            row = action_tensor[i]
            move_x = float(np.clip(row[0], -1.0, 1.0))
            move_y = float(np.clip(row[1], -1.0, 1.0))
            atk_idx = int(np.clip(round(float(row[2])), 0, max(len(enemy_tags) - 1, 0)))
            act_type = ActionType(int(np.clip(round(float(row[3])), 0, 2)))

            current_pos = unit_positions.get(tag, Point2(0.0, 0.0))
            target_pos = Point2(
                current_pos.x + move_x * self._move_scale,
                current_pos.y + move_y * self._move_scale,
            )

            if act_type == ActionType.HOLD:
                commands.append(SC2HoldCommand(unit_tag=tag))
            elif act_type == ActionType.ATTACK and atk_idx < len(enemy_tags):
                commands.append(SC2AttackCommand(
                    unit_tag=tag, target_tag=enemy_tags[atk_idx], target_position=target_pos,
                ))
            elif act_type == ActionType.ATTACK and squad.role == "attack":
                commands.append(SC2AttackCommand(
                    unit_tag=tag, target_tag=0, target_position=target_pos,
                ))
            else:
                commands.append(SC2MoveCommand(unit_tag=tag, target_position=target_pos))

        return commands

    def sample_action(self, unit_mask: np.ndarray) -> np.ndarray:
        action = np.zeros((self._max_units, 4), dtype=np.float32)
        for i in range(self._max_units):
            if i < len(unit_mask) and unit_mask[i]:
                action[i, 0] = float(np.random.uniform(-1.0, 1.0))
                action[i, 1] = float(np.random.uniform(-1.0, 1.0))
                action[i, 2] = float(np.random.randint(0, self._max_enemies))
                action[i, 3] = float(np.random.randint(0, 3))
        return action
