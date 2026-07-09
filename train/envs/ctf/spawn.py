"""
SpawnController — roster generation and unit placement for CTF episodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from train.envs.ctf.config import CTFConfig, SquadConfig
from train.envs.ctf.game_data import get_unit_cost
from train.envs.ctf.region_loader import Point2, RegionDef

_SIDE_TO_PLAYER: Dict[str, int] = {"player_a": 1, "player_b": 2}

try:
    from train.envs.ctf.game_data import uid as _uid
    _PYLON_TYPE_ID = _uid("PYLON")
except Exception:
    _PYLON_TYPE_ID = 60


@dataclass
class UnitSpawnSpec:
    type_id: int
    position: Point2
    player_index: int


@dataclass
class RosterEntry:
    type_id: int
    position: Point2


@dataclass
class SquadRoster:
    squad_id: str
    player_index: int
    units: List[RosterEntry] = field(default_factory=list)
    pylon: Optional[RosterEntry] = None


class SpawnController:
    def __init__(
        self,
        config: CTFConfig,
        regions: Dict[str, RegionDef],
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.regions = regions
        self.rng = rng
        self._rosters: Dict[str, SquadRoster] = {}

    def generate_rosters(self, squad_manager: Any = None) -> Dict[str, SquadRoster]:
        self._rosters = {}
        for sq_cfg in self.config.squads:
            roster = self._generate_squad_roster(sq_cfg)
            self._rosters[sq_cfg.id] = roster
        return self._rosters

    def spawn_all(self, sc2_client: Any) -> None:
        if not self._rosters:
            raise RuntimeError("Call generate_rosters() before spawn_all().")
        self._collect_spawn_specs()

    def get_roster(self, squad_id: str) -> Optional[SquadRoster]:
        return self._rosters.get(squad_id)

    def collect_spawn_specs(self) -> List[UnitSpawnSpec]:
        return self._collect_spawn_specs()

    def _generate_squad_roster(self, sq_cfg: SquadConfig) -> SquadRoster:
        player_index = _SIDE_TO_PLAYER[sq_cfg.side]
        roster = SquadRoster(squad_id=sq_cfg.id, player_index=player_index)
        spawn_center = self._get_region_center(sq_cfg.spawn_region)

        minerals_left = sq_cfg.budget.minerals
        vespene_left = sq_cfg.budget.vespene
        pool = list(sq_cfg.spawn_pool)

        unit_index = 0
        while True:
            affordable = [
                t for t in pool
                if self._can_afford(t, minerals_left, vespene_left)
            ]
            if not affordable:
                break
            type_id = int(self.rng.choice(affordable))
            m_cost, v_cost = get_unit_cost(type_id)
            minerals_left -= m_cost
            vespene_left -= v_cost
            pos = self._spawn_position(spawn_center, unit_index)
            roster.units.append(RosterEntry(type_id=type_id, position=pos))
            unit_index += 1

        if sq_cfg.role == "defend" and sq_cfg.defend_anchors:
            anchor_center = self._get_region_center(sq_cfg.defend_anchors[0])
            roster.pylon = RosterEntry(type_id=_PYLON_TYPE_ID, position=anchor_center)

        return roster

    def _can_afford(self, type_id: int, minerals: int, vespene: int) -> bool:
        m_cost, v_cost = get_unit_cost(type_id)
        return m_cost <= minerals and v_cost <= vespene

    def _get_region_center(self, region_name: str) -> Point2:
        if region_name not in self.regions:
            raise KeyError(
                f"spawn_region '{region_name}' not found in map regions. "
                f"Available: {sorted(self.regions)}"
            )
        reg = self.regions[region_name]
        if hasattr(reg, "center"):
            return reg.center
        raise ValueError(f"Region '{region_name}' has no center attribute.")

    def _spawn_position(self, center: Point2, unit_index: int) -> Point2:
        step = 1.0
        cols = 4
        col = unit_index % cols
        row = unit_index // cols
        x = center.x + (col - cols / 2.0 + 0.5) * step
        y = center.y + (row - 1.0) * step
        return Point2(x, y)

    def _collect_spawn_specs(self) -> List[UnitSpawnSpec]:
        specs: List[UnitSpawnSpec] = []
        for roster in self._rosters.values():
            for entry in roster.units:
                specs.append(UnitSpawnSpec(
                    type_id=entry.type_id,
                    position=entry.position,
                    player_index=roster.player_index,
                ))
            if roster.pylon is not None:
                specs.append(UnitSpawnSpec(
                    type_id=roster.pylon.type_id,
                    position=roster.pylon.position,
                    player_index=roster.player_index,
                ))
        return specs
