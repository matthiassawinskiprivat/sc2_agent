"""
CTFObsBuilder — spatial + global observation construction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from train.envs.ctf.config import CTFConfig
from train.envs.ctf.region_loader import Point2, RegionDef
from train.envs.ctf.squad_manager import SquadManager, SquadState, SquadStatus

_OBS_VERSION = "ctf_v1"
_N_CHANNELS = 10
_GLOBAL_VEC_DIM = 7

CH_OWN_UNIT = 0
CH_ENEMY_UNIT = 1
CH_OWN_PYLON = 2
CH_ENEMY_PYLON = 3
CH_OWN_HEALTH = 4
CH_ENEMY_HEALTH = 5
CH_ATK_REGION = 6
CH_DEF_REGION = 7
CH_ACTIVE_SQUAD = 8
CH_SQUAD_ROLE = 9


class UnitSnapshot:
    __slots__ = (
        "tag", "type_id", "owner", "position",
        "health", "max_health", "shield", "max_shield",
        "is_pylon",
    )

    def __init__(
        self,
        tag: int,
        type_id: int,
        owner: int,
        position: Point2,
        health: float,
        max_health: float,
        shield: float = 0.0,
        max_shield: float = 0.0,
        is_pylon: bool = False,
    ) -> None:
        self.tag = tag
        self.type_id = type_id
        self.owner = owner
        self.position = position
        self.health = health
        self.max_health = max(max_health, 1.0)
        self.shield = shield
        self.max_shield = max(max_shield, 0.001)
        self.is_pylon = is_pylon

    def normalized_health(self) -> float:
        total = self.health + self.shield
        max_total = self.max_health + self.max_shield
        return float(np.clip(total / max_total, 0.0, 1.0))


class CTFObsBuilder:
    def __init__(
        self,
        config: CTFConfig,
        map_origin: Point2 = Point2(0.0, 0.0),
    ) -> None:
        self.config = config
        self.map_origin = map_origin
        self._Hg = config.global_grid_size
        self._Wg = config.global_grid_size
        self._Ht = config.tactical_grid_size
        self._Wt = config.tactical_grid_size
        self._cpp = config.cells_per_world_unit
        self._max_units = config.max_units_per_squad
        self._max_enemies = config.max_enemies_visible

        self._max_pylons = max(
            sum(1 for sq in config.squads if sq.role == "defend" and sq.side == "player_a"),
            sum(1 for sq in config.squads if sq.role == "defend" and sq.side == "player_b"),
            1,
        )
        self._max_units_per_side = max(
            sum(1 for sq in config.squads if sq.side == "player_a") * 8,
            1,
        )

    def build(
        self,
        active_squad: Optional[SquadState],
        squad_manager: SquadManager,
        step: int,
        units: Optional[List[UnitSnapshot]] = None,
        regions: Optional[Dict[str, RegionDef]] = None,
    ) -> Dict[str, Any]:
        if units is None:
            units = []
        if regions is None:
            regions = {}

        learner_owner = 1 if squad_manager._learner_side == "player_a" else 2
        opponent_owner = 3 - learner_owner

        own_units = [u for u in units if u.owner == learner_owner]
        enemy_units = [u for u in units if u.owner == opponent_owner]
        own_pylons = [u for u in own_units if u.is_pylon]
        enemy_pylons = [u for u in enemy_units if u.is_pylon]

        active_tags: Set[int] = set()
        active_role: float = 0.0
        active_region: Optional[str] = None
        if active_squad is not None:
            active_tags = active_squad.unit_tags
            active_role = 1.0 if active_squad.role == "defend" else 0.0
            active_region = active_squad.current_region

        atk_regions = self._get_attack_regions(squad_manager)
        def_regions = self._get_defend_regions(squad_manager)

        global_grid = self._build_grid(
            H=self._Hg, W=self._Wg, origin=self.map_origin,
            own_units=own_units, enemy_units=enemy_units,
            own_pylons=own_pylons, enemy_pylons=enemy_pylons,
            active_tags=active_tags, active_role=active_role,
            atk_regions=atk_regions, def_regions=def_regions,
            regions=regions,
        )

        tac_origin = self._tactical_origin(active_region, regions)
        tactical_grid = self._build_grid(
            H=self._Ht, W=self._Wt, origin=tac_origin,
            own_units=own_units, enemy_units=enemy_units,
            own_pylons=own_pylons, enemy_pylons=enemy_pylons,
            active_tags=active_tags, active_role=active_role,
            atk_regions=atk_regions, def_regions=def_regions,
            regions=regions,
        )

        global_vec = self._build_global_vec(
            own_units=own_units, enemy_units=enemy_units,
            own_pylons=own_pylons, enemy_pylons=enemy_pylons,
            active_squad=active_squad, active_role=active_role, step=step,
        )

        unit_mask = self._build_unit_mask(active_tags, own_units)
        enemy_positions = self._build_enemy_positions(enemy_units)

        return {
            "spatial_global": global_grid,
            "spatial_tactical": tactical_grid,
            "global_vec": global_vec,
            "unit_mask": unit_mask,
            "enemy_positions": enemy_positions,
            "obs_version": _OBS_VERSION,
        }

    def _build_grid(
        self,
        H: int, W: int,
        origin: Point2,
        own_units: List[UnitSnapshot],
        enemy_units: List[UnitSnapshot],
        own_pylons: List[UnitSnapshot],
        enemy_pylons: List[UnitSnapshot],
        active_tags: Set[int],
        active_role: float,
        atk_regions: List[RegionDef],
        def_regions: List[RegionDef],
        regions: Dict[str, RegionDef],
    ) -> np.ndarray:
        grid = np.zeros((_N_CHANNELS, H, W), dtype=np.float32)

        for u in own_units:
            col, row = self._world_to_grid(u.position, origin, H, W)
            grid[CH_OWN_UNIT, row, col] = 1.0
            grid[CH_OWN_HEALTH, row, col] = u.normalized_health()
            if u.tag in active_tags:
                grid[CH_ACTIVE_SQUAD, row, col] = 1.0

        for u in enemy_units:
            col, row = self._world_to_grid(u.position, origin, H, W)
            grid[CH_ENEMY_UNIT, row, col] = 1.0
            grid[CH_ENEMY_HEALTH, row, col] = u.normalized_health()

        for p in own_pylons:
            col, row = self._world_to_grid(p.position, origin, H, W)
            grid[CH_OWN_PYLON, row, col] = 1.0

        for p in enemy_pylons:
            col, row = self._world_to_grid(p.position, origin, H, W)
            grid[CH_ENEMY_PYLON, row, col] = 1.0

        for reg in atk_regions:
            self._paint_region(grid, CH_ATK_REGION, reg, origin, H, W)
        for reg in def_regions:
            self._paint_region(grid, CH_DEF_REGION, reg, origin, H, W)

        grid[CH_SQUAD_ROLE, :, :] = active_role
        return grid

    def _paint_region(
        self,
        grid: np.ndarray,
        channel: int,
        region: RegionDef,
        origin: Point2,
        H: int,
        W: int,
    ) -> None:
        from train.envs.ctf.region_loader import CircleRegion, RectRegion

        if isinstance(region, CircleRegion):
            cx, cy = self._world_to_grid(region.center, origin, H, W)
            r_cells = int(region.radius * self._cpp)
            for dr in range(-r_cells - 1, r_cells + 2):
                for dc in range(-r_cells - 1, r_cells + 2):
                    row = cy + dr
                    col = cx + dc
                    if 0 <= row < H and 0 <= col < W:
                        wx = origin.x + col / self._cpp
                        wy = origin.y + row / self._cpp
                        if region.contains(Point2(wx, wy)):
                            grid[channel, row, col] = 1.0

        elif isinstance(region, RectRegion):
            c0, r0 = self._world_to_grid(region.min_corner, origin, H, W)
            c1, r1 = self._world_to_grid(region.max_corner, origin, H, W)
            r0, r1 = min(r0, r1), max(r0, r1)
            c0, c1 = min(c0, c1), max(c0, c1)
            r0, r1 = max(0, r0), min(H - 1, r1)
            c0, c1 = max(0, c0), min(W - 1, c1)
            grid[channel, r0:r1 + 1, c0:c1 + 1] = 1.0

    def _world_to_grid(self, pos: Point2, origin: Point2, H: int, W: int) -> Tuple[int, int]:
        col = int((pos.x - origin.x) * self._cpp)
        row = int((pos.y - origin.y) * self._cpp)
        col = int(np.clip(col, 0, W - 1))
        row = int(np.clip(row, 0, H - 1))
        return col, row

    def _tactical_origin(self, active_region: Optional[str], regions: Dict[str, RegionDef]) -> Point2:
        if active_region and active_region in regions:
            center = regions[active_region].center
        else:
            center = self.map_origin
        half_w = (self._Wt / 2.0) / self._cpp
        half_h = (self._Ht / 2.0) / self._cpp
        return Point2(center.x - half_w, center.y - half_h)

    def _build_global_vec(
        self,
        own_units: List[UnitSnapshot],
        enemy_units: List[UnitSnapshot],
        own_pylons: List[UnitSnapshot],
        enemy_pylons: List[UnitSnapshot],
        active_squad: Optional[SquadState],
        active_role: float,
        step: int,
    ) -> np.ndarray:
        vec = np.zeros(_GLOBAL_VEC_DIM, dtype=np.float32)
        vec[0] = len(own_pylons) / max(self._max_pylons, 1)
        vec[1] = len(enemy_pylons) / max(self._max_pylons, 1)
        vec[2] = len(own_units) / max(self._max_units_per_side, 1)
        vec[3] = len(enemy_units) / max(self._max_units_per_side, 1)
        vec[4] = (
            active_squad.unit_count / max(self._max_units, 1)
            if active_squad is not None else 0.0
        )
        vec[5] = active_role
        vec[6] = float(np.clip(step / max(self.config.max_steps, 1), 0.0, 1.0))
        return vec

    def _build_unit_mask(self, active_tags: Set[int], own_units: List[UnitSnapshot]) -> np.ndarray:
        mask = np.zeros(self._max_units, dtype=bool)
        active_list = sorted(tag for tag in active_tags)
        for i, _ in enumerate(active_list[: self._max_units]):
            mask[i] = True
        return mask

    def _build_enemy_positions(self, enemy_units: List[UnitSnapshot]) -> np.ndarray:
        arr = np.zeros((self._max_enemies, 2), dtype=np.float32)
        for i, u in enumerate(enemy_units[: self._max_enemies]):
            arr[i, 0] = float(np.clip(
                (u.position.x - self.map_origin.x) * self._cpp / self._Wg, 0.0, 1.0
            ))
            arr[i, 1] = float(np.clip(
                (u.position.y - self.map_origin.y) * self._cpp / self._Hg, 0.0, 1.0
            ))
        return arr

    def _get_attack_regions(self, sm: SquadManager) -> List[RegionDef]:
        return []

    def _get_defend_regions(self, sm: SquadManager) -> List[RegionDef]:
        return []
