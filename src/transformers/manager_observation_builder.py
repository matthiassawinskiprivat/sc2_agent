"""Assembles manager Dict observations (configurable global map + macro vector)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId

from src.agent.commands import INJECT_ENERGY_COST, Command
from src.agent.manager_constants import MANAGER_MACRO_VECTOR_DIM, MANAGER_MAP_RESOLUTION
from src.utils.layer_constants import LayerConstants


class ManagerObservationBuilder:
    def __init__(self, training_conf: Dict[str, Any]):
        self.training_conf = training_conf
        self.active_layers: List[str] = [
            str(x).lower() for x in (training_conf.get("active_layers") or [])
        ]
        self.map_res = int(
            training_conf.get("manager_map_resolution", MANAGER_MAP_RESOLUTION)
        )

    def build(self, bot) -> Dict[str, np.ndarray]:
        return {
            "map": self._build_map(bot).astype(np.float32),
            "macro_vector": self._build_macro_vector(bot).astype(np.float32),
        }

    def _build_map(self, bot) -> np.ndarray:
        from src.transformers.observation_transformer import ObservationTransformer

        layers = self.active_layers or ["terrain_pathable", "terrain_creep"]
        channels = len(layers)
        res = self.map_res
        out = np.zeros((channels, res, res), dtype=np.float32)
        if bot.game_info is None:
            return out

        obs_conf = dict(self.training_conf)
        obs_conf["active_layers"] = layers
        transformer = ObservationTransformer(
            map_size=(res, res),
            training_conf=obs_conf,
        )
        playable = bot.game_info.playable_area
        w = float(getattr(playable, "width", res) or res)
        h = float(getattr(playable, "height", res) or res)
        transformer.scale_x = w / float(res)
        transformer.scale_y = h / float(res)
        transformer.scale = (transformer.scale_x + transformer.scale_y) * 0.5
        transformer.crop_x = int(getattr(playable, "x", 0) or 0)
        transformer.crop_y = int(getattr(playable, "y", 0) or 0)
        transformer.offset_x = 0.0
        transformer.offset_y = 0.0
        transformer._obs_span_w = w
        transformer._obs_span_h = h
        transformer.is_tactical_mode = False
        transformer._update_active_indices()

        buf = np.zeros((LayerConstants.get_num_layers(), res, res), dtype=np.float32)
        gi = bot.game_info
        transformer._cache_pathing_grid(gi)
        transformer._draw_terrain_pathing(gi, buf)
        raw = {
            "game_info": bot.game_info,
            "creep": getattr(bot.state, "creep", None),
            "units": list(getattr(bot, "units", []) or []),
            "structures": list(getattr(bot, "structures", []) or []),
            "enemy_units": list(getattr(bot, "enemy_units", []) or []),
            "enemy_structures": list(getattr(bot, "enemy_structures", []) or []),
        }
        transformer._draw_terrain_creep(raw, buf)
        if hasattr(transformer, "_draw_terrain_buildable"):
            transformer._draw_terrain_buildable(raw, buf)
        for ch, ln in enumerate(layers):
            idx = LayerConstants.LAYERS.get(ln)
            if idx is not None:
                out[ch] = buf[idx]
        return out

    def _build_macro_vector(self, bot) -> np.ndarray:
        vec = np.zeros(MANAGER_MACRO_VECTOR_DIM, dtype=np.float32)
        vec[0] = min(float(getattr(bot, "minerals", 0)), 2000.0) / 2000.0
        vec[1] = min(float(getattr(bot, "vespene", 0)), 2000.0) / 2000.0
        supply_cap = max(float(getattr(bot, "supply_cap", 1) or 1), 1.0)
        vec[2] = float(getattr(bot, "supply_used", 0)) / supply_cap
        vec[3] = float(getattr(bot, "supply_army", 0)) / supply_cap
        workers = getattr(bot, "workers", None)
        vec[4] = float(len(workers) if workers is not None else 0) / 24.0
        townhalls = getattr(bot, "townhalls", None)
        vec[5] = float(len(townhalls) if townhalls is not None else 0) / 4.0
        structures = getattr(bot, "structures", None)
        vec[6] = float(len(structures) if structures is not None else 0) / 50.0
        army = getattr(bot, "units", None)
        army_n = 0
        if army is not None:
            try:
                army_n = int(army.amount) - int(getattr(workers, "amount", len(workers) if workers else 0))
            except Exception:
                army_n = max(0, len(list(army)) - (len(workers) if workers else 0))
        vec[7] = float(max(army_n, 0)) / 50.0
        state = getattr(bot, "state", None)
        game_loop = float(getattr(state, "game_loop", getattr(bot, "game_loop", 0)))
        vec[8] = game_loop / 10000.0
        # Queen energy summary (max energy / inject threshold).
        max_e = 0.0
        try:
            queens = list(bot.units(UnitTypeId.QUEEN))
            for q in queens:
                max_e = max(max_e, float(getattr(q, "energy", 0.0) or 0.0))
        except Exception:
            pass
        vec[9] = min(max_e, 200.0) / 200.0
        vec[10] = 1.0 if max_e >= INJECT_ENERGY_COST else 0.0
        enemy_structs = getattr(bot, "enemy_structures", None)
        vec[11] = float(len(enemy_structs) if enemy_structs is not None else 0) / 20.0
        return vec
