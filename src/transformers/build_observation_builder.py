"""Assembles build-agent Dict observations from bot state."""

import math
from typing import Any, Dict, List, Optional

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.build_constants import (
    BUILD_MAP_RESOLUTION,
    COMMAND_VECTOR_DIM,
    MACRO_VECTOR_DIM,
    MAX_BUILD_WORKERS,
    WORKER_FEAT_DIM,
)
from src.agent.commands import Command, CommandType
from src.runtime.worker_registry import WorkerRegistry, WorkerRole
from src.utils.layer_constants import LayerConstants


class BuildObservationBuilder:
    """Builds local 16x16 map crop + worker features for BuildNetwork."""

    def __init__(self, training_conf: Dict[str, Any]):
        self.training_conf = training_conf
        self.build_map_layers: List[str] = training_conf.get(
            "build_map_layers",
            ["terrain_pathable", "terrain_creep", "terrain_buildable"],
        )
        self.map_res = int(training_conf.get("build_map_resolution", BUILD_MAP_RESOLUTION))

    def build(
        self,
        bot,
        worker_registry: WorkerRegistry,
        active_command: Optional[Command],
        playable_size: tuple[float, float] = (256.0, 256.0),
    ) -> Dict[str, np.ndarray]:
        center = self._command_center(active_command, bot)
        local_map = self._build_local_map(bot, center, active_command)
        worker_features = self._build_worker_features(
            bot, worker_registry, playable_size
        )
        macro_vector = self._build_macro_vector(bot)
        command_vec = self._build_command_vector(active_command, playable_size)
        return {
            "local_map": local_map.astype(np.float32),
            "worker_features": worker_features.astype(np.float32),
            "macro_vector": macro_vector.astype(np.float32),
            "command": command_vec.astype(np.float32),
        }

    def _command_center(self, command: Optional[Command], bot) -> Point2:
        if command and command.target_pos is not None:
            return command.target_pos
        if bot.townhalls:
            return bot.townhalls.first.position
        return bot.start_location

    def _build_local_map(
        self, bot, center: Point2, command: Optional[Command]
    ) -> np.ndarray:
        from src.transformers.observation_transformer import ObservationTransformer

        layer_names = [ln.lower() for ln in self.build_map_layers]
        channels = len(layer_names)
        res = self.map_res
        out = np.zeros((channels, res, res), dtype=np.float32)

        if bot.game_info is None:
            return out

        raw_obs = self._raw_obs_from_bot(bot, command)
        obs_conf = dict(self.training_conf)
        obs_conf["active_layers"] = layer_names
        obs_conf["active_tactical_layers"] = layer_names

        transformer = ObservationTransformer(
            map_size=(res, res),
            training_conf=obs_conf,
        )
        half = res // 2
        cx, cy = float(center.x), float(center.y)
        transformer.scale = 1.0
        transformer.scale_x = 1.0
        transformer.scale_y = 1.0
        transformer.crop_x = math.floor(cx - half)
        transformer.crop_y = math.floor(cy - half)
        transformer.offset_x = 0.0
        transformer.offset_y = 0.0
        transformer._obs_span_w = float(res)
        transformer._obs_span_h = float(res)
        transformer.is_tactical_mode = True
        transformer._grid_x = np.arange(res).reshape(1, -1)
        transformer._grid_y = np.arange(res).reshape(-1, 1)
        transformer._update_active_indices()

        layers = np.zeros(
            (LayerConstants.get_num_layers(), res, res), dtype=np.float32
        )
        gi = bot.game_info
        transformer._cache_pathing_grid(gi)
        transformer._draw_terrain_pathing(gi, layers)
        transformer._draw_terrain_creep(raw_obs, layers)
        transformer._draw_terrain_buildable(raw_obs, layers)

        for ch, ln in enumerate(layer_names):
            global_idx = LayerConstants.LAYERS.get(ln)
            if global_idx is not None:
                out[ch] = layers[global_idx]
        return out

    def _raw_obs_from_bot(self, bot, command: Optional[Command]) -> Dict[str, Any]:
        geyser_mask = self._geyser_mask(bot)
        pending = command.unit_type_id if command else None
        return {
            "game_info": bot.game_info,
            "game_data": bot.game_data,
            "creep": getattr(bot.state, "creep", None),
            "race": self.training_conf.get("race", "zerg"),
            "pending_build_type": pending,
            "geyser_mask": geyser_mask,
            "power_mask": None,
            "units": list(bot.workers),
            "structures": list(bot.structures),
            "enemy_units": [],
            "enemy_structures": [],
        }

    def _geyser_mask(self, bot) -> Optional[np.ndarray]:
        if bot.game_info is None:
            return None
        path = bot.game_info.pathing_grid.data_numpy
        h, w = path.shape
        mask = np.zeros((h, w), dtype=np.float32)
        for geyser in bot.vespene_geyser:
            gx, gy = int(geyser.position.x), int(geyser.position.y)
            if 0 <= gx < w and 0 <= gy < h:
                mask[gy, gx] = 1.0
        return mask

    def _build_worker_features(
        self,
        bot,
        registry: WorkerRegistry,
        playable_size: tuple[float, float],
    ) -> np.ndarray:
        pw, ph = playable_size
        feats = np.zeros((MAX_BUILD_WORKERS, WORKER_FEAT_DIM), dtype=np.float32)
        workers = sorted(
            [u for u in bot.workers],
            key=lambda u: u.tag,
        )[:MAX_BUILD_WORKERS]
        for i, worker in enumerate(workers):
            role = registry.get_role(worker.tag) or WorkerRole.MINING
            feats[i, 0] = float(worker.position.x) / max(pw, 1.0)
            feats[i, 1] = float(worker.position.y) / max(ph, 1.0)
            feats[i, 2] = float(worker.health) / max(float(worker.health_max), 1.0)
            feats[i, 3] = 1.0 if worker.is_idle else 0.0
            feats[i, 4] = float(getattr(worker, "cargo_used", 0) or 0) / 8.0
            feats[i, 5] = 1.0 if role == WorkerRole.MINING else 0.0
            feats[i, 6] = 1.0 if role == WorkerRole.BUILD else 0.0
            feats[i, 7] = 1.0 if role == WorkerRole.MICRO else 0.0
        return feats

    def _build_macro_vector(self, bot) -> np.ndarray:
        vec = np.zeros(MACRO_VECTOR_DIM, dtype=np.float32)
        vec[0] = min(float(bot.minerals), 2000.0) / 2000.0
        vec[1] = min(float(bot.vespene), 2000.0) / 2000.0
        vec[2] = float(bot.supply_used) / max(float(bot.supply_cap), 1.0)
        vec[3] = float(bot.supply_army) / max(float(bot.supply_cap), 1.0)
        vec[4] = float(len(bot.workers)) / float(MAX_BUILD_WORKERS)
        vec[5] = float(len(bot.townhalls))
        vec[6] = float(len(bot.structures)) / 50.0
        state = getattr(bot, "state", None)
        game_loop = float(getattr(state, "game_loop", getattr(bot, "game_loop", 0)))
        vec[7] = game_loop / 10000.0
        return vec

    def _build_command_vector(
        self, command: Optional[Command], playable_size: tuple[float, float]
    ) -> np.ndarray:
        vec = np.zeros(COMMAND_VECTOR_DIM, dtype=np.float32)
        if command is None:
            return vec
        pw, ph = playable_size
        type_idx = list(CommandType).index(command.command_type)
        if 0 <= type_idx < 8:
            vec[type_idx] = 1.0
        if command.unit_type_id is not None:
            vec[8] = float(command.unit_type_id.value) / 2000.0
        if command.target_pos is not None:
            vec[9] = float(command.target_pos.x) / max(pw, 1.0)
            vec[10] = float(command.target_pos.y) / max(ph, 1.0)
        vec[11] = 1.0 if command.urgent else 0.0
        return vec

    @staticmethod
    def unit_type_index(unit_type: Optional[UnitTypeId]) -> int:
        if unit_type is None:
            return 0
        return int(unit_type.value) % 200
