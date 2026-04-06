"""
Professional Observation Transformer
Fixed:
1. COMMAND_ASSIGNMENT now draws ALL allied units (Bots don't 'select' units).
2. Preserved scaling/cropping logic.
3. Added Tactical Fovea (32x32 zoom centered on active unit).
4. Fixed Scale collision on small maps (e.g. 32x32).
5. Fixed Regressions: Beacon layers, Tactical Command Assignment, Terrain rendering.
6. Restored: Effects, Creep, Transport, and DPS drawing logic.
"""

import numpy as np
import json
import os
import math
from typing import Dict, Any, List, Optional, Union
import cv2

# SC2 Dependencies
from sc2.unit import Unit
from sc2.game_data import GameData
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.effect_id import EffectId
from sc2.ids.ability_id import AbilityId

# Project Constants
from src.utils.layer_constants import LayerConstants
from src.utils.command_constants import CommandConstants

# Playable size (game units) used as reference so COMMAND_ASSIGNMENT stays as visible as on 128×128 maps.
_ASSIGNMENT_REFERENCE_PLAYABLE = 128


class ObservationTransformer:
    def __init__(self, map_size: tuple[int, int] = (128, 128), unit_data_file: str = None, training_conf: Dict = None):
        self.feature_map_width = map_size[0]
        self.feature_map_height = map_size[1]
        self.training_conf = training_conf if training_conf is not None else {}

        # Transformation State
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.crop_x = 0.0
        self.crop_y = 0.0

        # Layer Map Selection
        self.current_layer_map = LayerConstants.LAYERS
        self.is_tactical_mode = False

        # Optimization: Pre-calculate active indices for fast lookup
        self.active_layers = self.training_conf.get('active_layers', [])
        arch = self.training_conf.get("agent_architecture", "original")
        
        # CRITICAL FIX: Default to 5 tactical layers for Universal if empty
        if not self.active_layers and arch == "universal":
            self.active_layers = [
                LayerConstants.ALLIED_UNIT_DENSITY,
                LayerConstants.ENEMY_UNIT_DENSITY,
                LayerConstants.TERRAIN_PATHABLE,
                LayerConstants.COMMAND_OBJECTIVE,
                LayerConstants.COMMAND_ASSIGNMENT
            ]
            
        self.active_tactical_layers = self.training_conf.get("active_tactical_layers", [])
        if not self.active_tactical_layers:
            self.active_tactical_layers = self.active_layers

        from loguru import logger
        logger.debug(f"ObservationTransformer init: Arch={arch}, ActiveLayers={len(self.active_layers)}, TacticalLayers={len(self.active_tactical_layers)}")

        self._update_active_indices()

        # Caching
        self._terrain_cache = None
        self._last_map_name = None
        self._last_structure_count = 0
        self._h_min = None
        self._h_max = None

        # Pre-allocated grids
        self._grid_y = np.arange(self.feature_map_height).reshape(-1, 1)
        self._grid_x = np.arange(self.feature_map_width).reshape(1, -1)

        self._liberator_positions = {'allied': [], 'enemy': []}

        # Load DB
        if unit_data_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            unit_data_file = os.path.join(base_dir, "data", "unit_data.json")

        self.fallback_db = self._load_fallback_db(unit_data_file)

    def _update_active_indices(self):
        """Updates active indices based on current layer map and pruning config."""
        # Tactical draws must honor active_tactical_layers so DPS/vitals are not
        # skipped when global active_layers is a smaller universal subset.
        if self.is_tactical_mode:
            names = self.active_tactical_layers
        else:
            names = self.active_layers
        if names:
            self.active_indices = {self.current_layer_map[ln.lower()] for ln in names if ln.lower() in self.current_layer_map}
        else:
            self.active_indices = set(range(len(self.current_layer_map)))

    def _load_fallback_db(self, path: str) -> Dict:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _get_unit_data_from_json(self, unit_name: str) -> Optional[Dict[str, Any]]:
        return self.fallback_db.get(unit_name.upper())

    def _get_numpy_data(self, grid_obj) -> np.ndarray:
        if hasattr(grid_obj, "data_numpy"):
            return grid_obj.data_numpy
        return grid_obj

    def transform_tactical(self, raw_obs: Dict[str, Any], active_unit: Unit) -> np.ndarray:
        """
        Creates a 32x32 tactical zoom centered on active_unit.
        Uses TACTICAL_LAYERS for reduced state space.
        """
        if active_unit is None:
            if self.active_tactical_layers:
                num_channels = sum(1 for ln in self.active_tactical_layers
                                   if ln.lower() in LayerConstants.TACTICAL_LAYERS)
            else:
                num_channels = LayerConstants.get_num_tactical_layers()
            return np.zeros((num_channels, 32, 32), dtype=np.float32)

        total_layers = LayerConstants.get_num_tactical_layers()
        layers = np.zeros((total_layers, 32, 32), dtype=np.float32)
        
        # Save original state
        old_state = (self.scale, self.offset_x, self.offset_y, self.crop_x, self.crop_y, 
                     self.feature_map_width, self.feature_map_height, 
                     self.current_layer_map, self.is_tactical_mode,
                     self._grid_x, self._grid_y)
        
        # Tactical setup (32x32 centered on unit)
        self.scale = 1.0
        self.feature_map_width = 32
        self.feature_map_height = 32
        self.crop_x = active_unit.position.x - 16.0
        self.crop_y = active_unit.position.y - 16.0
        self.offset_x = 0
        self.offset_y = 0
        self.current_layer_map = LayerConstants.TACTICAL_LAYERS
        self.is_tactical_mode = True
        self._update_active_indices()

        self._grid_x = np.arange(32).reshape(1, -1)
        self._grid_y = np.arange(32).reshape(-1, 1)

        try:
            game_info = raw_obs.get('game_info')
            allied_units = raw_obs.get('units') or []
            enemy_units = raw_obs.get('enemy_units') or []
            allied_structures = raw_obs.get('structures') or []
            enemy_structures = raw_obs.get('enemy_structures') or []
            game_data = raw_obs.get('game_data')
            effects = raw_obs.get('effects') or []
            
            # Draw
            self._draw_terrain_pathing(game_info, layers)
            self._draw_terrain_creep(raw_obs, layers)

            # Allied
            for unit in list(allied_units) + list(allied_structures):
                if abs(unit.position.x - active_unit.position.x) > 17 or abs(unit.position.y - active_unit.position.y) > 17:
                    continue
                self._process_single_unit(unit, layers, False, 32, 32, game_data)

            # Enemy
            for unit in list(enemy_units) + list(enemy_structures):
                if abs(unit.position.x - active_unit.position.x) > 17 or abs(unit.position.y - active_unit.position.y) > 17:
                    continue
                self._process_single_unit(unit, layers, True, 32, 32, game_data)

            # Enemy DPS from units outside view whose weapon range reaches in
            dps_idx = LayerConstants.TACTICAL_LAYERS.get(LayerConstants.ENEMY_G_DPS_BASE)
            if dps_idx is not None and dps_idx in self.active_indices:
                for unit in list(enemy_units) + list(enemy_structures):
                    dx = abs(unit.position.x - active_unit.position.x)
                    dy = abs(unit.position.y - active_unit.position.y)
                    if dx <= 17 and dy <= 17:
                        continue
                    if dx > 32 or dy > 32:
                        continue
                    px, py = self._get_pixel_coords_unclamped(unit.position)
                    if px is not None:
                        self._draw_unit_dps(unit, layers, True, px, py, game_data)

            # Draw Command Info (High-priority markers)
            tactical_raw_obs = raw_obs.copy()
            tactical_raw_obs['selected_units'] = [active_unit]
            self._draw_command_info(tactical_raw_obs, layers)

            if effects: self._draw_effects(effects, layers)

            # Normalize DPS layers with tanh (matches global transform)
            for dps_layer_name in [LayerConstants.ALLIED_G_DPS_BASE, LayerConstants.ENEMY_G_DPS_BASE]:
                dps_layer_idx = LayerConstants.TACTICAL_LAYERS.get(dps_layer_name)
                if dps_layer_idx is not None:
                    layers[dps_layer_idx] = np.tanh(layers[dps_layer_idx] / 50.0)

            # Clip
            layers = np.clip(layers, 0.0, 1.0).astype(np.float32)

            # Slice to active tactical layers (preserving config order)
            if self.active_tactical_layers:
                indices = [LayerConstants.TACTICAL_LAYERS[ln.lower()]
                           for ln in self.active_tactical_layers
                           if ln.lower() in LayerConstants.TACTICAL_LAYERS]
                tactical_data = layers[indices]
            else:
                tactical_data = layers
        finally:
            # Restore state
            (self.scale, self.offset_x, self.offset_y, self.crop_x, self.crop_y, 
             self.feature_map_width, self.feature_map_height, 
             self.current_layer_map, self.is_tactical_mode,
             self._grid_x, self._grid_y) = old_state
            self._update_active_indices()

        return tactical_data

    def transform(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Global transformation (128x128)."""
        total_layers = LayerConstants.get_num_layers()
        layers = np.zeros((total_layers, self.feature_map_height, self.feature_map_width), dtype=np.float32)

        # Setup Command Vector
        raw_cmd = raw_obs.get('high_level_command', 0)
        cmd_idx = CommandConstants.get_index(raw_cmd)
        command_vector = np.zeros(10, dtype=np.float32)
        if 0 <= cmd_idx < 10: command_vector[cmd_idx] = 1.0
        else: command_vector[0] = 1.0

        # Setup Drawing State
        self.current_layer_map = LayerConstants.LAYERS
        self.is_tactical_mode = False
        self._update_active_indices()

        allied_units = raw_obs.get('units') or []
        enemy_units = raw_obs.get('enemy_units') or []
        allied_structures = raw_obs.get('structures') or []
        enemy_structures = raw_obs.get('enemy_structures') or []
        effects = raw_obs.get('effects') or []
        game_info = raw_obs.get('game_info')
        game_data = raw_obs.get('game_data')

        # 1. Scaling
        if game_info:
            area = getattr(game_info, 'playable_area', None)
            if area:
                self.crop_x, self.crop_y, mw, mh = area.x, area.y, area.width, area.height
            else:
                self.crop_x, self.crop_y = 0, 0
                mw, mh = game_info.map_size.width, game_info.map_size.height
        else:
            self.crop_x, self.crop_y, mw, mh = 0, 0, 200, 200

        self.scale = min(self.feature_map_width / max(mw, 1), self.feature_map_height / max(mh, 1))
        self.offset_x = max(0, (self.feature_map_width - int(mw * self.scale)) // 2)
        self.offset_y = max(0, (self.feature_map_height - int(mh * self.scale)) // 2)

        # 2. Draw
        current_struct_count = len(allied_structures) + len(enemy_structures)
        self._draw_terrain_pathing(game_info, layers, current_struct_count)
        self._draw_terrain_creep(raw_obs, layers)

        self._liberator_positions = {'allied': [], 'enemy': []}
        lib_enum = getattr(UnitTypeId, 'LIBERATORAG', 154)
        lib_val = lib_enum.value if hasattr(lib_enum, "value") else 154

        for unit in list(allied_units) + list(allied_structures):
            if "MINERAL" in getattr(unit.type_id, "name", "").upper(): continue
            self._process_single_unit(unit, layers, False, mw, mh, game_data)
            if (unit.type_id.value if hasattr(unit.type_id, "value") else unit.type_id) == lib_val:
                self._liberator_positions['allied'].append(unit.position)

        for unit in list(enemy_units) + list(enemy_structures):
            if "MINERAL" in getattr(unit.type_id, "name", "").upper(): continue
            self._process_single_unit(unit, layers, True, mw, mh, game_data)
            if (unit.type_id.value if hasattr(unit.type_id, "value") else unit.type_id) == lib_val:
                self._liberator_positions['enemy'].append(unit.position)

        if effects: self._draw_effects(effects, layers)
        self._draw_command_info(raw_obs, layers)
        self._draw_transport_info(raw_obs, layers)

        # 3. Post-processing
        dps_start = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]
        hps_end = LayerConstants.LAYERS[LayerConstants.ENEMY_HPS_FIELD] + 1
        layers[dps_start:hps_end] = np.tanh(layers[dps_start:hps_end] / 50.0)
        layers[0:dps_start] = np.log1p(layers[0:dps_start])
        layers = np.clip(layers, 0.0, 1.0).astype(np.float32)

        if self.active_layers:
            indices = [LayerConstants.LAYERS[layer_name.lower()] for layer_name in self.active_layers if layer_name.lower() in LayerConstants.LAYERS]
            map_data = layers[indices]
        else:
            map_data = layers

        # ALWAYS return Dict to match the gymnasium.spaces.Dict in Level0Env
        return {"map": map_data, "command": command_vector}

    def _get_pixel_coords(self, position) -> tuple:
        if position is None: return None, None
        px = int((position.x - self.crop_x) * self.scale + self.offset_x)
        py = int(self.feature_map_height - 1 - ((position.y - self.crop_y) * self.scale + self.offset_y))
        if 0 <= px < self.feature_map_width and 0 <= py < self.feature_map_height:
            return px, py
        return None, None

    def _get_pixel_coords_unclamped(self, position) -> tuple:
        """Same as _get_pixel_coords but without bounds check.
        Used for DPS circles whose center is outside the view but whose
        radius overlaps it. _draw_circle_vectorized clips correctly."""
        if position is None: return None, None
        px = int((position.x - self.crop_x) * self.scale + self.offset_x)
        py = int(self.feature_map_height - 1 - ((position.y - self.crop_y) * self.scale + self.offset_y))
        return px, py

    def _process_single_unit(self, unit: Unit, layers: np.ndarray, is_enemy: bool,
                             map_w: int, map_h: int, game_data: GameData):
        px, py = self._get_pixel_coords(unit.position)
        if px is None: return

        u_id = unit.type_id.value if hasattr(unit.type_id, "value") else unit.type_id
        u_name = getattr(unit.type_id, "name", "").upper()
        is_beacon = "BEACON" in u_name

        # BEACON logic: Render ONLY to ENEMY_UNIT_DENSITY (and Command Objective later)
        if is_beacon:
            idx = self.current_layer_map.get(LayerConstants.ENEMY_UNIT_DENSITY)
            if idx is not None:
                r = max(1, int(1.5 * self.scale))
                self._draw_circle_vectorized(layers, idx, px, py, 0, r, 20.0, True)
            return

        # Normal Units
        if is_enemy and unit.is_cloaked and not unit.is_detected:
            if not getattr(unit, "is_snapshot", False): return

        if u_id == 170: # ScannerSweep
            idx = self.current_layer_map.get(LayerConstants.ENEMY_DETECTION if is_enemy else LayerConstants.ALLIED_DETECTION)
            if idx is not None: self._draw_circle_vectorized(layers, idx, px, py, 13.0, None, 1.0)
            return

        if u_id == 135: # ForceField
            idx_p = self.current_layer_map.get(LayerConstants.TERRAIN_PATHABLE)
            if idx_p is not None: self._draw_circle_vectorized(layers, idx_p, px, py, 1.5, None, 0.0, False)
            idx_d = self.current_layer_map.get(LayerConstants.ENEMY_UNIT_DENSITY if is_enemy else LayerConstants.ALLIED_UNIT_DENSITY)
            if idx_d is not None: self._draw_circle_vectorized(layers, idx_d, px, py, 0, 0, 1.0, True)
            return

        self._draw_unit_body(unit, layers, is_enemy, px, py)
        if not self.is_tactical_mode:
            self._draw_unit_vitals(unit, layers, is_enemy, px, py)
            self._draw_unit_detection(unit, layers, is_enemy, px, py)
            self._draw_unit_healing(unit, layers, is_enemy, px, py, game_data)
            self._draw_unit_dps(unit, layers, is_enemy, px, py, game_data)
        else:
            self._draw_unit_vitals(unit, layers, is_enemy, px, py)
            self._draw_unit_dps(unit, layers, is_enemy, px, py, game_data)

    def _draw_unit_body(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int):
        prefix = "enemy" if is_enemy else "allied"
        idx_den = self.current_layer_map.get(f"{prefix}_unit_density")
        if idx_den is None: return

        # Radius Logic: Tactical/Universal uses r=0 (Single Pixel)
        is_universal = self.training_conf.get("agent_architecture") == "universal"
        if self.is_tactical_mode or is_universal:
            r = 0
        else:
            r = int(unit.radius * self.scale) if unit.is_structure else 0

        val = 1.0
        if not is_enemy:
            cd = getattr(unit, 'weapon_cooldown', 0.0)
            val = 1.0 - (0.7 * (min(cd, 60.0) / 60.0))

        self._draw_circle_vectorized(layers, idx_den, px, py, 0, r, val, True)

        if not self.is_tactical_mode:
            # Attribute Layers (Not in tactical)
            for attr in ["biological", "mechanical", "armored", "massive", "flying"]:
                idx = self.current_layer_map.get(f"{prefix}_unit_{attr}")
                if idx is not None and getattr(unit, f"is_{attr}", False):
                    self._draw_circle_vectorized(layers, idx, px, py, 0, r, 1.0, True)

    def _draw_command_info(self, raw_obs, layers):
        idx_sel = self.current_layer_map.get(LayerConstants.COMMAND_ASSIGNMENT)
        if idx_sel is not None:
            units = raw_obs.get('selected_units') or raw_obs.get('units', [])
            for unit in units:
                if "BEACON" in getattr(unit.type_id, "name", "").upper(): continue
                px, py = self._get_pixel_coords(unit.position)
                if px is not None:
                    if self.is_tactical_mode:
                        r = 0
                    else:
                        ref_scale = min(self.feature_map_width, self.feature_map_height) / float(
                            _ASSIGNMENT_REFERENCE_PLAYABLE
                        )
                        r_draw = int(unit.radius * self.scale)
                        r_ref = int(unit.radius * ref_scale)
                        r = max(1, r_draw, r_ref)
                    self._draw_circle_vectorized(layers, idx_sel, px, py, 0, r, 1.0, True)

        idx_obj = self.current_layer_map.get(LayerConstants.COMMAND_OBJECTIVE)
        if idx_obj is not None:
            targets = raw_obs.get('command_target')
            if targets is None: 
                targets = []
            elif hasattr(targets, "x") or hasattr(targets, "position"):
                # Single target (Point2 or Unit), wrap it
                targets = [targets]
            elif not isinstance(targets, (list, tuple, set)): 
                targets = [targets]

            for target in targets:
                if target is None: continue
                val, r = 1.0, 3
                
                # Check if it's a beacon
                is_beacon = False
                u_id = getattr(target, "type_id", None)
                if u_id:
                    u_name = getattr(u_id, "name", str(u_id)).upper()
                    if "BEACON" in u_name:
                        is_beacon = True
                
                if is_beacon:
                    val, r = 20.0, max(3, int(1.5 * self.scale))

                pos = getattr(target, "position", target)
                px, py = self._get_pixel_coords(pos)
                if px is not None:
                    self._draw_circle_vectorized(layers, idx_obj, px, py, 0, r, val, False)

    def _draw_terrain_pathing(self, game_info, layers, current_structure_count=0):
        layer_idx = self.current_layer_map.get(LayerConstants.TERRAIN_PATHABLE)
        if layer_idx is None: return

        current_map_name = getattr(game_info, 'map_name', '')
        # Global View Cache (Only if not in tactical mode)
        if not self.is_tactical_mode:
            if (self._terrain_cache is not None and
                self._last_map_name == current_map_name and
                self._last_structure_count == current_structure_count):
                layers[layer_idx] = self._terrain_cache
                return

        if not hasattr(game_info, 'pathing_grid') or not hasattr(game_info, 'terrain_height'):
            return

        path_data = self._get_numpy_data(game_info.pathing_grid)
        height_data = self._get_numpy_data(game_info.terrain_height)
        
        # Calculate Global Walkable Range once for consistency
        if self._h_min is None or self._h_max is None:
            walkable_mask = path_data > 0
            if np.any(walkable_mask):
                walkable_heights = height_data[walkable_mask]
                self._h_min = walkable_heights.min()
                self._h_max = walkable_heights.max()
            else:
                self._h_min = height_data.min()
                self._h_max = height_data.max()

        total_w = game_info.map_size.width
        total_h = game_info.map_size.height

        x_indices = np.arange(self.feature_map_width)
        y_indices = np.arange(self.feature_map_height)
        sc2_x = ((x_indices - self.offset_x) / self.scale) + self.crop_x
        sc2_y = ((self.feature_map_height - 1 - y_indices) - self.offset_y) / self.scale + self.crop_y

        X_map, Y_map = np.meshgrid(sc2_x.astype(int), sc2_y.astype(int))
        valid_mask = (X_map >= 0) & (X_map < total_w) & (Y_map >= 0) & (Y_map < total_h)
        
        if np.any(valid_mask):
            v_x, v_y = X_map[valid_mask], Y_map[valid_mask]
            sampled_path = path_data[v_y, v_x]
            sampled_height = height_data[v_y, v_x]
            
            h_range = max(self._h_max - self._h_min, 1.0)
            # Base value 0.2 to 1.0 for better visibility in UI and Agent
            norm_height = 0.2 + ((sampled_height - self._h_min) / h_range) * 0.8
            
            res = np.zeros((self.feature_map_height, self.feature_map_width), dtype=np.float32)
            res[valid_mask] = norm_height * (sampled_path > 0).astype(np.float32)
            layers[layer_idx] = res

            if not self.is_tactical_mode:
                self._terrain_cache = res.copy()
                self._last_map_name = current_map_name
                self._last_structure_count = current_structure_count

    def _draw_terrain_creep(self, raw_obs, layers):
        idx = self.current_layer_map.get(LayerConstants.TERRAIN_CREEP)
        creep_map = raw_obs.get('creep')
        if idx is None or not creep_map: return
        data = self._get_numpy_data(creep_map)
        game_info = raw_obs.get('game_info')
        total_w, total_h = game_info.map_size.width, game_info.map_size.height
        
        x_indices = np.arange(self.feature_map_width)
        y_indices = np.arange(self.feature_map_height)
        sc2_x = ((x_indices - self.offset_x) / self.scale) + self.crop_x
        sc2_y = ((self.feature_map_height - 1 - y_indices) - self.offset_y) / self.scale + self.crop_y
        X_map, Y_map = np.meshgrid(sc2_x.astype(int), sc2_y.astype(int))
        valid_mask = (X_map >= 0) & (X_map < total_w) & (Y_map >= 0) & (Y_map < total_h)
        if np.any(valid_mask):
            layers[idx][valid_mask] = data[Y_map[valid_mask], X_map[valid_mask]]

    def _draw_circle_vectorized(self, layers, idx, cx, cy, radius_game, radius_px_override=None, value=1.0, accumulate=False):
        if idx not in self.active_indices: return
        r = int(radius_px_override) if radius_px_override is not None else max(1, int(radius_game * self.scale))
        if r == 0:
            if accumulate: layers[idx, cy, cx] += value
            else: layers[idx, cy, cx] = value
            return
        r_sq = r**2
        x_min, x_max = max(0, cx-r), min(self.feature_map_width, cx+r+1)
        y_min, y_max = max(0, cy-r), min(self.feature_map_height, cy+r+1)
        if x_min >= x_max or y_min >= y_max: return
        Y, X = self._grid_y[y_min:y_max, :], self._grid_x[:, x_min:x_max]
        mask = ((X - cx)**2 + (Y - cy)**2) <= r_sq
        if accumulate: layers[idx, y_min:y_max, x_min:x_max][mask] += value
        else: layers[idx, y_min:y_max, x_min:x_max][mask] = value

    def _draw_unit_vitals(self, unit, layers, is_enemy, px, py):
        prefix = "enemy" if is_enemy else "allied"
        idx_hp = self.current_layer_map.get(f"{prefix}_health")
        if idx_hp is not None:
            max_hp = (unit.health_max + unit.shield_max) or 1
            self._draw_circle_vectorized(layers, idx_hp, px, py, 0, 0, (unit.health+unit.shield)/max_hp, True)
        idx_nrg = self.current_layer_map.get(f"{prefix}_energy")
        if idx_nrg is not None and unit.energy_max > 0:
            self._draw_circle_vectorized(layers, idx_nrg, px, py, 0, 0, unit.energy/unit.energy_max, True)

    def _draw_unit_detection(self, unit, layers, is_enemy, px, py):
        prefix = "enemy" if is_enemy else "allied"
        idx = self.current_layer_map.get(f"{prefix}_detection")
        if idx is not None and unit.detect_range > 0:
            self._draw_circle_vectorized(layers, idx, px, py, unit.detect_range, None, 1.0, False)

    def _draw_unit_healing(self, unit, layers, is_enemy, px, py, game_data):
        prefix = "enemy" if is_enemy else "allied"
        idx = self.current_layer_map.get(f"{prefix}_hps_field")
        if idx is None: return
        hps, rng = 0.0, 0.0
        if unit.type_id == UnitTypeId.MEDIVAC and unit.energy > 5: hps, rng = 12.6, 4.0
        elif unit.type_id == UnitTypeId.SHIELDBATTERY and unit.energy > 3: hps, rng = 50.4, 6.0
        if hps > 0: self._draw_circle_vectorized(layers, idx, px, py, rng, None, hps, True)

    def _draw_unit_dps(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int, game_data: GameData):
        if unit.type_id in [UnitTypeId.LIBERATORAG, UnitTypeId.CARRIER, UnitTypeId.LURKERMP,
                            UnitTypeId.MEDIVAC, UnitTypeId.WARPPRISM, UnitTypeId.OBSERVER,
                            UnitTypeId.OVERLORD, UnitTypeId.OVERSEER]: return
        if unit.is_hallucination: return

        weapon_stats = []
        uid = unit.type_id
        if uid == UnitTypeId.WIDOWMINEBURROWED:
            weapon_stats = [{'base': 80.0 * max(0.0, 1.0 - (getattr(unit, 'weapon_cooldown', 0) / (29.0 * 22.4))), 'rng': 5.0, 'typ': 3}]
        elif uid == UnitTypeId.BANELING:
            weapon_stats = [{'base': 20.0, 'rng': 2.2, 'typ': 1}]
        elif uid == UnitTypeId.ORACLE and unit.energy >= 25:
            weapon_stats = [{'base': 24.6, 'rng': 4.0, 'typ': 1}]
        else:
            u_name = getattr(unit.type_id, "name", None)
            if u_name:
                db = self._get_unit_data_from_json(u_name)
                if db:
                    for w in db.get('weapons', []):
                        dps = (w['damage'] * w['attacks']) / (w['period'] or 1.0)
                        weapon_stats.append({'base': dps, 'rng': w['range'] + unit.radius, 'typ': w['type']})

        for w in weapon_stats:
            p = "enemy" if is_enemy else "allied"
            idx_g = self.current_layer_map.get(f"{p}_g_dps_base")
            idx_a = self.current_layer_map.get(f"{p}_a_dps_base")
            if w['typ'] in [1, 3] and idx_g is not None:
                self._draw_circle_vectorized(layers, idx_g, px, py, w['rng'], None, w['base'], True)
            if w['typ'] in [2, 3] and idx_a is not None:
                self._draw_circle_vectorized(layers, idx_a, px, py, w['rng'], None, w['base'], True)

    def _draw_effects(self, effects, layers):
        for effect in effects:
            eid = getattr(effect.id, "value", effect.id)
            dps, radius, hits_air, is_ally = 0.0, 0.0, False, False
            if eid in [82, 83]: # Liberator
                dps, radius = 85.0, 5.0
                for pos in effect.positions:
                    dist_a = min([math.hypot(p.x-pos.x, p.y-pos.y) for p in self._liberator_positions['allied']] + [999])
                    dist_e = min([math.hypot(p.x-pos.x, p.y-pos.y) for p in self._liberator_positions['enemy']] + [999])
                    if dist_a < dist_e and dist_a < 16.0: is_ally = True
            elif eid == 1: dps, radius = 28.0, 1.5 # Storm
            elif eid == 11: dps, radius, hits_air = 300.0, 8.0, True # Nuke
            elif eid == 12: dps, radius, hits_air = 200.0, 2.0, True # Bile
            elif eid == 6: # Scan
                idx = self.current_layer_map.get(LayerConstants.ENEMY_DETECTION)
                if idx is not None:
                    for pos in effect.positions:
                        px, py = self._get_pixel_coords(pos)
                        if px is not None: self._draw_circle_vectorized(layers, idx, px, py, 13.0, None, 1.0, False)
                continue
            if dps > 0:
                p = "allied" if is_ally else "enemy"
                idx = self.current_layer_map.get(f"{p}_g_dps_base")
                if idx is not None:
                    for pos in effect.positions:
                        px, py = self._get_pixel_coords(pos)
                        if px is not None:
                            self._draw_circle_vectorized(layers, idx, px, py, radius, None, dps, True)
                            if hits_air:
                                idx_a = self.current_layer_map.get(f"{p}_a_dps_base")
                                if idx_a is not None: self._draw_circle_vectorized(layers, idx_a, px, py, radius, None, dps, True)

    def _draw_transport_info(self, raw_obs, layers):
        units = raw_obs.get('units') or []
        idx_slots = self.current_layer_map.get(LayerConstants.TRANSPORT_SLOTS)
        idx_pass = self.current_layer_map.get(LayerConstants.PASSENGER_DENSITY)
        if idx_slots is None and idx_pass is None: return
        t_ids = {UnitTypeId.MEDIVAC.value, UnitTypeId.OVERLORD.value, UnitTypeId.OVERLORDTRANSPORT.value, UnitTypeId.WARPPRISM.value, UnitTypeId.WARPPRISMPHASING.value, UnitTypeId.BUNKER.value}
        for unit in units:
            u_id = getattr(unit.type_id, "value", unit.type_id)
            if u_id not in t_ids: continue
            px, py = self._get_pixel_coords(unit.position)
            if px is None: continue
            if unit.cargo_max > 0:
                r = int(unit.radius * self.scale) + 1
                if idx_slots: self._draw_circle_vectorized(layers, idx_slots, px, py, 0, r, 0.5 if unit.cargo_used < unit.cargo_max else 1.0, False)
                if idx_pass and unit.cargo_used > 0: self._draw_circle_vectorized(layers, idx_pass, px, py, 0, r, min(float(unit.cargo_used)/8.0, 1.0), True)

    def get_config(self):
        return {"map_size": (self.feature_map_width, self.feature_map_height), "scale": self.scale}
