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
from typing import Dict, Any, List, Optional, Union, Tuple
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

# Units with passive cliff traversal (Colossus: Cliff Walk, Reaper: jet-pack jumps).
# Used by the observation transformer to automatically widen the terrain_pathable layer
# for these unit types — no bot needs to set cliff_walker explicitly in raw_obs.
_CLIFF_WALKER_NAMES: frozenset = frozenset({'COLOSSUS', 'REAPER'})

# Fixed display value for cliff-edge walkable cells in the terrain_pathable layer.
# Shown at the midpoint of [0.2, 1.0] so cliff edges appear as a medium-gray band
# creating a visible gradient: low ground (≈0.2) → cliff edge (0.6) → high ground (≈1.0).
_CLIFF_FACE_DISPLAY_VALUE: float = 0.6

# 4-connected cross structuring element — used for binary erosion when finding
# walkable cells at the walkable boundary (adjacent to any non-walkable cell).
_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

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

        # Terrain caching — keyed by traversal type: "standard" or "cliff_walker"
        self._terrain_caches: Dict[str, Optional[np.ndarray]] = {}
        self._last_map_name = None
        self._last_structure_count = 0
        self._h_min = None
        self._h_max = None
        # Cliff face mask cache — depends only on the static pathing/height grid,
        # so it is valid for the lifetime of a map (recomputed on map change only).
        self._cliff_face_cache: Optional[np.ndarray] = None
        self._cliff_face_cache_map: Optional[str] = None
        # Per-cell display height for cliff face cells: local maximum walkable
        # height in a 5x5 window (= high-ground brightness for each cliff cell).
        self._cliff_display_heights: Optional[np.ndarray] = None

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
            cliff_walker = self._detect_cliff_walker(raw_obs, active_unit=active_unit)
            self._draw_terrain_pathing(game_info, layers, cliff_walker=cliff_walker)
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

            # Normalize all DPS channels with tanh (matches global transform)
            for dps_layer_name in [
                LayerConstants.ALLIED_G_DPS_BASE, LayerConstants.ENEMY_G_DPS_BASE,
                LayerConstants.ALLIED_G_DPS_LIGHT, LayerConstants.ENEMY_G_DPS_LIGHT,
                LayerConstants.ALLIED_G_DPS_ARMOR, LayerConstants.ENEMY_G_DPS_ARMOR,
                LayerConstants.ALLIED_G_DPS_MASSIVE, LayerConstants.ENEMY_G_DPS_MASSIVE,
            ]:
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
        cliff_walker = self._detect_cliff_walker(raw_obs)
        self._draw_terrain_pathing(game_info, layers, current_struct_count, cliff_walker=cliff_walker)
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
        # Health channels are binary damaged indicators; keep them in {0, 1}.
        allied_hp_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_HEALTH]
        enemy_hp_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_HEALTH]
        layers[allied_hp_idx] = (layers[allied_hp_idx] > 0.0).astype(np.float32)
        layers[enemy_hp_idx] = (layers[enemy_hp_idx] > 0.0).astype(np.float32)
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
        cd = float(getattr(unit, "weapon_cooldown", 0.0))
        cd_norm = min(cd, 60.0) / 60.0
        if not is_enemy:
            if self.is_tactical_mode:
                val = 1.0
            else:
                val = 1.0 - (0.7 * cd_norm)

        self._draw_circle_vectorized(layers, idx_den, px, py, 0, r, val, True)

        if self.is_tactical_mode:
            idx_cd = self.current_layer_map.get(
                LayerConstants.ENEMY_WEAPON_COOLDOWN if is_enemy else LayerConstants.ALLIED_WEAPON_COOLDOWN
            )
            if idx_cd is not None:
                self._draw_circle_vectorized(layers, idx_cd, px, py, 0, r, cd_norm, True)

        # Attribute flags — rendered in both global and tactical modes.
        # In tactical mode current_layer_map == TACTICAL_LAYERS; idx is None
        # for any attribute not configured there, so the idx-is-None guard
        # handles missing entries safely without an extra mode check.
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

    def _detect_cliff_walker(self, raw_obs: dict, active_unit=None) -> bool:
        """Return True if the focal unit has passive cliff traversal (Colossus or Reaper).

        Resolution order:
        1. ``active_unit`` parameter (most specific — used by transform_tactical).
        2. First entry of ``raw_obs['selected_units']`` (used by transform global view).
        3. ``raw_obs['cliff_walker']`` explicit override (backward-compat fallback).
        """
        focal = active_unit
        if focal is None:
            selected = raw_obs.get('selected_units') or []
            focal = selected[0] if selected else None
        if focal is not None:
            return getattr(focal.type_id, 'name', '').upper() in _CLIFF_WALKER_NAMES
        return bool(raw_obs.get('cliff_walker', False))


    @staticmethod
    def _compute_climber_grid(path_data, height_data):
        """
        Exact NumPy port of the SC2MapAnalysis C-extension climber_grid algorithm
        (eladyaniv01/SC2MapAnalysis, MapAnalyzer/cext/src/ma_ext.c, get_map_data()).

        Returns a boolean array (same shape as path_data) where True marks non-walkable
        cells that Reaper / Colossus can traverse (exactly one SC2 cliff tier of height
        difference = 16 between the walkable areas on both sides).

        A cell is included only when it is CLIMBABLE and has at least one adjacent
        CLIMBABLE neighbour (connectivity filter removes isolated edge artefacts).
        """
        LEVEL_DIFFERENCE = 16
        h, w = path_data.shape
        walkable = (path_data > 0).astype(np.uint8)
        heights = height_data.astype(np.int32)
        climbable = np.zeros((h, w), dtype=bool)

        # 4 asymmetric directions from the C source:
        #   dirs[8] = { -1, -1,  1, -1,  1, 0,  0, 1 }
        # (xdir, ydir): (-1,-1), (1,-1), (1,0), (0,1)
        # Together these cover all cardinal and diagonal cliff orientations:
        # each bridge (walkable–nonwalkable–walkable with height diff 16) is
        # detectable from one of its walkable-side neighbours.
        for xdir, ydir in [(-1, -1), (1, -1), (1, 0), (0, 1)]:
            y0s, y0e = 2, h - 2
            x0s, x0e = 2, w - 2
            # Keep (x + 2*xdir) and (y + 2*ydir) inside the array.
            if xdir > 0:
                x0e = min(x0e, w - 1 - 2 * xdir)
            elif xdir < 0:
                x0s = max(x0s, -2 * xdir)
            if ydir > 0:
                y0e = min(y0e, h - 1 - 2 * ydir)
            elif ydir < 0:
                y0s = max(y0s, -2 * ydir)
            # Keep (x1+1) and (y1+1) in bounds (2x2 height block around cliff cell).
            if xdir >= 0:
                x0e = min(x0e, w - 2 - xdir)
            if ydir >= 0:
                y0e = min(y0e, h - 2 - ydir)
            if y0s >= y0e or x0s >= x0e:
                continue

            W0 = walkable[y0s:y0e,                 x0s:x0e]
            W1 = walkable[y0s+ydir:y0e+ydir,       x0s+xdir:x0e+xdir]
            W2 = walkable[y0s+2*ydir:y0e+2*ydir,   x0s+2*xdir:x0e+2*xdir]
            base = (W0 == 1) & (W1 == 0) & (W2 == 1)

            # 2x2 height block around cliff-face cell (x1, y1):
            #   H0 = heights[y1+1, x1],  H1 = heights[y1+1, x1+1]
            #   H2 = heights[y1,   x1],  H3 = heights[y1,   x1+1]
            H0 = heights[y0s+ydir+1:y0e+ydir+1, x0s+xdir:x0e+xdir]
            H1 = heights[y0s+ydir+1:y0e+ydir+1, x0s+xdir+1:x0e+xdir+1]
            H2 = heights[y0s+ydir:y0e+ydir,     x0s+xdir:x0e+xdir]
            H3 = heights[y0s+ydir:y0e+ydir,     x0s+xdir+1:x0e+xdir+1]

            LD = LEVEL_DIFFERENCE
            climb = np.zeros(base.shape, dtype=bool)

            if xdir != 0 and ydir != 0:
                if xdir == ydir:    # (-1,-1) — SW diagonal
                    climb |= (((H0 == H1) | (H0 == H2)) & (H2 == H1 + LD) & (H0 == H3))
                    climb |= ((H0 == H1) & (H0 == H3) & (H0 == H2 + LD))
                    climb |= ((H0 == H2) & (H0 == H3) & (H1 == H2 + LD))
                else:               # (1,-1) — SE diagonal
                    climb |= ((H1 == H2) & (H1 == H3) & (H1 == H0 + LD))
                    climb |= ((H0 == H1) & (H0 == H2) & (H3 == H0 + LD))
                    climb |= ((H0 == H1) & (H0 == H2) & (H0 == H3 + LD))
                    climb |= ((H1 == H2) & (H1 == H3) & (H0 == H3 + LD))
            elif xdir != 0:         # (1, 0) — horizontal
                climb |= ((H0 == H2) & (H1 == H3) & (H0 + LD == H1))
                climb |= ((H0 == H2) & (H1 == H3) & (H0 == H1 + LD))
            else:                   # (0, 1) — vertical
                climb |= ((H0 == H1) & (H2 == H3) & (H0 + LD == H2))
                climb |= ((H0 == H1) & (H2 == H3) & (H0 == H2 + LD))

            climbable[y0s+ydir:y0e+ydir, x0s+xdir:x0e+xdir] |= (base & climb)

        # Connectivity filter: only keep cells adjacent to another climbable cell.
        from scipy.ndimage import binary_dilation
        nbr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        return climbable & binary_dilation(climbable, structure=nbr)

    def _draw_terrain_pathing(self, game_info, layers, current_structure_count=0, cliff_walker: bool = False):
        layer_idx = self.current_layer_map.get(LayerConstants.TERRAIN_PATHABLE)
        if layer_idx is None: return

        current_map_name = getattr(game_info, 'map_name', '')
        cache_key = "cliff_walker" if cliff_walker else "standard"

        # Global View Cache (Only if not in tactical mode)
        if not self.is_tactical_mode:
            if (cache_key in self._terrain_caches and
                self._last_map_name == current_map_name and
                self._last_structure_count == current_structure_count):
                layers[layer_idx] = self._terrain_caches[cache_key]
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

            if cliff_walker:
                # Use the exact SC2MapAnalysis climber_grid algorithm (ported from C):
                # non-walkable cells with exactly LEVEL_DIFFERENCE=16 height units between
                # the walkable areas on both sides are marked as cliff-traversable.
                # Results cached per map (static terrain never changes mid-game).
                if self._cliff_face_cache is None or self._cliff_face_cache_map != current_map_name:
                    self._cliff_face_cache = self._compute_climber_grid(path_data, height_data)
                    # For each cliff face cell, display at the mean of the local
                    # low-ground and high-ground heights (5x5 window on each side).
                    # This places cliff cells visually between low and high ground,
                    # creating a clear gradient: dark (low) → mid (cliff) → bright (high).
                    from scipy.ndimage import maximum_filter, minimum_filter
                    h_f = height_data.astype(np.float32)
                    large = float(h_f.max()) + 1.0
                    local_high_h = maximum_filter(
                        np.where(path_data > 0, h_f, 0.0), size=5)
                    local_low_h = minimum_filter(
                        np.where(path_data > 0, h_f, large), size=5)
                    cliff_mean_h = (local_high_h + local_low_h) * 0.5
                    h_rng = max(float(self._h_max) - float(self._h_min), 1.0)
                    self._cliff_display_heights = np.clip(
                        0.2 + (cliff_mean_h - float(self._h_min)) / h_rng * 0.8,
                        0.2, 1.0
                    ).astype(np.float32)
                    self._cliff_face_cache_map = current_map_name

                sampled_cliff = self._cliff_face_cache[v_y, v_x]
                cliff_brightness = self._cliff_display_heights[v_y, v_x]
                standard_terrain = norm_height * (sampled_path > 0).astype(np.float32)
                res[valid_mask] = np.where(sampled_cliff, cliff_brightness, standard_terrain)
            else:
                res[valid_mask] = norm_height * (sampled_path > 0).astype(np.float32)

            layers[layer_idx] = res

            if not self.is_tactical_mode:
                self._terrain_caches[cache_key] = res.copy()
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
            hp_ratio = (unit.health + unit.shield) / max_hp
            # Binary "damaged" signal:
            # - 0.0 when unit is effectively full health (>=95%)
            # - 1.0 when unit dropped below 95%
            hp_signal = 1.0 if hp_ratio < 0.95 else 0.0
            self._draw_circle_vectorized(layers, idx_hp, px, py, 0, 0, hp_signal, True)
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

    def _dps_weapon_game_radius(self, unit: Unit, catalog_range: float) -> Tuple[float, float]:
        """
        Weapon reach in game units for DPS discs: max(catalog range, live API range) + radius.

        ``catalog_range`` is the base weapon range from ``unit_data.json`` (no radius). The SC2
        observation fills ``unit.ground_range`` / ``unit.air_range`` with **current** values, so
        upgrades (e.g. Extended Thermal Lance, Hi-Sec Auto Tracking, Grooved Spines, Seismic
        Spines, Advanced Ballistics, Anion Crystals) match what allies and **enemies** can do.

        Returns (ground_ring_radius, air_ring_radius) in game units (same convention as before:
        catalog range + unit.radius, but catalog is upgraded by live stats when higher).
        """
        ur = float(getattr(unit, "radius", 0.0) or 0.0)
        base = float(catalog_range)
        g_live = float(getattr(unit, "ground_range", 0.0) or 0.0)
        a_live = float(getattr(unit, "air_range", 0.0) or 0.0)
        rng_g = max(base, g_live) + ur
        rng_a = max(base, a_live) + ur
        return rng_g, rng_a

    def _draw_unit_dps(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int, game_data: GameData):
        if unit.type_id in [UnitTypeId.LIBERATORAG, UnitTypeId.CARRIER, UnitTypeId.LURKERMP,
                            UnitTypeId.MEDIVAC, UnitTypeId.WARPPRISM, UnitTypeId.OBSERVER,
                            UnitTypeId.OVERLORD, UnitTypeId.OVERSEER]: return
        if unit.is_hallucination: return

        weapon_stats = []
        uid = unit.type_id
        if uid == UnitTypeId.WIDOWMINEBURROWED:
            weapon_stats = [
                {
                    'base': 80.0 * max(0.0, 1.0 - (getattr(unit, 'weapon_cooldown', 0) / (29.0 * 22.4))),
                    'range': 5.0,
                    'typ': 3,
                }
            ]
        elif uid == UnitTypeId.BANELING:
            # Explosion radius — not driven by unit.ground_range; keep legacy fixed draw size.
            weapon_stats = [{'base': 20.0, 'rng': 2.2, 'typ': 1}]
        elif uid == UnitTypeId.ORACLE and unit.energy >= 25:
            weapon_stats = [{'base': 24.6, 'range': 4.0, 'typ': 1}]
        else:
            u_name = getattr(unit.type_id, "name", None)
            if u_name:
                db = self._get_unit_data_from_json(u_name)
                if db:
                    for w in db.get('weapons', []):
                        period = w.get('period') or 1.0
                        dps = (w['damage'] * w['attacks']) / period
                        weapon_stats.append({
                            'base': dps,
                            'range': float(w['range']),
                            'typ': w['type'],
                            'period': period,
                            'bonuses': w.get('bonuses', []),
                        })

        # Weapon bonus type → typed-DPS layer suffix.
        # Values confirmed from unit_data.json samples:
        #   1=Light (Colossus, Hellion, Baneling), 2=Armored (Marauder, SiegeTank),
        #   7=Massive (Corruptor, Tempest).  Type 3=Biological has no typed layer.
        _BONUS_TYPE_SUFFIX = {1: "light", 2: "armor", 7: "massive"}

        for w in weapon_stats:
            p = "enemy" if is_enemy else "allied"
            idx_g = self.current_layer_map.get(f"{p}_g_dps_base")
            idx_a = self.current_layer_map.get(f"{p}_a_dps_base")
            if 'rng' in w and 'range' not in w:
                rng_g = float(w['rng'])
                rng_a = rng_g
            else:
                rng_g, rng_a = self._dps_weapon_game_radius(unit, float(w['range']))
            typ = int(w['typ'])
            if typ in (1, 3) and idx_g is not None:
                self._draw_circle_vectorized(layers, idx_g, px, py, rng_g, None, w['base'], True)
            if typ in (2, 3) and idx_a is not None:
                self._draw_circle_vectorized(layers, idx_a, px, py, rng_a, None, w['base'], True)

            # Typed DPS bonus channels — bonus damage per target attribute.
            # Each entry is the extra DPS dealt only against targets with that attribute;
            # the base DPS is already covered by the _dps_base channels above.
            period = w.get('period', 1.0) or 1.0
            for bonus in w.get('bonuses', []):
                suffix = _BONUS_TYPE_SUFFIX.get(bonus.get('type'))
                if suffix is None:
                    continue
                bonus_dps = bonus.get('bonus', 0.0) / period
                if typ in (1, 3):
                    idx_gb = self.current_layer_map.get(f"{p}_g_dps_{suffix}")
                    if idx_gb is not None:
                        self._draw_circle_vectorized(layers, idx_gb, px, py, rng_g, None, bonus_dps, True)
                if typ in (2, 3):
                    idx_ab = self.current_layer_map.get(f"{p}_a_dps_{suffix}")
                    if idx_ab is not None:
                        self._draw_circle_vectorized(layers, idx_ab, px, py, rng_a, None, bonus_dps, True)

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
