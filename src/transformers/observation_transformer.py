"""
Professional Observation Transformer
Fixed:
1. Viewport now crops to 'playable_area' instead of full 'map_size' to remove black borders.
2. Coordinate transforms updated to handle crop offsets (crop_x, crop_y).
3. Preserved all previous logic (Effect/Unit rendering).
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

# Project Constants
from src.utils.layer_constants import LayerConstants


class ObservationTransformer:
    def __init__(self, map_size: tuple[int, int] = (128, 128), unit_data_file: str = None):
        self.feature_map_width = map_size[0]
        self.feature_map_height = map_size[1]

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.crop_x = 0.0
        self.crop_y = 0.0

        self._terrain_cache = None
        self._last_map_name = None

        self._grid_y = np.arange(self.feature_map_height).reshape(-1, 1)
        self._grid_x = np.arange(self.feature_map_width).reshape(1, -1)

        self._liberator_positions = {'allied': [], 'enemy': []}

        if unit_data_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            unit_data_file = os.path.join(base_dir, "data", "unit_data.json")

        self.fallback_db = self._load_fallback_db(unit_data_file)

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

    def transform(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        total_layers = LayerConstants.get_num_layers()
        layers = np.zeros((total_layers, self.feature_map_height, self.feature_map_width), dtype=np.float32)

        allied_units = raw_obs.get('units', [])
        enemy_units = raw_obs.get('enemy_units', [])
        effects = raw_obs.get('effects', [])
        game_info = raw_obs.get('game_info')
        game_data = raw_obs.get('game_data')

        # 1. Calculate Scale & Crop based on Playable Area
        if game_info:
            # Prefer Playable Area (Camera Bounds) to remove void borders
            if hasattr(game_info, 'playable_area'):
                area = game_info.playable_area
                self.crop_x = area.x
                self.crop_y = area.y
                mw = area.width
                mh = area.height
            elif hasattr(game_info, 'map_size'):
                self.crop_x = 0
                self.crop_y = 0
                mw = game_info.map_size.width
                mh = game_info.map_size.height
            else:
                self.crop_x, self.crop_y, mw, mh = 0, 0, 200, 200
        else:
            self.crop_x, self.crop_y, mw, mh = 0, 0, 200, 200

        # Scale to fit the playable area within feature map
        self.scale = min(self.feature_map_width / max(mw, 1), self.feature_map_height / max(mh, 1))
        self.scale = min(self.scale, 1.0)

        scaled_w = int(mw * self.scale)
        scaled_h = int(mh * self.scale)
        self.offset_x = max(0, (self.feature_map_width - scaled_w) // 2)
        self.offset_y = max(0, (self.feature_map_height - scaled_h) // 2)

        # 2. Draw Environment
        self._draw_terrain_pathing(game_info, layers)
        self._draw_terrain_creep(raw_obs, layers)

        # 3. Draw Units
        self._liberator_positions = {'allied': [], 'enemy': []}
        for unit in allied_units:
            type_name = getattr(unit.type_id, "name", None)
            if type_name and "MINERAL" in type_name.upper(): continue

            self._process_single_unit(unit, layers, False, mw, mh, game_data)
            if unit.type_id == UnitTypeId.LIBERATORAG:
                self._liberator_positions['allied'].append(unit.position)

        for unit in enemy_units:
            type_name = getattr(unit.type_id, "name", None)
            if type_name and "MINERAL" in type_name.upper(): continue

            self._process_single_unit(unit, layers, True, mw, mh, game_data)
            if unit.type_id == UnitTypeId.LIBERATORAG:
                self._liberator_positions['enemy'].append(unit.position)

        # 4. Draw Effects
        if effects:
            self._draw_effects(effects, layers)

        # 5. Normalize
        dps_start = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]
        hps_end = LayerConstants.LAYERS[LayerConstants.ENEMY_HPS_FIELD] + 1
        layers[dps_start:hps_end] = np.tanh(layers[dps_start:hps_end] / 50.0)
        layers[0:dps_start] = np.log1p(layers[0:dps_start])

        return layers

    def _get_pixel_coords(self, position) -> tuple:
        if position is None:
            return None, None

        # Adjust for Crop Offset (Playable Area)
        rel_x = position.x - self.crop_x
        rel_y = position.y - self.crop_y

        px = int(rel_x * self.scale + self.offset_x)
        py = int(self.feature_map_height - 1 - (rel_y * self.scale + self.offset_y))

        if 0 <= px < self.feature_map_width and 0 <= py < self.feature_map_height:
            return px, py
        return None, None

    def _draw_terrain_pathing(self, game_info, layers):
        layer_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
        current_map_name = getattr(game_info, 'map_name', '')

        if self._terrain_cache is not None and self._last_map_name == current_map_name:
            layers[layer_idx] = self._terrain_cache
            return

        if not hasattr(game_info, 'pathing_grid') or not hasattr(game_info, 'terrain_height'):
            return

        path_data = self._get_numpy_data(game_info.pathing_grid)
        height_data = self._get_numpy_data(game_info.terrain_height)

        if path_data is None or path_data.size == 0:
            return

        # Raw grid dimensions (Total Map)
        dim0, dim1 = path_data.shape
        total_w = game_info.map_size.width
        total_h = game_info.map_size.height

        x_indices = np.arange(self.feature_map_width)
        y_indices = np.arange(self.feature_map_height)

        # Inverse Transform: Feature Map -> SC2 World
        # sc2_x = ((px - offset) / scale) + crop_x
        sc2_x = ((x_indices - self.offset_x) / self.scale) + self.crop_x
        sc2_x = sc2_x.astype(int)

        sc2_y = ((self.feature_map_height - 1 - y_indices) - self.offset_y) / self.scale
        sc2_y = (sc2_y + self.crop_y).astype(int)

        X_map, Y_map = np.meshgrid(sc2_x, sc2_y)

        # Mask: Check against TOTAL map bounds
        valid_mask = (X_map >= 0) & (X_map < total_w) & (Y_map >= 0) & (Y_map < total_h)

        valid_sc2_x = X_map[valid_mask]
        valid_sc2_y = Y_map[valid_mask]

        sampled_path = None
        sampled_height = None

        if dim0 == total_w and dim1 == total_h:
            if np.all(valid_sc2_x < dim0) and np.all(valid_sc2_y < dim1):
                sampled_path = path_data[valid_sc2_x, valid_sc2_y]
                sampled_height = height_data[valid_sc2_x, valid_sc2_y]
        elif dim0 == total_h and dim1 == total_w:
            if np.all(valid_sc2_y < dim0) and np.all(valid_sc2_x < dim1):
                sampled_path = path_data[valid_sc2_y, valid_sc2_x]
                sampled_height = height_data[valid_sc2_y, valid_sc2_x]

        if sampled_path is None:
            cx = np.clip(valid_sc2_x, 0, dim0 - 1)
            cy = np.clip(valid_sc2_y, 0, dim1 - 1)
            sampled_path = path_data[cx, cy]
            sampled_height = height_data[cx, cy]

        height_layer = np.zeros((self.feature_map_height, self.feature_map_width), dtype=np.float32)
        valid_pathable = sampled_path != 0

        if np.any(valid_pathable):
             h_vals = sampled_height[valid_pathable]
             if h_vals.size > 0:
                 min_h, max_h = h_vals.min(), h_vals.max()
                 h_range = max(max_h - min_h, 1.0)
                 normalized_all = 0.2 + ((sampled_height - min_h) / h_range) * 0.8
                 height_layer[valid_mask] = normalized_all * (sampled_path > 0).astype(np.float32)

        layers[layer_idx] = height_layer
        self._terrain_cache = layers[layer_idx].copy()
        self._last_map_name = current_map_name

    def _draw_terrain_creep(self, raw_obs, layers):
        layer_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_CREEP]
        creep_map = raw_obs.get('creep')
        if not creep_map: return

        creep_data = self._get_numpy_data(creep_map)
        if creep_data is None or creep_data.size == 0:
            return

        dim0, dim1 = creep_data.shape
        game_info = raw_obs.get('game_info')
        total_w, total_h = 200, 200
        if game_info and hasattr(game_info, 'map_size'):
            total_w = game_info.map_size.width
            total_h = game_info.map_size.height

        x_indices = np.arange(self.feature_map_width)
        y_indices = np.arange(self.feature_map_height)

        sc2_x = ((x_indices - self.offset_x) / self.scale) + self.crop_x
        sc2_x = sc2_x.astype(int)

        sc2_y = ((self.feature_map_height - 1 - y_indices) - self.offset_y) / self.scale
        sc2_y = (sc2_y + self.crop_y).astype(int)

        X_map, Y_map = np.meshgrid(sc2_x, sc2_y)
        valid_mask = (X_map >= 0) & (X_map < total_w) & (Y_map >= 0) & (Y_map < total_h)

        if not np.any(valid_mask): return

        valid_sc2_x = X_map[valid_mask]
        valid_sc2_y = Y_map[valid_mask]
        sampled_creep = None

        if dim0 == total_w and dim1 == total_h:
            if np.all(valid_sc2_x < dim0) and np.all(valid_sc2_y < dim1):
                sampled_creep = creep_data[valid_sc2_x, valid_sc2_y]
        elif dim0 == total_h and dim1 == total_w:
            if np.all(valid_sc2_y < dim0) and np.all(valid_sc2_x < dim1):
                sampled_creep = creep_data[valid_sc2_y, valid_sc2_x]

        if sampled_creep is None:
            cx = np.clip(valid_sc2_x, 0, dim0 - 1)
            cy = np.clip(valid_sc2_y, 0, dim1 - 1)
            if dim0 >= total_w:
                 sampled_creep = creep_data[cx, cy]
            else:
                 sampled_creep = creep_data[cy, cx]

        if sampled_creep is not None:
            layers[layer_idx][valid_mask] = sampled_creep

    def _process_single_unit(self, unit: Unit, layers: np.ndarray, is_enemy: bool,
                             map_w: int, map_h: int, game_data: GameData):
        px, py = self._get_pixel_coords(unit.position)
        if px is None: return

        if is_enemy and unit.is_cloaked and not unit.is_detected:
            is_snapshot = unit.is_snapshot if hasattr(unit, "is_snapshot") else False
            if not is_snapshot: return

        # Scan
        scan_enum_val = getattr(UnitTypeId, 'SCANNERSWEEP', 170)
        u_id = unit.type_id.value if hasattr(unit.type_id, "value") else unit.type_id
        expected_val = scan_enum_val.value if hasattr(scan_enum_val, "value") else scan_enum_val

        if u_id == expected_val or u_id == 170:
            prefix = "enemy" if is_enemy else "allied"
            idx_det = LayerConstants.LAYERS[f"{prefix}_detection"]
            self._draw_circle_vectorized(layers, idx_det, px, py, 13.0, None, 1.0)
            return

        # Force Field
        ff_id = getattr(UnitTypeId, 'FORCEFIELD', 135)
        ff_val = ff_id.value if hasattr(ff_id, "value") else 135
        if u_id == ff_val or u_id == 135:
            idx_path = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
            self._draw_circle_vectorized(layers, idx_path, px, py, 1.5, None, 0.0, accumulate=False)
            prefix = "enemy" if is_enemy else "allied"
            idx_den = LayerConstants.LAYERS[f"{prefix}_unit_density"]
            self._draw_circle_vectorized(layers, idx_den, px, py, 0, 0, 1.0, accumulate=True)
            return

        # Sensor Blip
        if hasattr(unit, "is_blip") and unit.is_blip:
            if is_enemy:
                idx_den = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]
                self._draw_circle_vectorized(layers, idx_den, px, py, 0, 0, 0.5, True)
            return

        self._draw_unit_body(unit, layers, is_enemy, px, py)
        self._draw_unit_vitals(unit, layers, is_enemy, px, py)
        self._draw_unit_detection(unit, layers, is_enemy, px, py)
        self._draw_unit_healing(unit, layers, is_enemy, px, py)
        self._draw_unit_dps(unit, layers, is_enemy, px, py, game_data)

    def _draw_unit_body(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int):
        prefix = "enemy" if is_enemy else "allied"
        idx_den = LayerConstants.LAYERS[f"{prefix}_unit_density"]

        r = 0
        if unit.is_structure:
            r = int(unit.radius * self.scale)

        val = 1.0
        if not is_enemy and not unit.is_structure:
            cd = getattr(unit, 'weapon_cooldown', 0.0)
            norm_cd = min(cd, 60.0) / 60.0
            val = 1.0 - (0.7 * norm_cd)

        self._draw_circle_vectorized(layers, idx_den, px, py, 0, r, val, True)

        if unit.is_biological:
            self._draw_circle_vectorized(layers, LayerConstants.LAYERS[f"{prefix}_unit_biological"], px, py, 0, r, 1.0, True)
        if unit.is_mechanical:
            self._draw_circle_vectorized(layers, LayerConstants.LAYERS[f"{prefix}_unit_mechanical"], px, py, 0, r, 1.0, True)
        if unit.is_armored:
            self._draw_circle_vectorized(layers, LayerConstants.LAYERS[f"{prefix}_unit_armored"], px, py, 0, r, 1.0, True)
        if unit.is_massive:
            self._draw_circle_vectorized(layers, LayerConstants.LAYERS[f"{prefix}_unit_massive"], px, py, 0, r, 1.0, True)
        if unit.is_flying or unit.type_id == UnitTypeId.COLOSSUS:
            self._draw_circle_vectorized(layers, LayerConstants.LAYERS[f"{prefix}_unit_flying"], px, py, 0, r, 1.0, True)

        idx_clk = LayerConstants.LAYERS[f"{prefix}_unit_cloak"]
        if unit.is_cloaked:
            self._draw_circle_vectorized(layers, idx_clk, px, py, 0, r, 1.0, True)
        elif unit.type_id in [UnitTypeId.GHOST, UnitTypeId.BANSHEE, UnitTypeId.DARKTEMPLAR]:
            self._draw_circle_vectorized(layers, idx_clk, px, py, 0, r, 0.5, True)

    def _draw_unit_vitals(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int):
        prefix = "enemy" if is_enemy else "allied"

        r = 0
        if unit.is_structure:
            r = int(unit.radius * self.scale)

        idx_hp = LayerConstants.LAYERS[f"{prefix}_health"]
        max_hp = (unit.health_max + unit.shield_max) or 1
        current_hp = unit.health + unit.shield
        self._draw_circle_vectorized(layers, idx_hp, px, py, 0, r, current_hp / max_hp, True)

        if unit.energy_max > 0:
            idx_nrg = LayerConstants.LAYERS[f"{prefix}_energy"]
            self._draw_circle_vectorized(layers, idx_nrg, px, py, 0, r, unit.energy / unit.energy_max, True)

    def _draw_unit_detection(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int):
        is_ready = unit.is_ready if hasattr(unit, "is_ready") else (unit.build_progress == 1.0)
        if not is_ready: return
        prefix = "enemy" if is_enemy else "allied"
        idx_det = LayerConstants.LAYERS[f"{prefix}_detection"]
        radius = unit.detect_range
        if radius > 0:
            self._draw_circle_vectorized(layers, idx_det, px, py, radius, None, 1.0, False)

    def _draw_unit_healing(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int):
        prefix = "enemy" if is_enemy else "allied"
        layer_idx = LayerConstants.LAYERS[f"{prefix}_hps_field"]
        hps = 0.0
        rng = 0.0
        is_ready = unit.is_ready if hasattr(unit, "is_ready") else (unit.build_progress == 1.0)
        if is_ready:
            uid = unit.type_id
            if uid == UnitTypeId.MEDIVAC and unit.energy > 5:
                hps, rng = 12.6, 4.0
            elif uid == UnitTypeId.SHIELDBATTERY and unit.energy > 3:
                hps, rng = 50.4, 6.0
            elif uid == UnitTypeId.QUEEN and unit.energy >= 50:
                hps, rng = 25.0, 7.0
        if hps > 0:
            self._draw_circle_vectorized(layers, layer_idx, px, py, rng, None, hps, accumulate=True)

    def _draw_unit_dps(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int, game_data: GameData):
        if unit.type_id in [UnitTypeId.LIBERATORAG, UnitTypeId.CARRIER, UnitTypeId.LURKERMP,
                            UnitTypeId.MEDIVAC, UnitTypeId.WARPPRISM, UnitTypeId.OBSERVER,
                            UnitTypeId.OVERLORD, UnitTypeId.OVERSEER]:
            return
        if unit.is_hallucination: return
        is_ready = unit.is_ready if hasattr(unit, "is_ready") else (unit.build_progress == 1.0)
        if not is_ready: return

        weapon_stats = []
        uid = unit.type_id
        is_override = False

        if uid == UnitTypeId.WIDOWMINEBURROWED:
            is_override = True
            max_cd = 29.0 * 22.4
            cd = getattr(unit, 'weapon_cooldown', 0)
            ratio = max(0.0, 1.0 - (cd / max_cd))
            weapon_stats = [{'base': 80.0 * ratio, 'b_l': 0, 'b_a': 0, 'b_m': 0, 'rng': 5.0, 'typ': 3}]
        elif uid == UnitTypeId.BANELING:
            is_override = True
            weapon_stats = [{'base': 20.0, 'b_l': 15.0, 'b_a': 0, 'b_m': 0, 'rng': 2.2, 'typ': 1}]
        elif uid == UnitTypeId.ORACLE:
            is_override = True
            if unit.energy >= 25:
                weapon_stats = [{'base': 24.6, 'b_l': 16.4, 'b_a': 0, 'b_m': 0, 'rng': 4.0, 'typ': 1}]
        elif uid == UnitTypeId.TEMPEST:
            is_override = True
            upg = getattr(unit, "attack_upgrade_level", 0)
            d_a = (30 + 3 * upg) / 2.36
            d_g = (40 + 4 * upg) / 2.36
            weapon_stats = [
                {'base': d_a, 'b_l': 0, 'b_a': 0, 'b_m': (22 / 2.36), 'rng': 14.0, 'typ': 2},
                {'base': d_g, 'b_l': 0, 'b_a': 0, 'b_m': 0, 'rng': 10.0, 'typ': 1}
            ]

        if not is_override:
            u_name = getattr(unit.type_id, "name", None)
            if u_name:
                db = self._get_unit_data_from_json(u_name)
                if db:
                    upg = getattr(unit, "attack_upgrade_level", 0)
                    for w in db.get('weapons', []):
                        base = w['damage'] + (w.get('upgrade_bonus', 0) * upg)
                        period = w['period'] or 1.0
                        dps = (base * w['attacks']) / period
                        b_l, b_a, b_m = 0.0, 0.0, 0.0
                        for b in w.get('bonuses', []):
                            val = (b['bonus'] * w['attacks']) / period
                            if b['type'] == 1:
                                b_l += val
                            elif b['type'] == 2:
                                b_a += val
                            elif b['type'] == 7:
                                b_m += val
                        weapon_stats.append({
                            'base': dps, 'b_l': b_l, 'b_a': b_a, 'b_m': b_m,
                            'rng': w['range'] + unit.radius, 'typ': w['type']
                        })

        for w in weapon_stats:
            self._draw_dps_circle(layers, is_enemy, px, py, w['rng'], w)

    def _draw_dps_circle(self, layers, is_enemy, cx, cy, radius_game, w_stats):
        if not is_enemy:
            idx_g = [LayerConstants.LAYERS[k] for k in [
                LayerConstants.ALLIED_G_DPS_BASE, LayerConstants.ALLIED_G_DPS_LIGHT,
                LayerConstants.ALLIED_G_DPS_ARMOR, LayerConstants.ALLIED_G_DPS_MASSIVE]]
            idx_a = [LayerConstants.LAYERS[k] for k in [
                LayerConstants.ALLIED_A_DPS_BASE, LayerConstants.ALLIED_A_DPS_LIGHT,
                LayerConstants.ALLIED_A_DPS_ARMOR, LayerConstants.ALLIED_A_DPS_MASSIVE]]
        else:
            idx_g = [LayerConstants.LAYERS[k] for k in [
                LayerConstants.ENEMY_G_DPS_BASE, LayerConstants.ENEMY_G_DPS_LIGHT,
                LayerConstants.ENEMY_G_DPS_ARMOR, LayerConstants.ENEMY_G_DPS_MASSIVE]]
            idx_a = [LayerConstants.LAYERS[k] for k in [
                LayerConstants.ENEMY_A_DPS_BASE, LayerConstants.ENEMY_A_DPS_LIGHT,
                LayerConstants.ENEMY_A_DPS_ARMOR, LayerConstants.ENEMY_A_DPS_MASSIVE]]

        vals = [w_stats['base'], w_stats['b_l'], w_stats['b_a'], w_stats['b_m']]
        typ = w_stats['typ']

        if typ == 1 or typ == 3:
            for i in range(4):
                if vals[i] > 0:
                    self._draw_circle_vectorized(layers, idx_g[i], cx, cy, radius_game, None, vals[i], True)

        if typ == 2 or typ == 3:
            for i in range(4):
                if vals[i] > 0:
                    self._draw_circle_vectorized(layers, idx_a[i], cx, cy, radius_game, None, vals[i], True)

    def _draw_effects(self, effects, layers):
        for effect in effects:
            eid = effect.id
            dps, radius = 0.0, 0.0
            is_ally = False
            hits_air = False

            lib_delay = getattr(EffectId, 'LIBERATORTARGETMORPHDELAYPERSISTENT', 82)
            lib_morph = getattr(EffectId, 'LIBERATORTARGETMORPHPERSISTENT', 83)
            psi_storm = getattr(EffectId, 'PSISTORMPERSISTENT', 1)
            nuke = getattr(EffectId, 'NUKEPERSISTENT', 11)
            bile = getattr(EffectId, 'RAVAGERCORROSIVEBILECP', 12)
            scan = getattr(EffectId, 'SCANNERSWEEP', 6)

            if eid in [lib_delay, lib_morph]:
                dps, radius = 85.0, 5.0
                for pos in effect.positions:
                    p_x, p_y = pos.x, pos.y
                    min_ally = 999.0
                    for p in self._liberator_positions['allied']:
                        dist = math.hypot(p.x - p_x, p.y - p_y)
                        if dist < min_ally: min_ally = dist

                    min_enemy = 999.0
                    for p in self._liberator_positions['enemy']:
                        dist = math.hypot(p.x - p_x, p.y - p_y)
                        if dist < min_enemy: min_enemy = dist

                    if min_ally < min_enemy and min_ally < 16.0:
                        is_ally = True

            elif eid == psi_storm:
                dps, radius = 28.0, 1.5
            elif eid == nuke:
                dps, radius = 300.0, 8.0
                hits_air = True
            elif eid == bile:
                dps, radius = 200.0, 2.0
                hits_air = True
            elif eid == scan:
                dps, radius = 0.0, 13.0
                idx = LayerConstants.LAYERS[LayerConstants.ENEMY_DETECTION]
                for pos in effect.positions:
                    px, py = self._get_pixel_coords(pos)
                    if px is None: continue
                    self._draw_circle_vectorized(layers, idx, px, py, radius, None, 1.0, False)
                continue
            else:
                continue

            if dps > 0:
                target_layer = LayerConstants.ALLIED_G_DPS_BASE if is_ally else LayerConstants.ENEMY_G_DPS_BASE
                idx = LayerConstants.LAYERS[target_layer]

                idx_air = None
                if hits_air:
                    air_target = LayerConstants.ALLIED_A_DPS_BASE if is_ally else LayerConstants.ENEMY_A_DPS_BASE
                    idx_air = LayerConstants.LAYERS[air_target]

                for pos in effect.positions:
                    px, py = self._get_pixel_coords(pos)
                    if px is None: continue
                    self._draw_circle_vectorized(layers, idx, px, py, radius, None, dps, True)
                    if idx_air is not None:
                        self._draw_circle_vectorized(layers, idx_air, px, py, radius, None, dps, True)

    def _draw_circle_vectorized(self, layers, layer_idx, cx, cy, radius_game, radius_px_override=None, value=1.0,
                                accumulate=False):
        if radius_px_override is not None:
            r = int(radius_px_override)
        else:
            r = max(1, int(radius_game * self.scale))

        if r == 0:
            if accumulate:
                layers[layer_idx, cy, cx] += value
            else:
                layers[layer_idx, cy, cx] = value
            return

        r_sq = r ** 2
        x_min = max(0, cx - r)
        x_max = min(self.feature_map_width, cx + r + 1)
        y_min = max(0, cy - r)
        y_max = min(self.feature_map_height, cy + r + 1)

        if x_min >= x_max or y_min >= y_max: return

        Y = self._grid_y[y_min:y_max, :]
        X = self._grid_x[:, x_min:x_max]
        mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r_sq

        if accumulate:
            layers[layer_idx, y_min:y_max, x_min:x_max][mask] += value
        else:
            layers[layer_idx, y_min:y_max, x_min:x_max][mask] = value

    def get_config(self):
        return {
            "map_size": (self.feature_map_width, self.feature_map_height),
            "scale": self.scale,
            "offset": (self.offset_x, self.offset_y),
            "crop": (self.crop_x, self.crop_y)
        }