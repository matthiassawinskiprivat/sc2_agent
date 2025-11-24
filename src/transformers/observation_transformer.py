"""
Professional Observation Transformer
Transforms SC2 data into 40-Layer High-Fidelity Feature Maps
Refactored for Readability and Debugging
"""

import numpy as np
import json
import os
import math
from typing import Dict, Any, List, Optional

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

        # Caches
        self._terrain_cache = None
        self._last_map_name = None
        self._liberator_positions = {'allied': [], 'enemy': []}

        # Load Database
        if unit_data_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            unit_data_file = os.path.join(base_dir, "data", "unit_data.json")

        self.fallback_db = self._load_fallback_db(unit_data_file)

    def _load_fallback_db(self, path: str) -> Dict:
        if not os.path.exists(path): return {}
        try:
            with open(path, 'r') as f: return json.load(f)
        except Exception: return {}

    def _get_unit_data_from_json(self, unit_name: str) -> Optional[Dict[str, Any]]:
        return self.fallback_db.get(unit_name.upper())

    def transform(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """Main entry point for observation transformation."""

        # 1. Initialize clean tensor
        total_layers = 40
        layers = np.zeros((total_layers, self.feature_map_height, self.feature_map_width), dtype=np.float32)

        # 2. Extract Raw Data
        allied_units = raw_obs.get('units', [])
        enemy_units = raw_obs.get('enemy_units', [])
        effects = raw_obs.get('effects', [])
        game_info = raw_obs.get('game_info')
        game_data = raw_obs.get('game_data')

        # 3. Determine Map Scale Factors
        # SC2 Map Dimensions (World Units)
        sc2_map_width = 200
        sc2_map_height = 200
        if game_info and hasattr(game_info, 'map_size'):
            sc2_map_width = game_info.map_size.width
            sc2_map_height = game_info.map_size.height

        # 4. Process Environment
        self._draw_terrain_pathing(game_info, layers)
        self._draw_terrain_creep(raw_obs, layers)

        # 5. Reset Caches
        self._liberator_positions = {'allied': [], 'enemy': []}

        # 6. Process Units (Split into sub-tasks for readability)
        for unit in allied_units:
            self._process_single_unit(unit, layers, False, sc2_map_width, sc2_map_height, game_data)
            # Cache for Effect matching
            if unit.type_id == UnitTypeId.LIBERATORAG:
                self._liberator_positions['allied'].append(unit.position)

        for unit in enemy_units:
            self._process_single_unit(unit, layers, True, sc2_map_width, sc2_map_height, game_data)
            if unit.type_id == UnitTypeId.LIBERATORAG:
                self._liberator_positions['enemy'].append(unit.position)

        # 7. Process Effects
        if effects:
            self._draw_effects(effects, layers, sc2_map_width, sc2_map_height)

        # 8. Normalize DPS/HPS Layers (Indices 16 to 33)
        # Apply tanh to squash high damage values into [0, 1] range
        # 50 DPS -> 0.76, 150 DPS -> 0.99
        layers[16:34] = np.tanh(layers[16:34] / 50.0)

        return layers

    # ==============================================================================
    # ENVIRONMENT HANDLERS
    # ==============================================================================

    def _draw_terrain_pathing(self, game_info, layers):
        """Draws the static terrain height map masked by walkability."""
        layer_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]

        # Use Cache if possible
        if self._terrain_cache is not None and self._last_map_name == game_info.map_name:
            layers[layer_idx] = self._terrain_cache
            return

        if not hasattr(game_info, 'terrain_height') or not hasattr(game_info, 'pathing_grid'):
            return

        height_map = game_info.terrain_height
        path_map = game_info.pathing_grid

        # Scaling Factors
        scale_x = path_map.width / self.feature_map_width
        scale_y = path_map.height / self.feature_map_height

        # Temporary buffer to find min/max height for normalization
        raw_heights = np.full((self.feature_map_height, self.feature_map_width), -1.0, dtype=np.float32)
        valid_height_values = []

        # Iterate over FEATURE MAP pixels (Target)
        for y in range(self.feature_map_height):
            for x in range(self.feature_map_width):
                # Map to SC2 coordinates
                # Flip Y: Image(0) is Top, SC2(Max) is Top
                sc2_x = int(x * scale_x)
                sc2_y = int((self.feature_map_height - 1 - y) * scale_y)

                if sc2_x >= path_map.width or sc2_y >= path_map.height:
                    continue

                # Check Walkability
                is_walkable = False
                try:
                    if hasattr(path_map, 'is_pathable'):
                        is_walkable = path_map.is_pathable((sc2_x, sc2_y))
                    else:
                        is_walkable = path_map[(sc2_x, sc2_y)] != 0
                except:
                    pass

                if is_walkable:
                    h = height_map[(sc2_x, sc2_y)]
                    raw_heights[y, x] = h
                    valid_height_values.append(h)

        # Normalize and Write
        if valid_height_values:
            min_h = min(valid_height_values)
            max_h = max(valid_height_values)
            height_range = max(1.0, max_h - min_h)

            for y in range(self.feature_map_height):
                for x in range(self.feature_map_width):
                    raw_val = raw_heights[y, x]
                    if raw_val != -1.0:
                        # Normalize to 0.0 - 1.0
                        normalized = (raw_val - min_h) / height_range
                        # Remap to 0.2 - 1.0 (Reserve 0.0 for Walls)
                        layers[layer_idx, y, x] = 0.2 + (normalized * 0.8)

        # Save Cache
        self._terrain_cache = layers[layer_idx].copy()
        self._last_map_name = game_info.map_name

    def _draw_terrain_creep(self, raw_obs, layers):
        """Draws Zerg Creep from the dynamic observation state."""
        layer_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_CREEP]

        # Prefer live data from bot state
        creep_map = raw_obs.get('creep')
        if creep_map is None:
            game_info = raw_obs.get('game_info')
            creep_map = getattr(game_info, 'creep_grid', None)

        if not creep_map:
            return

        scale_x = creep_map.width / self.feature_map_width
        scale_y = creep_map.height / self.feature_map_height

        for y in range(self.feature_map_height):
            for x in range(self.feature_map_width):
                # Map Image(y) -> SC2(sy). Y=0 is Top.
                sc2_x = int(x * scale_x)
                sc2_y = int(y * scale_y)

                if sc2_x < creep_map.width and sc2_y < creep_map.height:
                    try:
                        # PixelMap access is [x, y]
                        if creep_map[(sc2_x, sc2_y)] != 0:
                            layers[layer_idx, y, x] = 0.5
                    except:
                        pass

    # ==============================================================================
    # UNIT HANDLERS (Modularized)
    # ==============================================================================

    def _process_single_unit(self, unit: Unit, layers: np.ndarray, is_enemy: bool,
                           map_w: int, map_h: int, game_data: GameData):
        """Orchestrates drawing all features for a single unit."""

        # 1. Convert Position to Pixel Coords
        px, py = self._get_pixel_coords(unit.position, map_w, map_h)
        if px is None: return

        # 2. Check Visibility (Simulate Fog of War for Cloaked units)
        if is_enemy and unit.is_cloaked and not unit.is_detected:
            return # Invisible

        # 3. Draw Components
        self._draw_unit_body(unit, layers, is_enemy, px, py)
        self._draw_unit_vitals(unit, layers, is_enemy, px, py)
        self._draw_unit_detection(unit, layers, is_enemy, px, py, map_w)
        self._draw_unit_healing(unit, layers, is_enemy, px, py, map_w)
        self._draw_unit_dps(unit, layers, is_enemy, px, py, map_w, game_data)

    def _get_pixel_coords(self, position, map_w, map_h):
        if not position: return None, None

        x = int(position.x / map_w * self.feature_map_width)
        # Flip Y
        y = self.feature_map_height - 1 - int(position.y / map_h * self.feature_map_height)

        if 0 <= x < self.feature_map_width and 0 <= y < self.feature_map_height:
            return x, y
        return None, None

    def _draw_unit_body(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int):
        """Draws Density, Attribute, Flying, and Cloak layers."""
        prefix = "enemy" if is_enemy else "allied"

        # 1. Density (Presence/Readiness)
        idx_den = LayerConstants.LAYERS[f"{prefix}_unit_density"]
        if is_enemy:
            layers[idx_den, py, px] = 1.0
        else:
            # Calculate Readiness
            # Approx 22.4 frames = 1 second.
            cooldown_factor = 1.0 if unit.weapon_cooldown == 0 else max(0.0, 1.0 - (unit.weapon_cooldown / 22.4))
            # Map to 0.5 (Just Fired) ... 1.0 (Ready)
            layers[idx_den, py, px] = 0.5 + (0.5 * cooldown_factor)

        # 2. Armor Attribute
        idx_arm = LayerConstants.LAYERS[f"{prefix}_unit_armor"]
        val_arm = 0.5 if unit.is_light else (1.0 if unit.is_armored else 0.0)
        layers[idx_arm, py, px] = val_arm

        # 3. Type Attribute (Bio/Mech)
        idx_typ = LayerConstants.LAYERS[f"{prefix}_unit_type"]
        val_typ = 0.0
        if unit.is_biological and unit.is_mechanical: val_typ = 0.75
        elif unit.is_biological: val_typ = 1.0
        elif unit.is_mechanical: val_typ = 0.5
        layers[idx_typ, py, px] = val_typ

        # 4. Special Attribute (Massive/Psionic)
        idx_spc = LayerConstants.LAYERS[f"{prefix}_unit_special"]
        val_spc = 1.0 if unit.is_massive else (0.5 if unit.is_psionic else 0.0)
        layers[idx_spc, py, px] = val_spc

        # 5. Flying (Hybrid Encoding)
        idx_fly = LayerConstants.LAYERS[f"{prefix}_unit_flying"]
        val_fly = 0.0
        if unit.is_flying: val_fly = 1.0
        elif unit.type_id == UnitTypeId.COLOSSUS: val_fly = 0.5
        layers[idx_fly, py, px] = val_fly

        # 6. Cloak
        idx_clk = LayerConstants.LAYERS[f"{prefix}_unit_cloak"]
        val_clk = 0.0
        if unit.is_cloaked:
            val_clk = 0.6 # Active
        elif unit.type_id in [UnitTypeId.GHOST, UnitTypeId.BANSHEE, UnitTypeId.DARKTEMPLAR,
                              UnitTypeId.WIDOWMINE, UnitTypeId.OBSERVER, UnitTypeId.MOTHERSHIP]:
            val_clk = 0.3 # Capable
        layers[idx_clk, py, px] = val_clk

    def _draw_unit_vitals(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int):
        prefix = "enemy" if is_enemy else "allied"

        # Health
        idx_hp = LayerConstants.LAYERS[f"{prefix}_health"]
        max_hp = (unit.health_max + unit.shield_max) or 1
        current_hp = unit.health + unit.shield
        layers[idx_hp, py, px] = current_hp / max_hp

        # Energy
        if unit.energy_max > 0:
            idx_nrg = LayerConstants.LAYERS[f"{prefix}_energy"]
            layers[idx_nrg, py, px] = unit.energy / unit.energy_max

    def _draw_unit_detection(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int, map_w: int):
        prefix = "enemy" if is_enemy else "allied"
        layer_idx = LayerConstants.LAYERS[f"{prefix}_detection"]

        # Get Range
        radius_game = unit.detect_range
        if radius_game <= 0:
            # Fallback to DB
            db_data = self._get_unit_data_from_json(unit.type_id.name)
            if db_data:
                radius_game = db_data.get("detect", 0.0)

        if radius_game > 0:
            self._draw_circle(layers, layer_idx, px, py, radius_game, map_w, value=1.0)

    def _draw_unit_healing(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int, map_w: int):
        prefix = "enemy" if is_enemy else "allied"
        layer_idx = LayerConstants.LAYERS[f"{prefix}_hps_field"]

        hps = 0.0
        range_game = 0.0

        # Must be constructed
        is_ready = unit.is_ready if hasattr(unit, "is_ready") else (unit.build_progress == 1.0)

        if is_ready:
            uid = unit.type_id
            if uid == UnitTypeId.MEDIVAC and unit.energy > 5:
                hps, range_game = 12.6, 4.0
            elif uid == UnitTypeId.SHIELDBATTERY and unit.energy > 3:
                hps, range_game = 50.4, 6.0
            elif uid == UnitTypeId.QUEEN and unit.energy >= 50:
                hps, range_game = 25.0, 7.0

        if hps > 0:
            # Normalized HPS value for the heatmap
            self._draw_circle(layers, layer_idx, px, py, range_game, map_w, value=hps, add_mode=True)

    def _draw_unit_dps(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int, map_w: int, game_data: GameData):
        # Exclude units handled by Effects or Sub-units
        if unit.type_id in [UnitTypeId.LIBERATORAG, UnitTypeId.CARRIER, UnitTypeId.LURKERMP,
                           UnitTypeId.MEDIVAC, UnitTypeId.WARPPRISM, UnitTypeId.OBSERVER,
                           UnitTypeId.OVERLORD, UnitTypeId.OVERSEER]:
            return

        weapon_stats = []
        is_override = False
        uid = unit.type_id

        # --- VIRTUAL OVERRIDES ---
        if uid == UnitTypeId.WIDOWMINEBURROWED:
            is_override = True
            max_cd = 29.0 * 22.4
            ratio = max(0.0, 1.0 - (unit.weapon_cooldown / max_cd))
            # 80 DPS Burst
            weapon_stats = [{'base': 80.0*ratio, 'b_l':0, 'b_a':0, 'b_m':0, 'rng':5.0, 'typ':3}]
        elif uid == UnitTypeId.BANELING:
            is_override = True
            # 20 Base + 15 vs Light
            weapon_stats = [{'base': 20.0, 'b_l':15.0, 'b_a':0, 'b_m':0, 'rng':2.2, 'typ':1}]
        elif uid == UnitTypeId.ORACLE:
            is_override = True
            if unit.energy >= 25:
                weapon_stats = [{'base': 24.4, 'b_l':16.3, 'b_a':0, 'b_m':0, 'rng':4.0, 'typ':1}]
        elif uid == UnitTypeId.TEMPEST:
            is_override = True
            upg = getattr(unit, "attack_upgrade_level", 0)
            d_a = (30+3*upg)/2.36
            d_g = (40+4*upg)/2.36
            # Air Weapon has +22 vs Massive
            weapon_stats = [
                {'base': d_a, 'b_l':0, 'b_a':0, 'b_m':(22/2.36), 'rng':14.0, 'typ':2},
                {'base': d_g, 'b_l':0, 'b_a':0, 'b_m':0,         'rng':10.0, 'typ':1}
            ]

        # --- DB/API LOOKUP ---
        if not is_override:
            db = self._get_unit_data_from_json(unit.type_id.name)
            if db:
                for w in db.get('weapons', []):
                    upg = getattr(unit, "attack_upgrade_level", 0)
                    base_dmg = w['damage'] + (w.get('upgrade_bonus', 0) * upg)

                    b_l, b_a, b_m = 0.0, 0.0, 0.0
                    for b in w.get('bonuses', []):
                        if b['type'] == 1: b_l += b['bonus']
                        elif b['type'] == 2: b_a += b['bonus']
                        elif b['type'] == 7: b_m += b['bonus']

                    period = w['period'] or 1.0
                    attacks = w['attacks']

                    weapon_stats.append({
                        'base': (base_dmg * attacks) / period,
                        'b_l': (b_l * attacks) / period,
                        'b_a': (b_a * attacks) / period,
                        'b_m': (b_m * attacks) / period,
                        'rng': w['range'] + unit.radius,
                        'typ': w['type']
                    })

        # --- DRAWING ---
        for w in weapon_stats:
            self._draw_dps_circle(layers, is_enemy, px, py, w['rng'], map_w, w)

    # ==============================================================================
    # EFFECTS HANDLER
    # ==============================================================================

    def _draw_effects(self, effects, layers, map_w, map_h):
        # IDs
        LIB_IDS = {82, 83}
        STORM_ID = 1
        NUKE_ID = 11
        BILE_ID = 12 # Approximate ID for Ravager Bile

        for effect in effects:
            e_id = effect.id.value if hasattr(effect.id, "value") else effect.id

            # Config
            dps, radius, target_type = 0.0, 0.0, 0

            if e_id in LIB_IDS:
                dps, radius, target_type = 85.0, 5.0, 1 # Ground
            elif e_id == STORM_ID:
                dps, radius, target_type = 28.0, 1.5, 1 # Ground (Usually treated as ground threat)
            elif e_id == NUKE_ID:
                dps, radius, target_type = 300.0, 8.0, 3 # Both
            elif e_id == BILE_ID:
                dps, radius, target_type = 200.0, 2.0, 3 # Both
            else:
                continue

            for pos in effect.positions:
                # Convert Effect Pos to Pixels
                px, py = self._get_pixel_coords(pos, map_w, map_h)
                if px is None: continue

                # Alliance Matching
                # Default to Enemy (Threat) unless we find a very close Ally source
                is_ally = False

                # Check proximity to cached allied units
                # (Optimized: Only check if we actually have allied casters of that type)
                # For now, assume Enemy Threat (Layers 24+) is safer for RL training
                # EXCEPT for Liberators where we tracked them explicitly.

                if e_id in LIB_IDS:
                     min_ally = min((pos.distance_to_point2(p) for p in self._liberator_positions['allied']), default=999)
                     min_enemy = min((pos.distance_to_point2(p) for p in self._liberator_positions['enemy']), default=999)
                     if min_ally < min_enemy and min_ally < 16.0:
                         is_ally = True

                # Construct pseudo-weapon stat
                w_stat = {
                    'base': dps, 'b_l': 0, 'b_a': 0, 'b_m': 0,
                    'typ': target_type
                }

                self._draw_dps_circle(layers, not is_ally, px, py, radius, map_w, w_stat)

    # ==============================================================================
    # DRAWING HELPERS
    # ==============================================================================

    def _draw_circle(self, layers, layer_idx, cx, cy, radius_game, map_w, value, add_mode=False):
        """Draws a filled circle on the specified layer."""
        radius_px = max(1, int(radius_game / map_w * self.feature_map_width))
        r_sq = radius_px ** 2

        y_min = max(0, cy - radius_px)
        y_max = min(self.feature_map_height, cy + radius_px + 1)
        x_min = max(0, cx - radius_px)
        x_max = min(self.feature_map_width, cx + radius_px + 1)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (x - cx)**2 + (y - cy)**2 <= r_sq:
                    if add_mode:
                        layers[layer_idx, y, x] += value
                    else:
                        layers[layer_idx, y, x] = value

    def _draw_dps_circle(self, layers, is_enemy, cx, cy, radius_game, map_w, w_stats):
        """Draws DPS values onto the correct layers based on target type."""

        # Resolve Layer Indices based on constants
        if not is_enemy:
            # Ally: Ground(16..19), Air(20..23)
            idx_g = [16, 17, 18, 19]
            idx_a = [20, 21, 22, 23]
        else:
            # Enemy: Ground(24..27), Air(28..31)
            idx_g = [24, 25, 26, 27]
            idx_a = [28, 29, 30, 31]

        # Prepare values
        vals = [w_stats['base'], w_stats['b_l'], w_stats['b_a'], w_stats['b_m']]
        typ = w_stats['typ']

        # Calculate Geometry
        radius_px = max(1, int(radius_game / map_w * self.feature_map_width))
        r_sq = radius_px ** 2

        y_min = max(0, cy - radius_px)
        y_max = min(self.feature_map_height, cy + radius_px + 1)
        x_min = max(0, cx - radius_px)
        x_max = min(self.feature_map_width, cx + radius_px + 1)

        # Draw
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (x - cx)**2 + (y - cy)**2 <= r_sq:
                    # Ground Layers
                    if typ == 1 or typ == 3:
                        for i in range(4): layers[idx_g[i], y, x] += vals[i]
                    # Air Layers
                    if typ == 2 or typ == 3:
                        for i in range(4): layers[idx_a[i], y, x] += vals[i]