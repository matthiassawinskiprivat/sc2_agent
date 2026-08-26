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
from scipy.ndimage import binary_dilation

# SC2 Dependencies
from sc2.unit import Unit
from sc2.game_data import GameData
from sc2.ids.ability_id import AbilityId
from sc2.ids.effect_id import EffectId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

try:
    from sc2.constants import IS_DETECTOR, UNIT_PHOTONCANNON
except ImportError:  # pragma: no cover — older burnysc2
    IS_DETECTOR = {
        UnitTypeId.OBSERVER,
        UnitTypeId.OBSERVERSIEGEMODE,
        UnitTypeId.RAVEN,
        UnitTypeId.MISSILETURRET,
        UnitTypeId.OVERSEER,
        UnitTypeId.OVERSEERSIEGEMODE,
        UnitTypeId.SPORECRAWLER,
    }
    UNIT_PHOTONCANNON = UnitTypeId.PHOTONCANNON

# Project Constants
from src.utils.layer_constants import LayerConstants
from src.utils.command_constants import CommandConstants
from src.utils.splash_data import (
    LIBERATOR_ZONE_DAMAGE,
    LIBERATOR_ZONE_RADIUS_GU,
    PARASITIC_BOMB_DAMAGE,
    PARASITIC_BOMB_DPS,
    PARASITIC_BOMB_RADIUS_GU,
    splash_spec,
)
from src.utils.obs_geometry import (
    TACTICAL_HALF,
    TACTICAL_OBS_SIZE,
    compute_global_obs_geometry,
    tactical_crop_origin,
)

# Tactical 1×1 stamps — flying support + tiny structures (not full tile footprints).
_TACTICAL_SINGLE_PIXEL_UNITS: frozenset = frozenset({
    "OVERLORD",
    "OVERLORDTRANSPORT",
    "OVERSEER",
    "OVERSEERSIEGEMODE",
    "OBSERVER",
    "OBSERVERSIEGEMODE",
    "CREEPTUMOR",
    "CREEPTUMORBURROWED",
})
# Units with passive cliff traversal (Colossus: Cliff Walk, Reaper: jet-pack jumps).
# Used by the observation transformer to automatically widen the terrain_pathable layer
# for these unit types — no bot needs to set cliff_walker explicitly in raw_obs.
_CLIFF_WALKER_NAMES: frozenset = frozenset({'COLOSSUS', 'REAPER'})

# 4-connected cross structuring element — used for binary erosion when finding
# walkable cells at the walkable boundary (adjacent to any non-walkable cell).
_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

# Playable size (game units) used as reference so COMMAND_ASSIGNMENT stays as visible as on 128×128 maps.
_ASSIGNMENT_REFERENCE_PLAYABLE = 128
# Fixed game-space radius for COMMAND_OBJECTIVE on the global 128×128 view (Point2 + beacon).
_OBJECTIVE_REFERENCE_RADIUS_GU = 3.0

# Half-extent (game units) for structure footprints when live API / unit_data omit radius.
# Values follow SC2 editor footprint sizes: NxN tiles → N/2 from center to edge.
_STRUCTURE_FOOTPRINT_RADIUS: Dict[str, float] = {
    # 5x5 town halls
    "COMMANDCENTER": 2.5,
    "COMMANDCENTERFLYING": 2.5,
    "ORBITALCOMMAND": 2.5,
    "ORBITALCOMMANDFLYING": 2.5,
    "PLANETARYFORTRESS": 2.5,
    "HATCHERY": 2.5,
    "LAIR": 2.5,
    "HIVE": 2.5,
    "NEXUS": 2.5,
    # 3x3 production / support
    "BARRACKS": 1.5,
    "FACTORY": 1.5,
    "STARPORT": 1.5,
    "BUNKER": 1.5,
    "ENGINEERINGBAY": 1.5,
    "ARMORY": 1.5,
    "FUSIONCORE": 1.5,
    "GHOSTACADEMY": 1.5,
    "REFINERY": 1.5,
    "EXTRACTOR": 1.5,
    "ASSIMILATOR": 1.5,
    "VESPENEGEYSER": 1.5,
    "MINERALFIELD": 1.5,
    "GATEWAY": 1.5,
    "WARPGATE": 1.5,
    "ROBOTICSFACILITY": 1.5,
    "STARGATE": 1.5,
    "FORGE": 1.5,
    "CYBERNETICSCORE": 1.5,
    "TWILIGHTCOUNCIL": 1.5,
    "ROBOTICSBAY": 1.5,
    "FLEETBEACON": 1.5,
    "TEMPLARARCHIVE": 1.5,
    "DARKSHRINE": 1.5,
    "ROACHWARREN": 1.5,
    "HYDRALISKDEN": 1.5,
    "BANELINGNEST": 1.5,
    "SPIRE": 1.5,
    "GREATERSPIRE": 1.5,
    "ULTRALISKCAVERN": 1.5,
    "INFESTATIONPIT": 1.5,
    "EVOLUTIONCHAMBER": 1.5,
    "SPINECRAWLER": 1.5,
    "SPORECRAWLER": 1.5,
    # 2x2 small structures
    "SUPPLYDEPOT": 1.0,
    "PYLON": 1.0,
    "MISSILETURRET": 1.0,
    "SENSORTOWER": 1.0,
    "AUTOTURRET": 1.0,
    "PHOTONCANNON": 1.0,
    "SHIELDBATTERY": 1.0,
    "REACTOR": 1.0,
    "TECHLAB": 1.0,
    # 1×1 zerg creep spreaders (not 2×2 buildings)
    "CREEPTUMOR": 0.375,
    "CREEPTUMORBURROWED": 0.375,
}

# Terran production buildings and their 2×2 add-on slot (center-to-center offset).
_ADDON_PARENT_TYPES: frozenset = frozenset({
    "BARRACKS",
    "FACTORY",
    "STARPORT",
    "BARRACKSFLYING",
    "FACTORYFLYING",
    "STARPORTFLYING",
})
_ADDON_TYPES: frozenset = frozenset({"TECHLAB", "REACTOR"})
_ADDON_SLOT_OFFSET: Tuple[float, float] = (2.5, -0.5)
_ADDON_FOOTPRINT_RADIUS: float = 1.0

# Fallback display radii for units missing from unit_data.json / CTF UNIT_DATA.
_UNIT_DISPLAY_RADIUS_SUPPLEMENT: Dict[str, float] = {
    "OBSERVER": 0.75,
    "OBSERVERSIEGEMODE": 0.75,
    # Protoss support
    "WARPPRISM": 1.0,
    "WARPPRISMPHASING": 1.0,
    "SENTRY": 0.5,
    # Workers / transports (also listed in CTF UNIT_DATA when import succeeds)
    "SCV": 0.375,
    "PROBE": 0.375,
    "DRONE": 0.375,
    "MULE": 0.375,
    "MEDIVAC": 1.0,
    "OVERLORD": 0.875,
    "OVERLORDTRANSPORT": 0.875,
    "OVERSEER": 0.875,
    "OVERSEERSIEGEMODE": 0.875,
    "ZEALOT": 0.5,
    "STALKER": 0.625,
    "IMMORTAL": 1.0,
    "ZERGLING": 0.375,
    "ROACH": 0.75,
    "HYDRALISK": 0.625,
    "ULTRALISK": 1.0,
    "ARCHON": 1.0,
    "QUEEN": 0.875,
}

# Body half-extent for mobiles whose unit_data ``radius`` is selection/collision sized
# (e.g. SIEGETANKSIEGED radius 2.0 == weapon min_range, not the rendered hull).
_UNIT_BODY_RADIUS_OVERRIDE: Dict[str, float] = {
    "SIEGETANKSIEGED": 1.25,
}

# Tactical 1:1 obs uses a fixed 2×2 hull for sieged tanks (catalog 1.25 gu spans 2–3
# pixels when sampled on the pathing grid; snapping avoids 2×3 / 3×3 wobble).
_TACTICAL_SIEGE_FOOTPRINT_CELLS = 2

# Tile-snapped N×N obs hulls — center-stamped odd sizes (1, 3, 5…) cannot hit 2×2.
_FIXED_FOOTPRINT_CELLS: Dict[str, int] = {
    "ULTRALISK": 2,
    "ARCHON": 2,
}

# Minimum siege dead-zone radius on the obs grid (pixels) when downsampling would hide it.
_MIN_WEAPON_MIN_RANGE_OBS_RADIUS_PX = 1.5

# SC2 cliff tiers are 16 gu apart; map-stable normalization keeps tactical/global fixed.
_SC2_TERRAIN_LEVEL_STEP = 16.0
_TERRAIN_HEIGHT_PERCENTILE_FLOOR = 5.0
_TERRAIN_HEIGHT_PERCENTILE_CAP = 95.0
# Each cliff tier (16 gu) advances brightness by a fixed step so 2- and 3-level maps match.
_TERRAIN_BRIGHTNESS_FLOOR = 0.2
_TERRAIN_BRIGHTNESS_PER_CLIFF_TIER = 0.4

def _build_liberator_zone_effect_ids() -> frozenset:
    """SC2 liberation-zone effect ids (enum values 8/9; keep 82/83 for legacy snapshots)."""
    ids: set = set()
    for name in ("LIBERATORTARGETMORPHDELAYPERSISTENT", "LIBERATORTARGETMORPHPERSISTENT"):
        enum_val = getattr(EffectId, name, None)
        if enum_val is not None:
            ids.add(int(getattr(enum_val, "value", enum_val)))
    ids.update({82, 83})
    return frozenset(ids)


_LIBERATOR_ZONE_EFFECT_IDS = _build_liberator_zone_effect_ids()


def _build_threat_effect_specs() -> Dict[int, Tuple[float, float, bool]]:
    """Map SC2 ``EffectId`` values to (dps, radius_game, hits_air)."""
    specs: Dict[int, Tuple[float, float, bool]] = {}

    def _add(enum_name: str, dps: float, radius: float, hits_air: bool = False) -> None:
        enum_val = getattr(EffectId, enum_name, None)
        if enum_val is not None:
            specs[int(getattr(enum_val, "value", enum_val))] = (dps, radius, hits_air)

    _add("PSISTORMPERSISTENT", 28.0, 1.5, False)
    _add("NUKEPERSISTENT", 300.0, 8.0, True)
    _add("RAVAGERCORROSIVEBILECP", 200.0, 2.0, True)
    # Legacy numeric ids from older snapshots / replays.
    specs.setdefault(1, (28.0, 1.5, False))
    specs.setdefault(7, (300.0, 8.0, True))
    specs.setdefault(11, (200.0, 2.0, True))
    return specs


_THREAT_EFFECT_SPECS = _build_threat_effect_specs()


def _build_named_effect_ids(enum_name: str, *legacy: int) -> frozenset:
    """SC2 effect ids from a live ``EffectId`` name plus legacy numeric snapshots."""
    ids: set = set(legacy)
    enum_val = getattr(EffectId, enum_name, None)
    if enum_val is not None:
        ids.add(int(getattr(enum_val, "value", enum_val)))
    return frozenset(ids)


_BILE_EFFECT_IDS = _build_named_effect_ids("RAVAGERCORROSIVEBILECP", 11)
_STORM_EFFECT_IDS = _build_named_effect_ids("PSISTORMPERSISTENT", 1)
_NUKE_EFFECT_IDS = _build_named_effect_ids("NUKEPERSISTENT", 7)
_SCAN_EFFECT_IDS = frozenset(
    eid
    for eid in (
        int(getattr(getattr(EffectId, "SCANNERSWEEP", None), "value", 6)),
        6,
    )
)

# Body cull for tactical unit stamps — matches Level7Bot._TACTICAL_FOVEA_MAX_DELTA.
_TACTICAL_BODY_MAX_DELTA = float(TACTICAL_HALF + 1)

# Unit density + orthogonal attribute counters accumulate per overlapping unit (see LayerConstants).
_ADDITIVE_UNIT_COUNTER_LAYERS = frozenset({
    LayerConstants.ALLIED_UNIT_DENSITY,
    LayerConstants.ENEMY_UNIT_DENSITY,
    LayerConstants.ALLIED_UNIT_BIOLOGICAL,
    LayerConstants.ALLIED_UNIT_MECHANICAL,
    LayerConstants.ALLIED_UNIT_ARMORED,
    LayerConstants.ALLIED_UNIT_MASSIVE,
    LayerConstants.ALLIED_UNIT_FLYING,
    LayerConstants.ALLIED_UNIT_CLOAK,
    LayerConstants.ENEMY_UNIT_BIOLOGICAL,
    LayerConstants.ENEMY_UNIT_MECHANICAL,
    LayerConstants.ENEMY_UNIT_ARMORED,
    LayerConstants.ENEMY_UNIT_MASSIVE,
    LayerConstants.ENEMY_UNIT_FLYING,
    LayerConstants.ENEMY_UNIT_CLOAK,
})


def _load_unit_display_radius() -> Dict[str, float]:
    """Structure defaults + CTF curriculum radii + supplements (unit_data.json merged in __init__)."""
    out: Dict[str, float] = dict(_STRUCTURE_FOOTPRINT_RADIUS)
    try:
        from train.envs.ctf.game_data import UNIT_DATA, _ID_TABLE

        id_to_name = {v: k for k, v in _ID_TABLE.items()}
        for type_id, info in UNIT_DATA.items():
            name = id_to_name.get(type_id)
            if name is not None:
                out[name] = float(info.footprint_radius)
    except ImportError:
        pass
    for name, radius in _UNIT_DISPLAY_RADIUS_SUPPLEMENT.items():
        out.setdefault(name, radius)
    return out


_UNIT_DISPLAY_RADIUS: Optional[Dict[str, float]] = None


def _get_unit_display_radius() -> Dict[str, float]:
    global _UNIT_DISPLAY_RADIUS
    if _UNIT_DISPLAY_RADIUS is None:
        _UNIT_DISPLAY_RADIUS = _load_unit_display_radius()
    return _UNIT_DISPLAY_RADIUS


class ObservationTransformer:
    def __init__(self, map_size: tuple[int, int] = (128, 128), unit_data_file: str = None, training_conf: Dict = None):
        self.feature_map_width = map_size[0]
        self.feature_map_height = map_size[1]
        self.training_conf = training_conf if training_conf is not None else {}

        # Transformation State
        self.scale = 1.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.crop_x = 0.0
        self.crop_y = 0.0
        self._obs_span_w = float(map_size[0])
        self._obs_span_h = float(map_size[1])

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
        self._map_terrain_h_min: Optional[float] = None
        self._map_terrain_tier_max: Optional[float] = None
        self._map_terrain_h_scale_map: Optional[str] = None
        # Cliff face mask cache — depends only on the static pathing/height grid,
        # so it is valid for the lifetime of a map (recomputed on map change only).
        self._cliff_face_cache: Optional[np.ndarray] = None
        self._cliff_face_cache_map: Optional[str] = None
        self._attached_addon_tags: set = set()
        # Per-cell cliff-face brightness in obs space (midpoint of local high/low tiers).
        self._cliff_display_heights: Optional[np.ndarray] = None

        # Pre-allocated grids
        self._grid_y = np.arange(self.feature_map_height).reshape(-1, 1)
        self._grid_x = np.arange(self.feature_map_width).reshape(1, -1)

        self._liberator_positions = {'allied': [], 'enemy': []}
        self._ravager_positions = {'allied': [], 'enemy': []}
        self._ht_positions = {'allied': [], 'enemy': []}
        self._ghost_positions = {'allied': [], 'enemy': []}
        # (x, y, detect_range) discs from the current allied lists — used so a
        # P2 trainee can still see P1 cloaked enemies when a P2 detector covers
        # them. Proto ``is_detected`` is the P1-client view after god-mode.
        self._allied_detector_discs: List[Tuple[float, float, float]] = []
        self._cached_path_data: Optional[np.ndarray] = None
        self._current_game_data: Optional[GameData] = None

        # Load DB
        if unit_data_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            unit_data_file = os.path.join(base_dir, "data", "unit_data.json")

        self.fallback_db = self._load_fallback_db(unit_data_file)
        self._display_radius_db = dict(_get_unit_display_radius())
        self._merge_json_display_radii()
        self._display_radius_by_type_id = self._build_display_radius_by_type_id()

    def _build_display_radius_by_type_id(self) -> Dict[int, float]:
        """Map UnitTypeId values to display radii when ``type_id.name`` is missing."""
        out: Dict[int, float] = {}
        try:
            from sc2.ids.unit_typeid import UnitTypeId

            for ut in UnitTypeId:
                name = getattr(ut, "name", None)
                if not name:
                    continue
                radius = self._display_radius_db.get(str(name).upper())
                if radius is not None:
                    out[int(ut.value)] = float(radius)
        except ImportError:
            pass
        return out

    @staticmethod
    def _resolve_unit_type_name(unit: Unit) -> Optional[str]:
        type_id = getattr(unit, "type_id", None)
        if type_id is None:
            return None
        name = getattr(type_id, "name", None)
        if name:
            return str(name).upper()
        try:
            from sc2.ids.unit_typeid import UnitTypeId

            return str(UnitTypeId(int(type_id)).name).upper()
        except (TypeError, ValueError, KeyError, AttributeError):
            return None

    @staticmethod
    def _is_addon_parent_type(name: Optional[str]) -> bool:
        return name in _ADDON_PARENT_TYPES if name else False

    @staticmethod
    def _is_addon_type(name: Optional[str]) -> bool:
        return name in _ADDON_TYPES if name else False

    def _refresh_attached_addon_tags(self, *structure_lists: List[Unit]) -> None:
        """Tags of TECHLAB/REACTOR units already drawn via their parent production building."""
        tags: set = set()
        for structures in structure_lists:
            for unit in structures or []:
                if not getattr(unit, "is_structure", False):
                    continue
                if not self._is_addon_parent_type(self._resolve_unit_type_name(unit)):
                    continue
                addon_tag = getattr(unit, "add_on_tag", 0)
                if isinstance(addon_tag, (int, np.integer)) and addon_tag:
                    tags.add(addon_tag)
        self._attached_addon_tags = tags

    def _structure_has_attached_addon(self, unit: Unit) -> bool:
        if not self._is_addon_parent_type(self._resolve_unit_type_name(unit)):
            return False
        addon_tag = getattr(unit, "add_on_tag", 0)
        return isinstance(addon_tag, (int, np.integer)) and addon_tag != 0

    def _is_attached_addon_unit(self, unit: Unit) -> bool:
        if not getattr(unit, "is_structure", False):
            return False
        if not self._is_addon_type(self._resolve_unit_type_name(unit)):
            return False
        tag = getattr(unit, "tag", None)
        return tag in self._attached_addon_tags

    @staticmethod
    def _addon_slot_position(unit: Unit) -> Tuple[float, float]:
        slot = getattr(unit, "add_on_position", None)
        if slot is not None and hasattr(slot, "x") and hasattr(slot, "y"):
            sx, sy = getattr(slot, "x", None), getattr(slot, "y", None)
            if isinstance(sx, (int, float)) and isinstance(sy, (int, float)):
                return float(sx), float(sy)
        cx, cy = float(unit.position.x), float(unit.position.y)
        ox, oy = _ADDON_SLOT_OFFSET
        return cx + ox, cy + oy

    def _merge_obs_rect_bounds(
        self,
        *bounds: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        x0 = min(b[0] for b in bounds)
        x1 = max(b[1] for b in bounds)
        y0 = min(b[2] for b in bounds)
        y1 = max(b[3] for b in bounds)
        return x0, x1, y0, y1

    def _merge_json_display_radii(self) -> None:
        """Apply unit_data.json radii (authoritative game footprint when present)."""
        for name, data in self.fallback_db.items():
            for key in ("display_radius", "footprint_radius", "radius"):
                val = data.get(key, 0)
                if isinstance(val, (int, float)) and val > 0:
                    self._display_radius_db[name] = float(val)
                    break

    def _positions_for_unit_type(
        self,
        type_id,
        allied_units,
        allied_structures,
        enemy_units,
        enemy_structures,
    ) -> Dict[str, list]:
        """Collect unit positions by type from perspective-split lists."""
        out: Dict[str, list] = {"allied": [], "enemy": []}
        if type_id is None:
            return out
        type_val = type_id.value if hasattr(type_id, "value") else type_id

        def _matches(unit: Unit) -> bool:
            tid = getattr(unit, "type_id", None)
            tid = tid.value if hasattr(tid, "value") else tid
            return tid == type_val

        for unit in list(allied_units or []) + list(allied_structures or []):
            if _matches(unit):
                out["allied"].append(unit.position)
        for unit in list(enemy_units or []) + list(enemy_structures or []):
            if _matches(unit):
                out["enemy"].append(unit.position)
        return out

    def _collect_liberator_positions(
        self,
        allied_units,
        allied_structures,
        enemy_units,
        enemy_structures,
    ) -> None:
        """Track caster positions for effect ownership (liberator, bile, storm, nuke)."""
        args = (allied_units, allied_structures, enemy_units, enemy_structures)
        self._liberator_positions = self._positions_for_unit_type(
            getattr(UnitTypeId, "LIBERATORAG", None), *args
        )
        self._ravager_positions = self._positions_for_unit_type(
            getattr(UnitTypeId, "RAVAGER", None), *args
        )
        self._ht_positions = self._positions_for_unit_type(
            getattr(UnitTypeId, "HIGHTEMPLAR", None), *args
        )
        self._ghost_positions = self._positions_for_unit_type(
            getattr(UnitTypeId, "GHOST", None), *args
        )

    def _has_active_ground_dps_layers(self, is_enemy: bool) -> bool:
        """True when any configured ground DPS channel is active for allied/enemy."""
        prefix = "enemy" if is_enemy else "allied"
        for suffix in ("base", "light", "armor", "massive"):
            idx = self.current_layer_map.get(f"{prefix}_g_dps_{suffix}")
            if idx is not None and idx in self.active_indices:
                return True
        return False

    def _has_active_air_dps_layers(self, is_enemy: bool) -> bool:
        """True when any configured air DPS channel is active for allied/enemy."""
        prefix = "enemy" if is_enemy else "allied"
        for suffix in ("base", "light", "armor", "massive"):
            idx = self.current_layer_map.get(f"{prefix}_a_dps_{suffix}")
            if idx is not None and idx in self.active_indices:
                return True
        return False

    def _has_active_dps_layers(self, is_enemy: bool) -> bool:
        """True when any ground or air DPS channel is active."""
        return self._has_active_ground_dps_layers(is_enemy) or self._has_active_air_dps_layers(
            is_enemy
        )

    def _has_active_splash_layers(self, is_enemy: bool) -> bool:
        """True when allied/enemy splash channel is configured and active."""
        name = LayerConstants.ENEMY_SPLASH if is_enemy else LayerConstants.ALLIED_SPLASH
        idx = self.current_layer_map.get(name)
        return idx is not None and idx in self.active_indices

    def _resolve_axis_dps_layer(
        self, prefix: str, axis: str, suffix: Optional[str] = None
    ) -> Optional[int]:
        """
        Active DPS layer for one axis (``g`` or ``a``), with cross-axis fallback.

        CTF and combat configs often enable only ``g_dps_*`` layers; air-only weapons
        (Tempest, Phoenix) still need to project threat there when ``a_dps_*`` is off.
        """
        layer_suffix = suffix or "base"
        primary = self.current_layer_map.get(f"{prefix}_{axis}_dps_{layer_suffix}")
        if primary is not None and primary in self.active_indices:
            return primary
        other = "a" if axis == "g" else "g"
        fallback = self.current_layer_map.get(f"{prefix}_{other}_dps_{layer_suffix}")
        if fallback is not None and fallback in self.active_indices:
            return fallback
        return None

    def _primary_axis_dps_base_active(self, prefix: str, axis: str) -> bool:
        """True when ``{prefix}_{axis}_dps_base`` is active (no cross-axis fallback).

        Typed bonus channels are bonus-only only when *this* axis carries base DPS.
        Cross-axis fallback for drawing must not suppress base+bonus on typed layers.
        """
        idx = self.current_layer_map.get(f"{prefix}_{axis}_dps_base")
        return idx is not None and idx in self.active_indices

    @staticmethod
    def _unit_on_cloak_layer(unit: Unit) -> bool:
        """True when the unit should stamp the cloak attribute layer."""
        return getattr(unit, "is_cloaked", False) or getattr(unit, "is_burrowed", False)

    @staticmethod
    def _unit_is_burrowed(unit: Unit) -> bool:
        """True for burrowed morphs (flag and/or *BURROWED type id)."""
        if bool(getattr(unit, "is_burrowed", False)):
            return True
        name = getattr(getattr(unit, "type_id", None), "name", None)
        return bool(name) and "BURROWED" in str(name).upper()

    @staticmethod
    def _skip_undetected_cloaked_enemy(
        unit: Unit,
        is_enemy: bool,
        *,
        allied_detector_covers: bool = False,
    ) -> bool:
        """True when an enemy unit should be omitted (cloaked and not detected).

        ``is_detected`` is the P1-client proto. After ``debug_control_enemy``, a
        P1 cloaked unit often has ``is_detected=False`` even when a P2 Observer
        covers it. Pass ``allied_detector_covers=True`` from owner-split allied
        detector discs so a P2 trainee still paints that enemy.
        """
        if not is_enemy:
            return False
        if not ObservationTransformer._unit_on_cloak_layer(unit):
            return False
        if allied_detector_covers:
            return False
        if getattr(unit, "is_detected", True):
            return False
        return not getattr(unit, "is_snapshot", False)

    @staticmethod
    def _unit_is_combat_ready(unit: Unit) -> bool:
        """False while a unit is warping or still under construction (no weapon DPS yet)."""
        if not getattr(unit, "is_ready", True):
            return False
        progress = float(getattr(unit, "build_progress", 1.0))
        return progress >= 1.0

    def _tactical_fovea_game_bounds(self) -> Tuple[float, float, float, float]:
        """Inclusive-min, exclusive-max game bounds of the 32×32 tactical window."""
        return (
            float(self.crop_x),
            float(self.crop_y),
            float(self.crop_x + TACTICAL_OBS_SIZE),
            float(self.crop_y + TACTICAL_OBS_SIZE),
        )

    def _tactical_game_point_inside_fovea(self, gx: float, gy: float) -> bool:
        x0, y0, x1, y1 = self._tactical_fovea_game_bounds()
        return x0 <= float(gx) < x1 and y0 <= float(gy) < y1

    def _tactical_footprint_intersects_fovea(self, unit: Unit) -> bool:
        """True when the unit's catalog footprint overlaps the tactical fovea."""
        game_r = self._footprint_game_radius(unit)
        cx = float(unit.position.x)
        cy = float(unit.position.y)
        ux0, ux1 = cx - game_r, cx + game_r
        uy0, uy1 = cy - game_r, cy + game_r
        fx0, fy0, fx1, fy1 = self._tactical_fovea_game_bounds()
        return ux0 < fx1 and ux1 > fx0 and uy0 < fy1 and uy1 > fy0

    def _tactical_body_inside_fovea(self, unit: Unit, active_unit: Unit) -> bool:
        del active_unit  # fovea is anchored on crop_x/crop_y, not focal distance
        return self._tactical_footprint_intersects_fovea(unit)

    def _unit_max_weapon_reach(self, unit: Unit) -> float:
        """Max ground or air weapon reach in game units (for tactical off-fovea DPS culling)."""
        if not self._unit_is_combat_ready(unit):
            return 0.0
        if unit.is_hallucination:
            return 0.0
        uid = unit.type_id
        if uid in [
            UnitTypeId.LIBERATORAG,
            UnitTypeId.CARRIER,
            UnitTypeId.LURKERMP,
            UnitTypeId.MEDIVAC,
            UnitTypeId.WARPPRISM,
            UnitTypeId.OBSERVER,
            UnitTypeId.OVERLORD,
            UnitTypeId.OVERSEER,
        ]:
            return 0.0
        if uid == UnitTypeId.WIDOWMINEBURROWED:
            return 5.0 + self._footprint_game_radius(unit)
        if uid == UnitTypeId.ORACLE and unit.energy >= 25:
            return 4.0 + self._footprint_game_radius(unit)
        u_name = getattr(unit.type_id, "name", None)
        if not u_name:
            return 0.0
        db = self._get_unit_data_from_json(u_name)
        if not db:
            return 0.0
        max_reach = 0.0
        for w in db.get("weapons", []):
            wtype = int(w.get("type", 0))
            if wtype not in (1, 2, 3):
                continue
            rng_g, rng_a = self._dps_weapon_game_radius(unit, float(w["range"]))
            if wtype in (1, 3):
                max_reach = max(max_reach, rng_g)
            if wtype in (2, 3):
                max_reach = max(max_reach, rng_a)
        return max_reach

    def _unit_max_ground_weapon_reach(self, unit: Unit) -> float:
        """Backward-compatible alias; includes air reach for off-fovea tactical culling."""
        return self._unit_max_weapon_reach(unit)

    def _tactical_weapon_ring_may_intersect_fovea(self, unit: Unit, active_unit: Unit) -> bool:
        """True when a weapon disc (ground or air) could overlap the 32×32 tactical fovea."""
        reach = self._unit_max_weapon_reach(unit)
        if reach <= 0:
            return False
        dx = abs(unit.position.x - active_unit.position.x)
        dy = abs(unit.position.y - active_unit.position.y)
        return max(dx, dy) <= float(TACTICAL_HALF) + reach

    def _tactical_splash_ring_may_intersect_fovea(self, unit: Unit, active_unit: Unit) -> bool:
        """True when a splash disc could overlap the 32×32 tactical fovea."""
        spec = splash_spec(unit)
        if spec is None:
            return False
        reach = float(spec[0])
        if reach <= 0:
            return False
        dx = abs(unit.position.x - active_unit.position.x)
        dy = abs(unit.position.y - active_unit.position.y)
        return max(dx, dy) <= float(TACTICAL_HALF) + reach

    def _draw_tactical_reach_in_dps(
        self,
        layers: np.ndarray,
        active_unit: Unit,
        allied_units,
        allied_structures,
        enemy_units,
        enemy_structures,
        game_data: GameData,
    ) -> None:
        """Draw weapon/splash rings for units outside the body cull whose reach overlaps the fovea."""
        for unit_list, is_enemy in (
            (list(allied_units) + list(allied_structures), False),
            (list(enemy_units) + list(enemy_structures), True),
        ):
            want_dps = self._has_active_dps_layers(is_enemy)
            want_splash = self._has_active_splash_layers(is_enemy)
            if not want_dps and not want_splash:
                continue
            for unit in unit_list:
                if self._is_attached_addon_unit(unit):
                    continue
                covers = is_enemy and self._allied_detector_covers(unit)
                if self._skip_undetected_cloaked_enemy(
                    unit, is_enemy, allied_detector_covers=covers
                ):
                    continue
                if self._tactical_body_inside_fovea(unit, active_unit):
                    continue
                px, py = self._get_pixel_coords_unclamped(unit.position)
                if px is None:
                    continue
                if want_dps and self._tactical_weapon_ring_may_intersect_fovea(unit, active_unit):
                    self._draw_unit_dps(unit, layers, is_enemy, px, py, game_data)
                if want_splash and self._tactical_splash_ring_may_intersect_fovea(unit, active_unit):
                    self._draw_unit_splash(unit, layers, is_enemy, px, py)

    def _effect_dps_layer_indices(
        self, prefix: str, hits_air: bool = False, hits_ground: bool = True
    ) -> List[int]:
        """DPS layer indices for effect zones (storm, nuke, liberator AA zone, parasitic bomb)."""
        indices: List[int] = []
        if hits_ground:
            base_idx = self.current_layer_map.get(f"{prefix}_g_dps_base")
            if base_idx is not None and base_idx in self.active_indices:
                indices.append(base_idx)
            else:
                for suffix in ("light", "armor", "massive"):
                    idx = self.current_layer_map.get(f"{prefix}_g_dps_{suffix}")
                    if idx is not None and idx in self.active_indices:
                        indices.append(idx)
        if hits_air:
            air_base = self.current_layer_map.get(f"{prefix}_a_dps_base")
            if air_base is not None and air_base in self.active_indices:
                indices.append(air_base)
            else:
                for suffix in ("light", "armor", "massive"):
                    idx = self.current_layer_map.get(f"{prefix}_a_dps_{suffix}")
                    if idx is not None and idx in self.active_indices:
                        indices.append(idx)
        if hits_air and not hits_ground and not indices:
            return self._effect_dps_layer_indices(prefix, hits_air=False, hits_ground=True)
        return indices

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
                     self._grid_x, self._grid_y,
                     self._obs_span_w, self._obs_span_h)
        
        # Tactical setup (32x32 centered on unit). Snap crop to integer game
        # coordinates so entities and terrain do not wiggle with fractional focal positions.
        self.scale = 1.0
        self.feature_map_width = 32
        self.feature_map_height = 32
        self.crop_x, self.crop_y = tactical_crop_origin(
            active_unit.position.x, active_unit.position.y
        )
        self.offset_x = 0
        self.offset_y = 0
        # Tactical fovea is always a full 32×32 window — do not inherit global
        # playable span / letterbox from transform().
        self._obs_span_w = float(TACTICAL_OBS_SIZE)
        self._obs_span_h = float(TACTICAL_OBS_SIZE)
        self.current_layer_map = LayerConstants.TACTICAL_LAYERS
        self.is_tactical_mode = True
        self._update_active_indices()

        self._grid_x = np.arange(32).reshape(1, -1)
        self._grid_y = np.arange(32).reshape(-1, 1)

        try:
            game_info = raw_obs.get('game_info')
            self._current_game_data = raw_obs.get('game_data')
            self._cache_pathing_grid(game_info)
            allied_units = raw_obs.get('units') or []
            enemy_units = raw_obs.get('enemy_units') or []
            allied_structures = raw_obs.get('structures') or []
            enemy_structures = raw_obs.get('enemy_structures') or []
            game_data = raw_obs.get('game_data')
            effects = raw_obs.get('effects') or []
            self._refresh_attached_addon_tags(allied_structures, enemy_structures)
            self._collect_allied_detector_discs(allied_units, allied_structures)
            
            # Draw
            cliff_walker = self._detect_cliff_walker(raw_obs, active_unit=active_unit)
            blocking_ground = [
                u for u in list(allied_units) + list(enemy_units)
                + list(allied_structures) + list(enemy_structures)
                if not getattr(u, "is_flying", False)
                and (
                    getattr(u, "is_structure", False)
                    or not self._unit_is_burrowed(u)
                )
            ]
            self._draw_terrain_pathing(
                game_info, layers, cliff_walker=cliff_walker, ground_units=blocking_ground
            )
            self._apply_map_blocker_terrain(layers, raw_obs)
            self._draw_terrain_creep(raw_obs, layers)
            self._draw_terrain_buildable(raw_obs, layers)

            # Allied / enemy bodies (and in-fovea DPS) — skip units outside the fovea.
            for unit in list(allied_units) + list(allied_structures):
                if not self._tactical_body_inside_fovea(unit, active_unit):
                    continue
                self._process_single_unit(unit, layers, False, TACTICAL_OBS_SIZE, TACTICAL_OBS_SIZE, game_data)

            for unit in list(enemy_units) + list(enemy_structures):
                if not self._tactical_body_inside_fovea(unit, active_unit):
                    continue
                self._process_single_unit(unit, layers, True, TACTICAL_OBS_SIZE, TACTICAL_OBS_SIZE, game_data)

            # Weapon rings from off-fovea units whose reach still overlaps the tactical window.
            self._draw_tactical_reach_in_dps(
                layers,
                active_unit,
                allied_units,
                allied_structures,
                enemy_units,
                enemy_structures,
                game_data,
            )

            # Draw Command Info (High-priority markers). Keep client selection when
            # present (debug UI multi-select); always include the focal unit.
            tactical_raw_obs = raw_obs.copy()
            selected = list(raw_obs.get("selected_units") or [])
            if active_unit is not None:
                active_tag = getattr(active_unit, "tag", None)
                if not any(getattr(u, "tag", None) == active_tag for u in selected):
                    selected.append(active_unit)
            if not selected and active_unit is not None:
                selected = [active_unit]
            tactical_raw_obs["selected_units"] = selected
            self._draw_command_info(tactical_raw_obs, layers)
            self._draw_transport_info(raw_obs, layers)

            self._collect_liberator_positions(
                allied_units, allied_structures, enemy_units, enemy_structures
            )
            if effects:
                self._draw_effects(effects, layers)

            # Normalize all DPS channels with tanh (matches global transform)
            for dps_layer_name in [
                LayerConstants.ALLIED_G_DPS_BASE, LayerConstants.ENEMY_G_DPS_BASE,
                LayerConstants.ALLIED_G_DPS_LIGHT, LayerConstants.ENEMY_G_DPS_LIGHT,
                LayerConstants.ALLIED_G_DPS_ARMOR, LayerConstants.ENEMY_G_DPS_ARMOR,
                LayerConstants.ALLIED_G_DPS_MASSIVE, LayerConstants.ENEMY_G_DPS_MASSIVE,
                LayerConstants.ALLIED_A_DPS_BASE, LayerConstants.ENEMY_A_DPS_BASE,
                LayerConstants.ALLIED_A_DPS_LIGHT, LayerConstants.ENEMY_A_DPS_LIGHT,
                LayerConstants.ALLIED_A_DPS_ARMOR, LayerConstants.ENEMY_A_DPS_ARMOR,
                LayerConstants.ALLIED_A_DPS_MASSIVE, LayerConstants.ENEMY_A_DPS_MASSIVE,
                LayerConstants.ALLIED_HPS_FIELD, LayerConstants.ENEMY_HPS_FIELD,
            ]:
                dps_layer_idx = LayerConstants.TACTICAL_LAYERS.get(dps_layer_name)
                if dps_layer_idx is not None:
                    layers[dps_layer_idx] = np.tanh(layers[dps_layer_idx] / 50.0)

            # Count layers: log1p accumulation (matches global transform pre-clip).
            for layer_name in _ADDITIVE_UNIT_COUNTER_LAYERS:
                layer_idx = self.current_layer_map.get(layer_name)
                if layer_idx is not None:
                    layers[layer_idx] = np.log1p(layers[layer_idx])

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
             self._grid_x, self._grid_y,
             self._obs_span_w, self._obs_span_h) = old_state
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
        self._current_game_data = game_data

        self._cache_pathing_grid(game_info)

        # 1. Scaling — fit playable area into obs pixels (inclusive edge mapping)
        if game_info:
            (
                self.crop_x,
                self.crop_y,
                mw,
                mh,
                self.scale_x,
                self.scale_y,
                self.offset_x,
                self.offset_y,
            ) = compute_global_obs_geometry(
                game_info,
                self.feature_map_width,
                self.feature_map_height,
                path_data=self._cached_path_data,
            )
            self._obs_span_w = float(mw)
            self._obs_span_h = float(mh)
            self.scale = min(self.scale_x, self.scale_y)
        else:
            self.crop_x, self.crop_y, mw, mh = 0, 0, 200, 200
            self._obs_span_w, self._obs_span_h = 200.0, 200.0
            self.scale = self.scale_x = self.scale_y = 1.0
            self.offset_x, self.offset_y = 0.0, 0.0

        # 2. Draw — global terrain is static map pathing only (cached; no live unit stamps).
        cliff_walker = self._detect_cliff_walker(raw_obs)
        self._draw_terrain_pathing(game_info, layers, cliff_walker=cliff_walker)
        self._apply_map_blocker_terrain(layers, raw_obs)
        self._draw_terrain_creep(raw_obs, layers)
        self._draw_terrain_buildable(raw_obs, layers)

        self._collect_liberator_positions(
            allied_units, allied_structures, enemy_units, enemy_structures
        )

        self._refresh_attached_addon_tags(allied_structures, enemy_structures)
        self._collect_allied_detector_discs(allied_units, allied_structures)

        for unit in list(allied_units) + list(allied_structures):
            if "MINERAL" in getattr(unit.type_id, "name", "").upper(): continue
            self._process_single_unit(unit, layers, False, mw, mh, game_data)

        for unit in list(enemy_units) + list(enemy_structures):
            if "MINERAL" in getattr(unit.type_id, "name", "").upper(): continue
            self._process_single_unit(unit, layers, True, mw, mh, game_data)

        if effects: self._draw_effects(effects, layers)
        self._draw_command_info(raw_obs, layers)
        self._draw_transport_info(raw_obs, layers)

        # 3. Post-processing
        dps_start = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]
        hps_end = LayerConstants.LAYERS[LayerConstants.ENEMY_HPS_FIELD] + 1
        allied_hp_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_HEALTH]
        enemy_hp_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_HEALTH]
        allied_hp = layers[allied_hp_idx].copy()
        enemy_hp = layers[enemy_hp_idx].copy()

        layers[dps_start:hps_end] = np.tanh(layers[dps_start:hps_end] / 50.0)
        layers[0:dps_start] = np.log1p(layers[0:dps_start])
        layers[allied_hp_idx] = allied_hp
        layers[enemy_hp_idx] = enemy_hp
        np.clip(layers, 0.0, 1.0, out=layers)

        if self.active_layers:
            indices = [LayerConstants.LAYERS[layer_name.lower()] for layer_name in self.active_layers if layer_name.lower() in LayerConstants.LAYERS]
            map_data = layers[indices]
        else:
            map_data = layers

        # ALWAYS return Dict to match the gymnasium.spaces.Dict in Level0Env
        return {"map": map_data, "command": command_vector}

    def _terrain_geometry_key(self) -> Tuple[Any, ...]:
        return (
            round(self.crop_x, 4),
            round(self.crop_y, 4),
            round(self._obs_span_w, 4),
            round(self._obs_span_h, 4),
            round(self.scale_x, 6),
            round(self.scale_y, 6),
            self.feature_map_width,
            self.feature_map_height,
        )

    def _cache_pathing_grid(self, game_info) -> None:
        if game_info is not None and hasattr(game_info, "pathing_grid"):
            self._cached_path_data = self._get_numpy_data(game_info.pathing_grid)
        else:
            self._cached_path_data = None

    def _obs_scales(self) -> Tuple[float, float]:
        if self.is_tactical_mode:
            return self.scale, self.scale
        return self.scale_x, self.scale_y

    def _game_to_obs_pixel_raw(self, gx: float, gy: float) -> Tuple[int, int]:
        """Map a SC2 world point to observation pixel indices without clamping."""
        scale_x, scale_y = self._obs_scales()
        content_x = (gx - self.crop_x) * scale_x + self.offset_x
        content_y = (gy - self.crop_y) * scale_y + self.offset_y
        if self.is_tactical_mode:
            px = int(math.floor(content_x))
            # Align with terrain rows (integer pathing grid); floor(31 - y) is one row north.
            py = int(self.feature_map_height - 1 - math.floor(content_y))
        else:
            px = int(round(content_x))
            py = int(round(self.feature_map_height - 1 - content_y))
        return px, py

    def _game_to_obs_pixel(self, gx: float, gy: float) -> Tuple[int, int]:
        """Map a SC2 world point to observation pixel indices."""
        px, py = self._game_to_obs_pixel_raw(gx, gy)
        px = max(0, min(px, self.feature_map_width - 1))
        py = max(0, min(py, self.feature_map_height - 1))
        return px, py

    def _obs_content_valid_mask(self) -> np.ndarray:
        """True where obs pixels fall inside the playable/content span (excludes letterbox)."""
        if self.is_tactical_mode:
            return np.ones(
                (self.feature_map_height, self.feature_map_width), dtype=bool
            )
        x_idx = np.arange(self.feature_map_width, dtype=np.float64)
        y_idx = np.arange(self.feature_map_height, dtype=np.float64)
        scale_x, scale_y = self._obs_scales()
        content_x = (x_idx - self.offset_x) / scale_x + self.crop_x
        content_y = (
            (self.feature_map_height - 1 - y_idx) - self.offset_y
        ) / scale_y + self.crop_y
        valid_x = (content_x >= self.crop_x) & (content_x < self.crop_x + self._obs_span_w)
        valid_y = (content_y >= self.crop_y) & (content_y < self.crop_y + self._obs_span_h)
        return np.logical_and.outer(valid_y, valid_x)

    def _obs_to_sc2_meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Map each observation pixel to SC2 pathing grid indices."""
        x_idx = np.arange(self.feature_map_width, dtype=np.float64)
        y_idx = np.arange(self.feature_map_height, dtype=np.float64)
        if self.is_tactical_mode:
            sc2_x = np.floor(((x_idx - self.offset_x) / self.scale) + self.crop_x).astype(int)
            sc2_y = np.floor(
                ((self.feature_map_height - 1 - y_idx - self.offset_y) / self.scale) + self.crop_y
            ).astype(int)
        else:
            scale_x, scale_y = self._obs_scales()
            sc2_x = np.round(((x_idx - self.offset_x) / scale_x) + self.crop_x).astype(int)
            sc2_y = np.round(
                ((self.feature_map_height - 1 - y_idx - self.offset_y) / scale_y) + self.crop_y
            ).astype(int)
        return np.meshgrid(sc2_x, sc2_y)

    def _get_pixel_coords(self, position) -> tuple:
        if position is None:
            return None, None
        px, py = self._game_to_obs_pixel(position.x, position.y)
        if 0 <= px < self.feature_map_width and 0 <= py < self.feature_map_height:
            return px, py
        return None, None

    def _get_pixel_coords_unclamped(self, position) -> tuple:
        """Map world position to obs pixels without clamping to the fovea bounds."""
        if position is None:
            return None, None
        return self._game_to_obs_pixel_raw(position.x, position.y)

    def _obs_pixel_to_game_center(self, px: int, py: int) -> Tuple[float, float]:
        """Game-space center of an obs pixel (inverse of _game_to_obs_pixel sampling)."""
        scale_x, scale_y = self._obs_scales()
        gx = (float(px) + 0.5 - self.offset_x) / scale_x + self.crop_x
        content_y = (self.feature_map_height - 1 - float(py)) + 0.5 - self.offset_y
        gy = content_y / scale_y + self.crop_y
        return gx, gy

    def _game_center_meshgrid_in_rect(
        self,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Float game (x, y) at the center of each obs pixel inside an inclusive rect."""
        scale_x, scale_y = self._obs_scales()
        px_idx = np.arange(x0, x1 + 1, dtype=np.float64)
        py_idx = np.arange(y0, y1 + 1, dtype=np.float64)
        gx = (px_idx + 0.5 - self.offset_x) / scale_x + self.crop_x
        content_y = (self.feature_map_height - 1 - py_idx) + 0.5 - self.offset_y
        gy = content_y / scale_y + self.crop_y
        return np.meshgrid(gx, gy)

    def _circle_center_game(
        self,
        cx: int,
        cy: int,
        unit: Optional[Unit] = None,
    ) -> Tuple[float, float]:
        """Game-space center for weapon discs — unit position when available."""
        if unit is not None:
            return float(unit.position.x), float(unit.position.y)
        return self._obs_pixel_to_game_center(cx, cy)

    def _process_single_unit(self, unit: Unit, layers: np.ndarray, is_enemy: bool,
                             map_w: int, map_h: int, game_data: GameData):
        if self._is_attached_addon_unit(unit):
            return

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
        covers = is_enemy and self._allied_detector_covers(unit)
        if self._skip_undetected_cloaked_enemy(
            unit, is_enemy, allied_detector_covers=covers
        ):
            return

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
            self._draw_unit_splash(unit, layers, is_enemy, px, py)
        else:
            self._draw_unit_vitals(unit, layers, is_enemy, px, py)
            self._draw_unit_detection(unit, layers, is_enemy, px, py)
            self._draw_unit_healing(unit, layers, is_enemy, px, py, game_data)
            self._draw_unit_dps(unit, layers, is_enemy, px, py, game_data)
            self._draw_unit_splash(unit, layers, is_enemy, px, py)

    def _unit_is_detector_structure(self, unit: Unit) -> bool:
        type_id = getattr(unit, "type_id", None)
        if type_id in IS_DETECTOR:
            return True
        if type_id == UNIT_PHOTONCANNON:
            return True
        name = getattr(type_id, "name", "")
        return str(name).upper() == "PHOTONCANNON"

    def _unit_detection_range(self, unit: Unit) -> float:
        """Detection radius in game units (0 when the unit does not actively detect)."""
        if not getattr(unit, "is_ready", True):
            return 0.0
        if float(getattr(unit, "build_progress", 1.0)) < 1.0:
            return 0.0
        if getattr(unit, "is_structure", False):
            if not self._unit_is_detector_structure(unit):
                return 0.0
            if not getattr(unit, "is_ready", True):
                return 0.0
            if float(getattr(unit, "build_progress", 1.0)) < 1.0:
                return 0.0
            type_id = getattr(unit, "type_id", None)
            if type_id == UNIT_PHOTONCANNON or getattr(type_id, "name", "").upper() == "PHOTONCANNON":
                if not getattr(unit, "is_powered", True):
                    return 0.0
        detect = float(getattr(unit, "detect_range", 0.0) or 0.0)
        if detect > 0:
            return detect
        u_name = self._resolve_unit_type_name(unit)
        if u_name:
            db = self._get_unit_data_from_json(u_name)
            if db:
                catalog = float(db.get("detect", 0.0) or 0.0)
                if catalog > 0:
                    return catalog
        return 0.0

    def _collect_allied_detector_discs(self, allied_units, allied_structures) -> None:
        """Cache allied detector (x, y, range) for cloak skip after god-mode."""
        discs: List[Tuple[float, float, float]] = []
        for unit in list(allied_units or []) + list(allied_structures or []):
            rng = self._unit_detection_range(unit)
            if rng <= 0:
                continue
            pos = getattr(unit, "position", None)
            if pos is None:
                continue
            discs.append((float(pos.x), float(pos.y), float(rng)))
        self._allied_detector_discs = discs

    def _allied_detector_covers(self, unit: Unit) -> bool:
        """True when an owner-split allied detector disc covers *unit*."""
        pos = getattr(unit, "position", None)
        if pos is None:
            return False
        ux, uy = float(pos.x), float(pos.y)
        for dx, dy, rng in self._allied_detector_discs:
            if (ux - dx) ** 2 + (uy - dy) ** 2 <= rng * rng:
                return True
        return False

    def _catalog_footprint_radius(self, unit: Unit) -> Optional[float]:
        """Footprint from python-sc2 GameData (creation ability), when available."""
        game_data = self._current_game_data
        if game_data is None:
            return None
        type_id = getattr(getattr(unit, "type_id", None), "value", None)
        if type_id is None:
            return None
        type_data = game_data.units.get(type_id)
        if type_data is None:
            return None
        footprint = type_data.footprint_radius
        if isinstance(footprint, (int, float)) and footprint > 0:
            return float(footprint)
        return None

    def _structure_game_radius(self, unit: Unit) -> float:
        """Catalog footprint for structures — never live API / selection ``radius``."""
        u_name = self._resolve_unit_type_name(unit)
        if u_name:
            data = self.fallback_db.get(u_name, {})
            fp = data.get("footprint_radius")
            if isinstance(fp, (int, float)) and fp > 0:
                return float(fp)
        catalog = self._catalog_footprint_radius(unit)
        if catalog is not None:
            return float(catalog)
        if u_name:
            default = _STRUCTURE_FOOTPRINT_RADIUS.get(u_name)
            if default is not None:
                return float(default)
        return 1.0

    def _unit_game_radius(self, unit: Unit) -> float:
        """Game-space half-extent for density/footprint layers (not live selection radius)."""
        if getattr(unit, "is_structure", False):
            return self._structure_game_radius(unit)
        u_name = self._resolve_unit_type_name(unit)
        if u_name and u_name in _UNIT_BODY_RADIUS_OVERRIDE:
            return float(_UNIT_BODY_RADIUS_OVERRIDE[u_name])
        if u_name:
            catalog = self._display_radius_db.get(u_name)
            if catalog is not None:
                return float(catalog)
        type_val = getattr(getattr(unit, "type_id", None), "value", getattr(unit, "type_id", None))
        if isinstance(type_val, int):
            catalog = self._display_radius_by_type_id.get(type_val)
            if catalog is not None:
                return float(catalog)
        return 0.375

    def _footprint_game_radius(self, unit: Unit) -> float:
        """Game radius for axis-aligned footprint bounds (structures + flagged mobile units)."""
        if getattr(unit, "is_structure", False):
            return self._structure_game_radius(unit)
        return self._unit_game_radius(unit)

    def _structure_half_extent_px(self, unit: Unit) -> int:
        game_radius = self._structure_game_radius(unit)
        ref_scale = min(self.feature_map_width, self.feature_map_height) / float(
            _ASSIGNMENT_REFERENCE_PLAYABLE
        )
        r_draw = int(round(game_radius * self.scale))
        r_ref = int(round(game_radius * ref_scale))
        return max(1, r_draw, r_ref)

    def _game_bounds_to_obs_rect(
        self,
        cx: float,
        cy: float,
        game_r: float,
    ) -> Tuple[int, int, int, int]:
        """Inclusive obs pixel bounds for an axis-aligned square in game space."""
        x0_g, x1_g = cx - game_r, cx + game_r
        y0_g, y1_g = cy - game_r, cy + game_r

        scale_x, scale_y = self._obs_scales()
        px0 = int(math.floor((x0_g - self.crop_x) * scale_x + self.offset_x))
        px1 = int(math.ceil((x1_g - self.crop_x) * scale_x + self.offset_x)) - 1

        content_y0 = (y0_g - self.crop_y) * scale_y + self.offset_y
        content_y1 = (y1_g - self.crop_y) * scale_y + self.offset_y
        py0 = int(self.feature_map_height - 1 - math.ceil(content_y1))
        py1 = int(self.feature_map_height - 1 - math.floor(content_y0))

        px0 = max(0, px0)
        px1 = min(self.feature_map_width - 1, px1)
        py0 = max(0, py0)
        py1 = min(self.feature_map_height - 1, py1)
        return px0, px1, py0, py1

    def _structure_obs_rect_bounds(self, unit: Unit) -> Tuple[int, int, int, int]:
        """Inclusive obs pixel bounds for a structure (parent + add-on slot when attached)."""
        bounds = [
            self._game_bounds_to_obs_rect(
                float(unit.position.x),
                float(unit.position.y),
                self._structure_game_radius(unit),
            )
        ]
        if self._structure_has_attached_addon(unit):
            ax, ay = self._addon_slot_position(unit)
            bounds.append(
                self._game_bounds_to_obs_rect(ax, ay, _ADDON_FOOTPRINT_RADIUS)
            )
        return self._merge_obs_rect_bounds(*bounds)

    def _footprint_half_extent_px(self, game_r: float) -> int:
        """Stable obs half-width from catalog game radius at the current map scale."""
        if game_r < 0.5:
            return 0
        scale_x, scale_y = self._obs_scales()
        max_scale = max(scale_x, scale_y, 1e-9)
        diameter = max(1, int(math.ceil(2.0 * game_r * max_scale - 1e-9)))
        return (diameter - 1) // 2

    def _center_stamp_obs_bounds(self, unit: Unit) -> Tuple[int, int, int, int]:
        """Center-anchored obs bounds — stable while a unit moves sub-tile."""
        u_name = self._resolve_unit_type_name(unit)
        game_r = self._unit_game_radius(unit)
        px, py = self._get_pixel_coords(unit.position)
        if px is None:
            return self._game_bounds_to_obs_rect(
                float(unit.position.x),
                float(unit.position.y),
                game_r,
            )
        if game_r < 0.5 or (u_name and u_name in _TACTICAL_SINGLE_PIXEL_UNITS):
            return px, px, py, py
        half = self._footprint_half_extent_px(game_r)
        return (
            max(0, px - half),
            min(self.feature_map_width - 1, px + half),
            max(0, py - half),
            min(self.feature_map_height - 1, py + half),
        )

    def _unit_fixed_footprint_cells(self, unit: Unit) -> Optional[int]:
        """N when the unit uses a fixed N×N obs hull; None for center-stamped footprints."""
        u_name = self._resolve_unit_type_name(unit)
        if not u_name:
            return None
        return _FIXED_FOOTPRINT_CELLS.get(u_name)

    def _tactical_siege_content_pos(self, unit: Unit) -> Tuple[float, float]:
        """Unit position in tactical content coordinates (matches _game_to_obs_pixel_raw)."""
        scale_x, scale_y = self._obs_scales()
        cx = (float(unit.position.x) - self.crop_x) * scale_x + self.offset_x
        cy = (float(unit.position.y) - self.crop_y) * scale_y + self.offset_y
        return cx, cy

    def _fixed_cell_footprint_obs_bounds(
        self, unit: Unit, cells: int
    ) -> Tuple[int, int, int, int]:
        """Fixed N×N obs hull centered on the unit's fractional position."""
        cx, cy = self._tactical_siege_content_pos(unit)
        span = cells - 1
        px0 = int(math.floor(cx - 0.5))
        px1 = px0 + span
        c_low = math.floor(cy - 0.5)
        c_high = c_low + span
        py_a = int(self.feature_map_height - 1 - math.floor(c_low))
        py_b = int(self.feature_map_height - 1 - math.floor(c_high))
        px0 = max(0, px0)
        px1 = min(self.feature_map_width - 1, px1)
        py0 = max(0, min(py_a, py_b))
        py1 = min(self.feature_map_height - 1, max(py_a, py_b))
        return px0, px1, py0, py1

    def _tactical_siege_obs_bounds(self, unit: Unit) -> Tuple[int, int, int, int]:
        """Fixed 2×2 tactical hull for sieged tanks."""
        return self._fixed_cell_footprint_obs_bounds(unit, _TACTICAL_SIEGE_FOOTPRINT_CELLS)

    def _tactical_siege_footprint_mask_in_rect(
        self,
        unit: Unit,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """True only on the fixed 2×2 siege hull inside an inclusive obs rect."""
        footprint = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
        sx0, sx1, sy0, sy1 = self._tactical_siege_obs_bounds(unit)
        rx0 = max(sx0, x0)
        rx1 = min(sx1, x1)
        ry0 = max(sy0, y0)
        ry1 = min(sy1, y1)
        if rx0 <= rx1 and ry0 <= ry1:
            footprint[
                ry0 - y0 : ry1 - y0 + 1,
                rx0 - x0 : rx1 - x0 + 1,
            ] = True
        return footprint

    def _tactical_siege_block_center_obs(self, unit: Unit) -> Tuple[float, float]:
        """Obs-pixel center for siege hull / dead zone (unit position, not block centroid)."""
        cx, cy = self._tactical_siege_content_pos(unit)
        return cx, float(self.feature_map_height - 1) - cy + 0.5

    def _tactical_siege_min_range_obs_px(self, min_radius_game: float) -> int:
        """Inner weapon dead-zone radius in obs pixels (tactical siege, block-centered)."""
        if min_radius_game <= 0:
            return 0
        scale_x, scale_y = self._obs_scales()
        max_scale = max(scale_x, scale_y, 1e-9)
        return max(
            int(math.ceil(float(min_radius_game) * max_scale - 1e-9)),
            int(math.ceil(self._obs_min_weapon_range_px(float(min_radius_game)))),
        )

    def _obs_chebyshev_mask_in_rect(
        self,
        center_px: float,
        center_py: float,
        radius_px: float,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """True where obs pixel centers fall within Chebyshev radius of a point."""
        px_idx = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
        py_idx = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(px_idx, py_idx)
        cheb = np.maximum(np.abs(gx - center_px), np.abs(gy - center_py))
        return cheb <= float(radius_px) + 1e-9

    def _tactical_siege_dead_ring_iterations(self, min_radius_game: float) -> int:
        """Chessboard dilation steps for a uniform dead ring outside the 2×2 hull."""
        inner_px = self._tactical_siege_min_range_obs_px(min_radius_game)
        # 2×2 hull is ±0.5 px from unit center; min_range inner_px clears ~1 px beyond hull.
        return max(1, inner_px - (_TACTICAL_SIEGE_FOOTPRINT_CELLS // 2))

    def _tactical_siege_no_dps_mask_in_rect(
        self,
        unit: Unit,
        min_radius_game: float,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """Hull plus a uniform pixel ring for tactical siege min_range (matches hull stamp)."""
        hull = self._tactical_siege_footprint_mask_in_rect(unit, x0, x1, y0, y1)
        if not np.any(hull):
            return hull
        ring_iters = self._tactical_siege_dead_ring_iterations(min_radius_game)
        structure = np.ones((3, 3), dtype=bool)
        return binary_dilation(hull, structure=structure, iterations=ring_iters)

    def _mobile_obs_rect_bounds(self, unit: Unit) -> Tuple[int, int, int, int]:
        """Inclusive obs bounds for a mobile unit from catalog game radius."""
        fixed_cells = self._unit_fixed_footprint_cells(unit)
        if fixed_cells is not None:
            return self._fixed_cell_footprint_obs_bounds(unit, fixed_cells)
        if self._unit_uses_circular_body(unit):
            if self.is_tactical_mode:
                return self._tactical_siege_obs_bounds(unit)
            game_r = self._unit_game_radius(unit)
            return self._game_bounds_to_obs_rect(
                float(unit.position.x),
                float(unit.position.y),
                game_r,
            )
        return self._center_stamp_obs_bounds(unit)

    def _unit_marker_obs_bounds(self, unit: Unit) -> Tuple[int, int, int, int]:
        """Obs footprint from catalog game radius (structures vs mobiles)."""
        if getattr(unit, "is_structure", False):
            return self._structure_obs_rect_bounds(unit)
        return self._mobile_obs_rect_bounds(unit)

    def _mobile_unit_footprint_mask(
        self,
        unit: Unit,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """Hull mask inside center-anchored stamp (circle for siege tank, filled square otherwise)."""
        shape = (y1 - y0 + 1, x1 - x0 + 1)
        u_name = self._resolve_unit_type_name(unit)
        game_r = self._unit_game_radius(unit)
        if self._unit_fixed_footprint_cells(unit) is not None:
            return np.ones(shape, dtype=bool)
        if game_r < 0.5 or (u_name and u_name in _TACTICAL_SINGLE_PIXEL_UNITS):
            return np.ones(shape, dtype=bool)
        if self._unit_uses_circular_body(unit):
            # Tactical: fixed 2×2 stamp (see _tactical_siege_obs_bounds). Global keeps circle.
            if self.is_tactical_mode:
                return np.ones(shape, dtype=bool)
            return self._mobile_body_footprint_mask(unit, x0, x1, y0, y1)
        return np.ones(shape, dtype=bool)

    def _tactical_stamp_obs_bounds(self, unit: Unit) -> Tuple[int, int, int, int]:
        """Tactical mobile bounds — game-anchored for circular hulls, pixel-centered otherwise."""
        return self._mobile_obs_rect_bounds(unit)

    def _tactical_single_pixel_obs_bounds(self, unit: Unit) -> Optional[Tuple[int, int, int, int]]:
        """One obs pixel at the unit center (tactical fovea)."""
        u_name = self._resolve_unit_type_name(unit)
        if not u_name or u_name not in _TACTICAL_SINGLE_PIXEL_UNITS:
            return None
        px, py = self._get_pixel_coords(unit.position)
        if px is None:
            return None
        return px, px, py, py

    def _unit_draw_obs_bounds(self, unit: Unit) -> Tuple[int, int, int, int]:
        if self.is_tactical_mode:
            single = self._tactical_single_pixel_obs_bounds(unit)
            if single is not None:
                return single
            if getattr(unit, "is_structure", False):
                return self._structure_obs_rect_bounds(unit)
            return self._tactical_stamp_obs_bounds(unit)
        return self._unit_marker_obs_bounds(unit)

    def _circular_footprint_mask_in_rect(
        self,
        cx: float,
        cy: float,
        game_r: float,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """Circular game-space footprint inside an inclusive obs-pixel rect."""
        region_x, region_y = self._game_center_meshgrid_in_rect(x0, x1, y0, y1)
        dx = region_x - float(cx)
        dy = region_y - float(cy)
        return (dx * dx + dy * dy) <= float(game_r) * float(game_r)

    def _unit_uses_circular_body(self, unit: Unit) -> bool:
        """Mobile units whose hull is a circle, not a tile-aligned square (siege mode tank)."""
        return self._resolve_unit_type_name(unit) == "SIEGETANKSIEGED"

    def _mobile_body_footprint_mask(
        self,
        unit: Unit,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """Circular hull stamp for mobile units (matches siege-tank DPS dead-zone center)."""
        game_r = self._unit_game_radius(unit)
        if game_r < 0.5:
            shape = (y1 - y0 + 1, x1 - x0 + 1)
            return np.zeros(shape, dtype=bool)
        return self._circular_footprint_mask_in_rect(
            float(unit.position.x),
            float(unit.position.y),
            game_r,
            x0,
            x1,
            y0,
            y1,
        )

    def _game_footprint_mask_in_rect(
        self,
        unit: Unit,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        center: Optional[Tuple[float, float]] = None,
        game_r: Optional[float] = None,
    ) -> np.ndarray:
        """True where obs samples fall inside a game-space square footprint."""
        X_map, Y_map = self._obs_to_sc2_meshgrid()
        region_x = X_map[y0:y1 + 1, x0:x1 + 1]
        region_y = Y_map[y0:y1 + 1, x0:x1 + 1]
        if center is None:
            cx, cy = float(unit.position.x), float(unit.position.y)
        else:
            cx, cy = center
        if game_r is None:
            game_r = self._footprint_game_radius(unit)
        return (
            (region_x >= cx - game_r)
            & (region_x <= cx + game_r)
            & (region_y >= cy - game_r)
            & (region_y <= cy + game_r)
        )

    def _pathing_unpathable_in_rect(
        self,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """All unpathable pathing cells whose obs samples fall inside an inclusive rect."""
        shape = (y1 - y0 + 1, x1 - x0 + 1)
        path_data = self._cached_path_data
        if path_data is None:
            return np.zeros(shape, dtype=bool)

        X_map, Y_map = self._obs_to_sc2_meshgrid()
        region_x = X_map[y0:y1 + 1, x0:x1 + 1]
        region_y = Y_map[y0:y1 + 1, x0:x1 + 1]

        path_h, path_w = path_data.shape
        clipped_x = np.clip(region_x, 0, path_w - 1)
        clipped_y = np.clip(region_y, 0, path_h - 1)
        return path_data[clipped_y, clipped_x] == 0

    def _structure_game_footprint_mask(
        self,
        unit: Unit,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """Unpathable pathing cells inside the structure's game-space footprint."""
        shape = (y1 - y0 + 1, x1 - x0 + 1)
        path_data = self._cached_path_data
        if path_data is None:
            return np.zeros(shape, dtype=bool)

        in_footprint = self._game_footprint_mask_in_rect(unit, x0, x1, y0, y1)
        if self._structure_has_attached_addon(unit):
            ax, ay = self._addon_slot_position(unit)
            addon_mask = self._game_footprint_mask_in_rect(
                unit,
                x0,
                x1,
                y0,
                y1,
                center=(ax, ay),
                game_r=_ADDON_FOOTPRINT_RADIUS,
            )
            in_footprint = in_footprint | addon_mask
            path_in_rect = self._pathing_unpathable_in_rect(x0, x1, y0, y1)
            if np.any(path_in_rect):
                # Fill the full L-shaped pathing hole inside the combined bounds.
                return path_in_rect
        X_map, Y_map = self._obs_to_sc2_meshgrid()
        region_x = X_map[y0:y1 + 1, x0:x1 + 1]
        region_y = Y_map[y0:y1 + 1, x0:x1 + 1]
        path_h, path_w = path_data.shape
        clipped_x = np.clip(region_x, 0, path_w - 1)
        clipped_y = np.clip(region_y, 0, path_h - 1)
        unpathable = path_data[clipped_y, clipped_x] == 0
        return in_footprint & unpathable

    def _density_footprint_mask(
        self,
        unit: Unit,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """Density-layer mask — full pathing hole for minerals/geysers/gas structures."""
        if self._unit_is_map_terrain_blocker(unit) or self._is_vespene_acquisition_structure(unit):
            path_mask = self._pathing_unpathable_in_rect(x0, x1, y0, y1)
            if np.any(path_mask):
                return path_mask
        return self._structure_game_footprint_mask(unit, x0, x1, y0, y1)

    def _terrain_overlay_mask(
        self,
        unit: Unit,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """Terrain-layer stamp — map blockers use the full pathing hole; buildings use game footprint."""
        if self._unit_is_map_terrain_blocker(unit):
            path_mask = self._pathing_unpathable_in_rect(x0, x1, y0, y1)
            if np.any(path_mask):
                return path_mask
            return np.ones((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
        return self._structure_game_footprint_mask(unit, x0, x1, y0, y1)

    def _unit_obs_half_extent(self, unit: Unit) -> int:
        """Half-width in obs pixels from unit center to footprint edge (scale-aware)."""
        px, py = self._get_pixel_coords(unit.position)
        if px is None:
            return 0
        x0, x1, y0, y1 = self._unit_marker_obs_bounds(unit)
        if x0 > x1 and y0 > y1:
            return 0
        half_x = max(px - x0, x1 - px, 0) if x0 <= x1 else 0
        half_y = max(py - y0, y1 - py, 0) if y0 <= y1 else 0
        return max(half_x, half_y)

    def _uses_additive_accumulation(self, idx: int) -> bool:
        for name, layer_idx in self.current_layer_map.items():
            if layer_idx == idx:
                return name in _ADDITIVE_UNIT_COUNTER_LAYERS
        return False

    def _accumulate_region(
        self,
        region: np.ndarray,
        mask: np.ndarray,
        value: float,
        accumulate: bool,
        additive: bool,
    ) -> None:
        if accumulate:
            if additive:
                region[mask] += value
            else:
                region[mask] = np.maximum(region[mask], value)
        else:
            region[mask] = value

    def _draw_entity_footprint(
        self,
        layers: np.ndarray,
        idx: int,
        unit: Unit,
        value: float,
        accumulate: bool,
        require_unpathable: bool,
    ) -> None:
        if idx not in self.active_indices:
            return

        x0, x1, y0, y1 = self._unit_draw_obs_bounds(unit)
        if x0 > x1 or y0 > y1:
            return

        if require_unpathable and not self.is_tactical_mode:
            mask = self._density_footprint_mask(unit, x0, x1, y0, y1)
            if not np.any(mask) and getattr(unit, "is_structure", False):
                # Pylons and other buildings on pathable tiles still need a density stamp.
                mask = self._game_footprint_mask_in_rect(unit, x0, x1, y0, y1)
        elif getattr(unit, "is_structure", False) and self.is_tactical_mode:
            mask = self._density_footprint_mask(unit, x0, x1, y0, y1)
            if not np.any(mask):
                mask = self._game_footprint_mask_in_rect(unit, x0, x1, y0, y1)
        else:
            mask = self._mobile_unit_footprint_mask(unit, x0, x1, y0, y1)
        if not np.any(mask):
            return

        region = layers[idx, y0:y1 + 1, x0:x1 + 1]
        self._accumulate_region(
            region,
            mask,
            value,
            accumulate,
            self._uses_additive_accumulation(idx),
        )

    def _draw_unit_marker(
        self,
        layers: np.ndarray,
        idx: Optional[int],
        unit: Unit,
        px: int,
        py: int,
        value: float,
        accumulate: bool = True,
    ) -> None:
        if idx is None:
            return
        additive = self._uses_additive_accumulation(idx)
        if unit.is_structure:
            self._draw_entity_footprint(
                layers,
                idx,
                unit,
                value,
                accumulate,
                require_unpathable=True,
            )
            return
        # Sub-half-tile units: one obs pixel at the position sample.
        if self._unit_game_radius(unit) < 0.5:
            self._draw_circle_vectorized(
                layers, idx, px, py, 0, 0, value, accumulate,
                additive_accumulation=additive,
            )
            return
        x0, x1, y0, y1 = self._unit_draw_obs_bounds(unit)
        if x0 > x1 or y0 > y1:
            self._draw_circle_vectorized(
                layers, idx, px, py, 0, 0, value, accumulate,
                additive_accumulation=additive,
            )
            return
        self._draw_entity_footprint(
            layers,
            idx,
            unit,
            value,
            accumulate,
            require_unpathable=False,
        )

    def _draw_unit_body(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int):
        prefix = "enemy" if is_enemy else "allied"
        idx_den = self.current_layer_map.get(f"{prefix}_unit_density")
        if idx_den is None:
            return

        val = 1.0
        cd = float(getattr(unit, "weapon_cooldown", 0.0))
        cd_norm = min(cd, 60.0) / 60.0
        if not is_enemy:
            if self.is_tactical_mode:
                val = 1.0
            else:
                val = 1.0 - (0.7 * cd_norm)

        self._draw_unit_marker(layers, idx_den, unit, px, py, val, True)

        # Sensor blips reveal position only — no unit-type attribute channels.
        if getattr(unit, "is_blip", False):
            return

        if self.is_tactical_mode:
            idx_cd = self.current_layer_map.get(
                LayerConstants.ENEMY_WEAPON_COOLDOWN if is_enemy else LayerConstants.ALLIED_WEAPON_COOLDOWN
            )
            if idx_cd is not None:
                self._draw_unit_marker(layers, idx_cd, unit, px, py, cd_norm, True)

        for attr in ["biological", "mechanical", "armored", "massive", "flying"]:
            idx = self.current_layer_map.get(f"{prefix}_unit_{attr}")
            if idx is not None and getattr(unit, f"is_{attr}", False):
                self._draw_unit_marker(layers, idx, unit, px, py, 1.0, True)

        idx_cloak = self.current_layer_map.get(f"{prefix}_unit_cloak")
        if idx_cloak is not None and self._unit_on_cloak_layer(unit):
            self._draw_unit_marker(layers, idx_cloak, unit, px, py, 1.0, True)

    def _command_objective_obs_radius_px(self) -> int:
        """
        Obs-pixel radius for COMMAND_OBJECTIVE.

        Global view: same reference footprint as on 128×128 curriculum maps regardless
        of playable area size (matches COMMAND_ASSIGNMENT / debug UI scaling).
        Tactical fovea: compact ~1 gu marker so objectives do not dominate 32×32 input.
        """
        if self.is_tactical_mode:
            return max(1, int(round(0.5 * self.scale)))
        ref_scale = min(self.feature_map_width, self.feature_map_height) / float(
            _ASSIGNMENT_REFERENCE_PLAYABLE
        )
        game_r = _OBJECTIVE_REFERENCE_RADIUS_GU
        r_draw = int(round(game_r * self.scale))
        r_ref = int(round(game_r * ref_scale))
        return max(1, r_draw, r_ref)

    def _command_objective_marker_params(self, is_beacon: bool) -> Tuple[float, int]:
        """Pixel radius and intensity for COMMAND_OBJECTIVE stamps."""
        val = 20.0 if is_beacon else 1.0
        return val, self._command_objective_obs_radius_px()

    def _draw_command_info(self, raw_obs, layers):
        idx_sel = self.current_layer_map.get(LayerConstants.COMMAND_ASSIGNMENT)
        if idx_sel is not None:
            units = raw_obs.get('selected_units') or raw_obs.get('units', [])
            for unit in units:
                if "BEACON" in getattr(unit.type_id, "name", "").upper(): continue
                px, py = self._get_pixel_coords(unit.position)
                if px is not None:
                    self._draw_unit_marker(layers, idx_sel, unit, px, py, 1.0, True)

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
                is_beacon = False
                u_id = getattr(target, "type_id", None)
                if u_id:
                    u_name = getattr(u_id, "name", str(u_id)).upper()
                    if "BEACON" in u_name:
                        is_beacon = True
                val, r = self._command_objective_marker_params(is_beacon)

                pos = getattr(target, "position", target)
                if self.is_tactical_mode and hasattr(pos, "x") and hasattr(pos, "y"):
                    if not self._tactical_game_point_inside_fovea(pos.x, pos.y):
                        continue
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


    def _playable_walkable_heights(
        self,
        path_data: np.ndarray,
        height_data: np.ndarray,
        game_info,
    ) -> np.ndarray:
        """Walkable height samples inside SC2 playable_area (falls back to full map)."""
        area = getattr(game_info, "playable_area", None)
        if area is None:
            walkable = path_data > 0
            return height_data[walkable].astype(np.float64)

        x0 = max(0, int(area.x))
        y0 = max(0, int(area.y))
        x1 = min(path_data.shape[1], x0 + int(area.width))
        y1 = min(path_data.shape[0], y0 + int(area.height))
        if x1 <= x0 or y1 <= y0:
            walkable = path_data > 0
            return height_data[walkable].astype(np.float64)

        region_path = path_data[y0:y1, x0:x1]
        region_height = height_data[y0:y1, x0:x1]
        playable_walkable = region_path > 0
        if not np.any(playable_walkable):
            walkable = path_data > 0
            return height_data[walkable].astype(np.float64)
        return region_height[playable_walkable].astype(np.float64)

    @staticmethod
    def _walkable_height_tier_indices(
        path_data: np.ndarray,
        height_data: np.ndarray,
        min_h: float,
        cap_h: float,
    ) -> np.ndarray:
        """Non-negative cliff-tier indices for walkable cells (outliers capped at ``cap_h``)."""
        walkable = path_data > 0
        if not np.any(walkable):
            return np.zeros(0, dtype=np.int32)
        capped = np.clip(height_data[walkable].astype(np.float64), float(min_h), float(cap_h))
        return np.round((capped - float(min_h)) / _SC2_TERRAIN_LEVEL_STEP).astype(np.int32)

    def _ensure_map_terrain_height_scale(
        self,
        path_data: np.ndarray,
        height_data: np.ndarray,
        map_name: str,
        game_info=None,
    ) -> Tuple[float, float]:
        """Per-map walkable height scale — stable for global cache and tactical panning."""
        if (
            self._map_terrain_h_min is not None
            and self._map_terrain_tier_max is not None
            and self._map_terrain_h_scale_map == map_name
        ):
            return float(self._map_terrain_h_min), float(self._map_terrain_tier_max)

        walkable_heights = self._playable_walkable_heights(
            path_data, height_data, game_info
        )
        if walkable_heights.size:
            min_h = float(
                np.percentile(walkable_heights, _TERRAIN_HEIGHT_PERCENTILE_FLOOR)
            )
            cap_h = float(
                np.percentile(walkable_heights, _TERRAIN_HEIGHT_PERCENTILE_CAP)
            )
        else:
            min_h = float(height_data.min())
            cap_h = float(height_data.max())

        tier_indices = self._walkable_height_tier_indices(
            path_data, height_data, min_h, cap_h
        )
        max_tier_idx = int(tier_indices.max()) if tier_indices.size else 0
        tier_max = float(max(max_tier_idx, 1))
        self._map_terrain_h_min = min_h
        self._map_terrain_tier_max = tier_max
        self._map_terrain_h_scale_map = map_name
        return min_h, tier_max

    @staticmethod
    def _normalize_terrain_heights(
        heights: np.ndarray,
        min_h: float,
        tier_max: float,
    ) -> np.ndarray:
        """Map raw SC2 heights to [0.2, 1.0] with a fixed step per cliff tier (16 gu)."""
        del tier_max  # retained for call-site compatibility; brightness step is tier-uniform
        tier_idx = (heights.astype(np.float32) - float(min_h)) / _SC2_TERRAIN_LEVEL_STEP
        tier_idx = np.maximum(tier_idx, 0.0)
        return np.clip(
            _TERRAIN_BRIGHTNESS_FLOOR + tier_idx * _TERRAIN_BRIGHTNESS_PER_CLIFF_TIER,
            _TERRAIN_BRIGHTNESS_FLOOR,
            1.0,
        ).astype(np.float32)

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
        # Use 8-connectivity so diagonal cliff runs stay intact (4-connectivity drops
        # corner-only neighbours along 45° cliff lines). Isolated single cells still
        # fail because none of their eight neighbours are climbable.
        from scipy.ndimage import binary_dilation
        nbr = np.ones((3, 3), dtype=bool)
        nbr[1, 1] = False
        return climbable & binary_dilation(climbable, structure=nbr)

    @staticmethod
    def _unit_is_map_terrain_blocker(unit: Unit) -> bool:
        if getattr(unit, "is_mineral_field", False) or getattr(unit, "is_vespene_geyser", False):
            return True
        name = getattr(getattr(unit, "type_id", None), "name", "").upper()
        return "MINERAL" in name or "VESPENEGEYSER" in name or "GEYSER" in name

    @staticmethod
    def _is_vespene_acquisition_structure(unit: Unit) -> bool:
        name = getattr(getattr(unit, "type_id", None), "name", "")
        if not name:
            return False
        return str(name).upper() in {
            "EXTRACTOR",
            "REFINERY",
            "ASSIMILATOR",
            "EXTRACTORRICH",
            "REFINERYRICH",
            "ASSIMILATORRICH",
        }

    def _geyser_covered_by_vespene_structure(
        self,
        geyser: Unit,
        structures: List[Unit],
    ) -> bool:
        """Built gas structures replace the neutral geyser terrain stamp."""
        gx = float(geyser.position.x)
        gy = float(geyser.position.y)
        for structure in structures:
            if not self._is_vespene_acquisition_structure(structure):
                continue
            if abs(float(structure.position.x) - gx) < 2.5 and abs(float(structure.position.y) - gy) < 2.5:
                return True
        return False

    def _collect_map_terrain_blockers(self, raw_obs: Dict[str, Any]) -> List[Unit]:
        """Minerals, geysers, and similar static map blockers for terrain stamping."""
        blockers: List[Unit] = []
        seen: set = set()
        all_structures = list(raw_obs.get("structures") or []) + list(raw_obs.get("enemy_structures") or [])

        def _add(unit: Unit) -> None:
            tag = getattr(unit, "tag", None)
            key = tag if tag is not None else id(unit)
            if key in seen:
                return
            seen.add(key)
            blockers.append(unit)

        for key in ("mineral_field", "vespene_geyser"):
            for unit in raw_obs.get(key) or []:
                if key == "vespene_geyser" and self._geyser_covered_by_vespene_structure(unit, all_structures):
                    continue
                _add(unit)

        for unit in list(raw_obs.get("structures") or []) + list(raw_obs.get("enemy_structures") or []):
            if self._unit_is_map_terrain_blocker(unit):
                _add(unit)

        return blockers

    def _collect_ground_structure_blockers(self, raw_obs: Dict[str, Any]) -> List[Unit]:
        """Ground structures that block pathing but are not minerals/geysers."""
        blockers: List[Unit] = []
        seen: set = set()
        for unit in list(raw_obs.get("structures") or []) + list(raw_obs.get("enemy_structures") or []):
            if getattr(unit, "is_flying", False):
                continue
            if not getattr(unit, "is_structure", False):
                continue
            if self._unit_is_map_terrain_blocker(unit):
                continue
            tag = getattr(unit, "tag", None)
            key = tag if tag is not None else id(unit)
            if key in seen:
                continue
            seen.add(key)
            blockers.append(unit)
        return blockers

    def _apply_map_blocker_terrain(self, layers: np.ndarray, raw_obs: Dict[str, Any]) -> None:
        layer_idx = self.current_layer_map.get(LayerConstants.TERRAIN_PATHABLE)
        if layer_idx is None:
            return
        blockers = (
            self._collect_map_terrain_blockers(raw_obs)
            + self._collect_ground_structure_blockers(raw_obs)
        )
        if blockers:
            self._overlay_map_blocker_terrain(layers[layer_idx], blockers)

    def _overlay_map_blocker_terrain(self, terrain: np.ndarray, blockers: List[Unit]) -> None:
        """Stamp minerals/geysers (and similar blockers) as unpathable on the terrain layer."""
        for unit in blockers:
            if getattr(unit, "is_flying", False):
                continue
            x0, x1, y0, y1 = self._structure_obs_rect_bounds(unit)
            if x0 > x1 or y0 > y1:
                continue
            mask = self._terrain_overlay_mask(unit, x0, x1, y0, y1)
            if not np.any(mask):
                continue
            region = terrain[y0:y1 + 1, x0:x1 + 1]
            region[mask] = 0.0

    def _overlay_ground_unit_terrain_blocks(self, terrain: np.ndarray, ground_units: List[Unit]) -> None:
        """Tactical only: stamp catalog footprints as unpathable (not cached global terrain)."""
        for unit in ground_units:
            if getattr(unit, "is_flying", False):
                continue
            # Burrowed mobiles do not block ground pathing in SC2.
            if not getattr(unit, "is_structure", False) and self._unit_is_burrowed(unit):
                continue
            if getattr(unit, "is_structure", False):
                x0, x1, y0, y1 = self._unit_draw_obs_bounds(unit)
                if x0 > x1 or y0 > y1:
                    continue
                mask = self._terrain_overlay_mask(unit, x0, x1, y0, y1)
            else:
                x0, x1, y0, y1 = self._tactical_stamp_obs_bounds(unit)
                if x0 > x1 or y0 > y1:
                    continue
                u_name = self._resolve_unit_type_name(unit)
                if (
                    self._unit_game_radius(unit) < 0.5
                    or (u_name and u_name in _TACTICAL_SINGLE_PIXEL_UNITS)
                ):
                    terrain[y0:y1 + 1, x0:x1 + 1] = 0.0
                    continue
                if self._unit_uses_circular_body(unit):
                    mask = self._tactical_siege_footprint_mask_in_rect(
                        unit, x0, x1, y0, y1
                    )
                else:
                    mask = np.ones((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
            if not np.any(mask):
                continue
            region = terrain[y0:y1 + 1, x0:x1 + 1]
            region[mask] = 0.0

    def _draw_terrain_pathing(
        self,
        game_info,
        layers,
        cliff_walker: bool = False,
        ground_units=None,
    ):
        layer_idx = self.current_layer_map.get(LayerConstants.TERRAIN_PATHABLE)
        if layer_idx is None:
            return

        current_map_name = getattr(game_info, 'map_name', '')
        cache_key = "cliff_walker" if cliff_walker else "standard"
        ground_units = ground_units or []

        # Global view: static map pathing only (cached per map + geometry; no live unit overlays).
        if not self.is_tactical_mode:
            terrain_cache_key = (
                cache_key,
                current_map_name,
                self._terrain_geometry_key(),
            )
            if terrain_cache_key in self._terrain_caches:
                layers[layer_idx] = self._terrain_caches[terrain_cache_key].copy()
                return

        if not hasattr(game_info, 'pathing_grid') or not hasattr(game_info, 'terrain_height'):
            return

        path_data = self._get_numpy_data(game_info.pathing_grid)
        height_data = self._get_numpy_data(game_info.terrain_height)
        path_h, path_w = path_data.shape

        if self._last_map_name != current_map_name:
            self._terrain_caches.clear()
            self._map_terrain_h_min = None
            self._map_terrain_tier_max = None
            self._map_terrain_h_scale_map = None

        min_h, tier_max = self._ensure_map_terrain_height_scale(
            path_data, height_data, current_map_name, game_info=game_info
        )

        X_map, Y_map = self._obs_to_sc2_meshgrid()
        X_map = np.clip(X_map, 0, path_w - 1)
        Y_map = np.clip(Y_map, 0, path_h - 1)
        valid_mask = self._obs_content_valid_mask()

        res = np.zeros(
            (self.feature_map_height, self.feature_map_width), dtype=np.float32
        )
        if np.any(valid_mask):
            v_x, v_y = X_map[valid_mask], Y_map[valid_mask]
            sampled_path = path_data[v_y, v_x]
            sampled_height = height_data[v_y, v_x]

            norm_height = self._normalize_terrain_heights(
                sampled_height, min_h, tier_max
            )

            if cliff_walker:
                if (
                    self._cliff_face_cache is None
                    or self._cliff_face_cache_map != current_map_name
                ):
                    self._cliff_face_cache = self._compute_climber_grid(
                        path_data, height_data
                    )
                    from scipy.ndimage import maximum_filter, minimum_filter

                    h_f = height_data.astype(np.float32)
                    large = float(h_f.max()) + 1.0
                    local_high_h = maximum_filter(
                        np.where(path_data > 0, h_f, 0.0), size=5
                    )
                    local_low_h = minimum_filter(
                        np.where(path_data > 0, h_f, large), size=5
                    )
                    high_obs = self._normalize_terrain_heights(
                        local_high_h, min_h, tier_max
                    )
                    low_obs = self._normalize_terrain_heights(
                        local_low_h, min_h, tier_max
                    )
                    self._cliff_display_heights = np.clip(
                        (high_obs + low_obs) * 0.5,
                        0.2,
                        1.0,
                    ).astype(np.float32)
                    self._cliff_face_cache_map = current_map_name

                sampled_cliff = self._cliff_face_cache[v_y, v_x]
                cliff_brightness = self._cliff_display_heights[v_y, v_x]
                standard_terrain = norm_height * (sampled_path > 0).astype(np.float32)
                res[valid_mask] = np.where(
                    sampled_cliff, cliff_brightness, standard_terrain
                )
            else:
                res[valid_mask] = norm_height * (sampled_path > 0).astype(np.float32)

        if self.is_tactical_mode and ground_units:
            visible_ground = [
                u for u in ground_units
                if self._tactical_footprint_intersects_fovea(u)
            ]
            self._overlay_ground_unit_terrain_blocks(res, visible_ground)

        layers[layer_idx] = res

        if not self.is_tactical_mode:
            self._terrain_caches[terrain_cache_key] = res.copy()
            self._last_map_name = current_map_name

    def _draw_terrain_creep(self, raw_obs, layers):
        idx = self.current_layer_map.get(LayerConstants.TERRAIN_CREEP)
        creep_map = raw_obs.get('creep')
        if idx is None or not creep_map: return
        data = self._get_numpy_data(creep_map)
        game_info = raw_obs.get('game_info')
        creep_h, creep_w = data.shape

        X_map, Y_map = self._obs_to_sc2_meshgrid()
        X_map = np.clip(X_map, 0, creep_w - 1)
        Y_map = np.clip(Y_map, 0, creep_h - 1)
        valid_mask = (X_map >= 0) & (Y_map >= 0)
        if np.any(valid_mask):
            layers[idx][valid_mask] = data[Y_map[valid_mask], X_map[valid_mask]]

    def _draw_terrain_buildable(self, raw_obs, layers):
        idx = self.current_layer_map.get(LayerConstants.TERRAIN_BUILDABLE)
        if idx is None or idx not in self.active_indices:
            return
        game_info = raw_obs.get("game_info")
        if game_info is None or not hasattr(game_info, "pathing_grid"):
            return

        from sc2.data import Race
        from src.transformers.buildability import compute_buildable_mask

        path_data = self._get_numpy_data(game_info.pathing_grid)
        creep_map = raw_obs.get("creep")
        creep_data = self._get_numpy_data(creep_map) if creep_map else None
        placement_data = None
        if hasattr(game_info, "placement_grid"):
            placement_data = self._get_numpy_data(game_info.placement_grid)
            if not isinstance(placement_data, np.ndarray):
                placement_data = None

        race_name = (raw_obs.get("race") or "zerg").lower()
        race = Race.Zerg
        if race_name == "terran":
            race = Race.Terran
        elif race_name == "protoss":
            race = Race.Protoss

        pending = raw_obs.get("pending_build_type")
        geyser_mask = raw_obs.get("geyser_mask")
        if geyser_mask is not None:
            geyser_mask = np.asarray(geyser_mask, dtype=np.float32)
        power_mask = raw_obs.get("power_mask")

        buildable_sc2 = compute_buildable_mask(
            race,
            pending,
            path_data,
            creep_data=creep_data,
            placement_data=placement_data,
            geyser_mask=geyser_mask,
            power_mask=power_mask,
        )

        X_map, Y_map = self._obs_to_sc2_meshgrid()
        creep_h, creep_w = buildable_sc2.shape
        X_map = np.clip(X_map, 0, creep_w - 1)
        Y_map = np.clip(Y_map, 0, creep_h - 1)
        valid_mask = self._obs_content_valid_mask()
        if np.any(valid_mask):
            layers[idx][valid_mask] = buildable_sc2[Y_map[valid_mask], X_map[valid_mask]]

    def _effective_dps_min_range_game(self, unit: Optional[Unit], min_radius_game: float) -> float:
        """Weapon min_range dead zone in game units (independent of hull footprint)."""
        if min_radius_game <= 0:
            return 0.0
        scale_x, scale_y = self._obs_scales()
        max_scale = max(scale_x, scale_y, 1e-9)
        # On coarse downsampling no pixel center may fall inside the true min_range disc;
        # expand inner hole so at least _obs_min_weapon_range_px around the unit is cleared.
        min_obs_px = self._obs_min_weapon_range_px(float(min_radius_game))
        return max(float(min_radius_game), min_obs_px / max_scale)

    def _obs_min_weapon_range_px(self, min_radius_game: float) -> float:
        """Minimum inner-hole radius in obs pixels when downsampling would hide min_range."""
        if min_radius_game <= 0:
            return 0.0
        scale_x, scale_y = self._obs_scales()
        max_scale = max(scale_x, scale_y, 1e-9)
        return max(float(min_radius_game) * max_scale, _MIN_WEAPON_MIN_RANGE_OBS_RADIUS_PX)

    def _min_observable_weapon_reach_game(self) -> float:
        """Smallest weapon reach (gu) that can still hit an obs pixel center from the unit position."""
        scale_x, scale_y = self._obs_scales()
        max_scale = max(scale_x, scale_y, 1e-9)
        half_cell = 0.5 / max_scale
        return math.hypot(half_cell, half_cell)

    def _apply_dps_footprint_exclusion(
        self,
        mask: np.ndarray,
        unit: Optional[Unit],
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
    ) -> np.ndarray:
        """Remove DPS inside the unit hull (circle for mobiles, square for structures)."""
        if unit is None or not np.any(mask):
            return mask
        x0, x1 = x_min, x_max - 1
        y0, y1 = y_min, y_max - 1
        if getattr(unit, "is_structure", False):
            footprint = self._game_footprint_mask_in_rect(unit, x0, x1, y0, y1)
        elif self._unit_uses_circular_body(unit):
            if self.is_tactical_mode:
                footprint = self._tactical_siege_footprint_mask_in_rect(
                    unit, x0, x1, y0, y1
                )
            else:
                footprint = self._mobile_body_footprint_mask(unit, x0, x1, y0, y1)
        else:
            footprint = self._game_footprint_mask_in_rect(unit, x0, x1, y0, y1)
        if footprint.shape != mask.shape:
            return mask
        return mask & ~footprint

    def _draw_circle_vectorized(
        self,
        layers,
        idx,
        cx,
        cy,
        radius_game,
        radius_px_override=None,
        value=1.0,
        accumulate=False,
        min_radius_game=0.0,
        min_radius_px_override=None,
        unit: Optional[Unit] = None,
        additive_accumulation: bool = True,
    ):
        if idx not in self.active_indices:
            return

        scale_x, scale_y = self._obs_scales()
        effective_min = self._effective_dps_min_range_game(unit, min_radius_game)
        # Game-unit circles (solid discs and weapon annuli) when no pixel override is given.
        if radius_px_override is None and float(radius_game) > 0:
            outer = float(radius_game)
            inner = float(effective_min) if effective_min > 0 else 0.0
            if outer <= inner:
                return
            center_gx, center_gy = self._circle_center_game(cx, cy, unit)
            # Off-fovea tactical DPS must not anchor draw rects to clamped edge pixels —
            # scan the full 32×32 window in game space so rings stay world-fixed as the fovea pans.
            if self.is_tactical_mode and unit is not None:
                x_min, y_min = 0, 0
                x_max = self.feature_map_width
                y_max = self.feature_map_height
            else:
                max_scale = max(scale_x, scale_y, 1e-9)
                r_px = max(1, int(math.ceil(outer * max_scale)))
                x_min, x_max = max(0, cx - r_px), min(self.feature_map_width, cx + r_px + 1)
                y_min, y_max = max(0, cy - r_px), min(self.feature_map_height, cy + r_px + 1)
            if x_min >= x_max or y_min >= y_max:
                return
            region_x, region_y = self._game_center_meshgrid_in_rect(
                x_min, x_max - 1, y_min, y_max - 1
            )
            dx = region_x - center_gx
            dy = region_y - center_gy
            dist_game_sq = dx * dx + dy * dy
            outer_sq = outer * outer
            if inner > 0:
                if (
                    self.is_tactical_mode
                    and unit is not None
                    and self._unit_uses_circular_body(unit)
                ):
                    dead = self._tactical_siege_no_dps_mask_in_rect(
                        unit,
                        min_radius_game,
                        x_min,
                        x_max - 1,
                        y_min,
                        y_max - 1,
                    )
                    mask = (dist_game_sq <= outer_sq) & ~dead
                else:
                    inner_sq = inner * inner
                    mask = (dist_game_sq <= outer_sq) & (dist_game_sq >= inner_sq)
                    mask = self._apply_dps_footprint_exclusion(
                        mask, unit, x_min, x_max, y_min, y_max
                    )
            else:
                mask = dist_game_sq <= outer_sq
            if not np.any(mask):
                return
            region = layers[idx, y_min:y_max, x_min:x_max]
            self._accumulate_region(region, mask, value, accumulate, additive_accumulation)
            return

        r = int(radius_px_override) if radius_px_override is not None else max(1, int(radius_game * self.scale))
        if min_radius_px_override is not None:
            min_r = int(min_radius_px_override)
        elif effective_min > 0:
            min_r = max(
                0,
                int(math.ceil(self._obs_min_weapon_range_px(effective_min))),
            )
        else:
            min_r = 0
        if r == 0:
            if accumulate:
                if additive_accumulation:
                    layers[idx, cy, cx] += value
                else:
                    layers[idx, cy, cx] = max(layers[idx, cy, cx], value)
            else:
                layers[idx, cy, cx] = value
            return
        if min_r >= r:
            return
        r_sq = r ** 2
        min_r_sq = min_r ** 2
        x_min, x_max = max(0, cx - r), min(self.feature_map_width, cx + r + 1)
        y_min, y_max = max(0, cy - r), min(self.feature_map_height, cy + r + 1)
        if x_min >= x_max or y_min >= y_max:
            return
        Y, X = self._grid_y[y_min:y_max, :], self._grid_x[:, x_min:x_max]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        if min_r > 0:
            mask = (dist_sq <= r_sq) & (dist_sq >= min_r_sq)
            mask = self._apply_dps_footprint_exclusion(mask, unit, x_min, x_max, y_min, y_max)
        else:
            mask = dist_sq <= r_sq
        region = layers[idx, y_min:y_max, x_min:x_max]
        self._accumulate_region(region, mask, value, accumulate, additive_accumulation)

    def _health_damage_signal(self, unit: Unit) -> float:
        """0 = undamaged, 1 = fully damaged (linear in missing HP+shields)."""
        max_hp = (unit.health_max + unit.shield_max) or 1
        hp_ratio = (unit.health + unit.shield) / max_hp
        return float(np.clip(1.0 - hp_ratio, 0.0, 1.0))

    def _draw_unit_vitals(self, unit, layers, is_enemy, px, py):
        prefix = "enemy" if is_enemy else "allied"
        idx_hp = self.current_layer_map.get(f"{prefix}_health")
        if idx_hp is not None:
            hp_signal = self._health_damage_signal(unit)
            self._draw_unit_marker(layers, idx_hp, unit, px, py, hp_signal, True)
        idx_nrg = self.current_layer_map.get(f"{prefix}_energy")
        if idx_nrg is not None and unit.energy_max > 0:
            self._draw_unit_marker(layers, idx_nrg, unit, px, py, unit.energy / unit.energy_max, True)

    def _draw_unit_detection(self, unit, layers, is_enemy, px, py):
        detect_range = self._unit_detection_range(unit)
        if detect_range <= 0:
            return
        prefix = "enemy" if is_enemy else "allied"
        idx = self.current_layer_map.get(f"{prefix}_detection")
        if idx is not None:
            self._draw_circle_vectorized(
                layers, idx, px, py, detect_range, None, 1.0, False
            )

    def _draw_unit_healing(self, unit, layers, is_enemy, px, py, game_data):
        prefix = "enemy" if is_enemy else "allied"
        idx = self.current_layer_map.get(f"{prefix}_hps_field")
        if idx is None: return
        hps, rng = 0.0, 0.0
        if unit.type_id == UnitTypeId.MEDIVAC and unit.energy > 5: hps, rng = 12.6, 4.0
        elif unit.type_id == UnitTypeId.SHIELDBATTERY and unit.energy > 3: hps, rng = 50.4, 6.0
        elif unit.type_id == UnitTypeId.QUEEN and unit.energy >= 50: hps, rng = 125.0, 7.0
        elif unit.type_id == UnitTypeId.SCV: hps, rng = 9.3, 1.0
        if hps > 0: self._draw_circle_vectorized(layers, idx, px, py, rng, None, hps, True)

    def _player_upgrade_ids(self, unit: Unit) -> frozenset:
        """Researched upgrades from the P1-client game state.

        CTF ``debug_upgrade`` is game-wide (both SC2 owners). Do not gate on
        proto ``is_mine``: after ``debug_control_enemy`` that flag is the P1
        client view, so a P2 trainee's allied DPS rings would miss range techs.
        Same contract as ``SquadObservationBuilder._bot_state_upgrades``.
        """
        bot = getattr(unit, "_bot_object", None)
        state = getattr(bot, "state", None) if bot is not None else None
        upgrades = getattr(state, "upgrades", None) if state is not None else None
        if not isinstance(upgrades, (set, frozenset, list, tuple)):
            return frozenset()
        return frozenset(upgrades)

    def _weapon_range_upgrade_deltas(self, unit: Unit) -> Tuple[float, float]:
        """
        Additive ground/air range bonuses from player upgrades.

        python-sc2 ``Unit.ground_range`` / ``air_range`` read static GameData and omit these
        (see ``Unit.calculate_damage_vs_target`` for the same upgrade table).
        Delegates to the shared ``weapon_range_upgrade_deltas`` helper so the same
        table is used by SquadObservationBuilder without importing this class.
        """
        from src.utils.unit_feature_helpers import weapon_range_upgrade_deltas
        return weapon_range_upgrade_deltas(unit, self._player_upgrade_ids(unit))

    def _dps_weapon_game_radius(self, unit: Unit, catalog_range: float) -> Tuple[float, float]:
        """
        Weapon reach in game units for DPS annuli: max(catalog range, live API range, upgraded range)
        + footprint.

        Uses the same display footprint radius as density/terrain markers (not the inflated live
        selection ``unit.radius``), so siege tanks and other large units stay concentric with their
        rendered body. Live ``ground_range`` / ``air_range`` are still consulted when they exceed
        catalog values; range upgrades are applied via ``_weapon_range_upgrade_deltas``.
        """
        ur = self._footprint_game_radius(unit)
        base = float(catalog_range)
        g_live = float(getattr(unit, "ground_range", 0.0) or 0.0)
        a_live = float(getattr(unit, "air_range", 0.0) or 0.0)
        g_up, a_up = self._weapon_range_upgrade_deltas(unit)
        min_reach = self._min_observable_weapon_reach_game()
        rng_g = max(max(base, g_live, base + g_up) + ur, min_reach)
        rng_a = max(max(base, a_live, base + a_up) + ur, min_reach)
        return rng_g, rng_a

    def _draw_unit_splash(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int):
        """Paint splash collateral reach (circle; sieged tank uses min_range donut)."""
        if not self._has_active_splash_layers(is_enemy):
            return
        if not self._unit_is_combat_ready(unit):
            return
        if getattr(unit, "is_hallucination", False):
            return
        spec = splash_spec(unit)
        if spec is None:
            return
        radius_gu, min_range_gu, splash_damage = spec
        layer_name = LayerConstants.ENEMY_SPLASH if is_enemy else LayerConstants.ALLIED_SPLASH
        idx = self.current_layer_map.get(layer_name)
        if idx is None or idx not in self.active_indices:
            return
        self._draw_circle_vectorized(
            layers, idx, px, py, float(radius_gu), None, float(splash_damage), True,
            min_radius_game=float(min_range_gu), unit=unit,
        )

    def _draw_unit_dps(self, unit: Unit, layers: np.ndarray, is_enemy: bool, px: int, py: int, game_data: GameData):
        if not self._unit_is_combat_ready(unit):
            return
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
        elif uid == UnitTypeId.ORACLE and unit.energy >= 25:
            weapon_stats = [{'base': 24.6, 'range': 4.0, 'typ': 1}]
        else:
            u_name = getattr(unit.type_id, "name", None)
            if u_name:
                db = self._get_unit_data_from_json(u_name)
                if db:
                    atk_lvl = int(getattr(unit, "attack_upgrade_level", 0) or 0)
                    for w in db.get('weapons', []):
                        period = w.get('period') or 1.0
                        attacks = float(w.get('attacks', 1) or 1)
                        upgrade_bonus = float(w.get('upgrade_bonus', 0) or 0)
                        dps = (w['damage'] * attacks) / period
                        dps += (upgrade_bonus * atk_lvl) / period
                        weapon_stats.append({
                            'base': dps,
                            'range': float(w['range']),
                            'min_range': float(w.get('min_range', 0.0)),
                            'typ': w['type'],
                            'period': period,
                            'attacks': attacks,
                            'bonuses': w.get('bonuses', []),
                        })

        # Weapon bonus type → typed-DPS layer suffix.
        # Values confirmed from unit_data.json samples:
        #   1=Light (Colossus, Hellion, Baneling), 2=Armored (Marauder, SiegeTank),
        #   7=Massive (Corruptor, Tempest).  Type 3=Biological has no typed layer.
        _BONUS_TYPE_SUFFIX = {1: "light", 2: "armor", 7: "massive"}

        for w in weapon_stats:
            p = "enemy" if is_enemy else "allied"
            if 'rng' in w and 'range' not in w:
                rng_g = float(w['rng'])
                rng_a = rng_g
            else:
                rng_g, rng_a = self._dps_weapon_game_radius(unit, float(w['range']))
            min_rng = float(w.get('min_range', 0.0))
            typ = int(w['typ'])
            idx_g = self._resolve_axis_dps_layer(p, "g") if typ in (1, 3) else None
            idx_a = self._resolve_axis_dps_layer(p, "a") if typ in (2, 3) else None
            # Bonus-only only when this axis's own _dps_base is active — not when base was
            # merely projected onto the other axis via cross-axis fallback.
            base_on_ground = self._primary_axis_dps_base_active(p, "g")
            base_on_air = self._primary_axis_dps_base_active(p, "a")
            if typ in (1, 3) and idx_g is not None:
                self._draw_circle_vectorized(
                    layers, idx_g, px, py, rng_g, None, w['base'], True,
                    min_radius_game=min_rng, unit=unit,
                )
            if typ in (2, 3) and idx_a is not None:
                self._draw_circle_vectorized(
                    layers, idx_a, px, py, rng_a, None, w['base'], True,
                    min_radius_game=min_rng, unit=unit,
                )

            # Typed DPS channels: bonus-only when _dps_base is active; otherwise base+bonus
            # (CTF uses typed layers without _dps_base — siege threat must still show as a donut).
            period = w.get('period', 1.0) or 1.0
            attacks = float(w.get('attacks', 1) or 1)
            for bonus in w.get('bonuses', []):
                raw_type = bonus.get('type')
                if hasattr(raw_type, "value"):
                    raw_type = raw_type.value
                try:
                    suffix = _BONUS_TYPE_SUFFIX.get(int(raw_type))
                except (TypeError, ValueError):
                    suffix = None
                if suffix is None:
                    continue
                # damage_bonus is per attack hit (Phoenix/Viking fire multiple attacks).
                bonus_dps = (float(bonus.get('bonus', 0.0) or 0.0) * attacks) / period
                if typ in (1, 3):
                    idx_gb = self._resolve_axis_dps_layer(p, "g", suffix)
                    if idx_gb is not None:
                        typed_dps = bonus_dps + (0.0 if base_on_ground else w['base'])
                        self._draw_circle_vectorized(
                            layers, idx_gb, px, py, rng_g, None, typed_dps, True,
                            min_radius_game=min_rng, unit=unit,
                        )
                if typ in (2, 3):
                    idx_aa = self._resolve_axis_dps_layer(p, "a", suffix)
                    if idx_aa is not None:
                        typed_dps = bonus_dps + (0.0 if base_on_air else w['base'])
                        self._draw_circle_vectorized(
                            layers, idx_aa, px, py, rng_a, None, typed_dps, True,
                            min_radius_game=min_rng, unit=unit,
                        )

    def _stamp_effect_splash(self, layers, is_ally, positions, radius, intensity):
        """Paint allied/enemy splash at effect positions (e.g. Parasitic Bomb cloud)."""
        is_enemy = not is_ally
        if not self._has_active_splash_layers(is_enemy):
            return
        layer_name = LayerConstants.ALLIED_SPLASH if is_ally else LayerConstants.ENEMY_SPLASH
        idx = self.current_layer_map.get(layer_name)
        if idx is None or idx not in self.active_indices:
            return
        for pos in positions:
            px, py = self._get_pixel_coords(pos)
            if px is not None:
                self._draw_circle_vectorized(
                    layers, idx, px, py, radius, None, intensity, True
                )

    @staticmethod
    def _nearest_caster_is_ally(positions, caster_positions, max_dist: float) -> Optional[bool]:
        """True/False when a caster is within max_dist of the effect; else None."""
        allied = list((caster_positions or {}).get("allied") or [])
        enemy = list((caster_positions or {}).get("enemy") or [])
        best_a = 999.0
        best_e = 999.0
        for pos in positions or []:
            dist_a = min([math.hypot(p.x - pos.x, p.y - pos.y) for p in allied] + [999.0])
            dist_e = min([math.hypot(p.x - pos.x, p.y - pos.y) for p in enemy] + [999.0])
            best_a = min(best_a, dist_a)
            best_e = min(best_e, dist_e)
        if min(best_a, best_e) >= max_dist:
            return None
        return best_a < best_e

    @staticmethod
    def _effect_proto_is_ally(effect) -> bool:
        """True when the SC2 effect proto marks this as Self/Ally (not MagicMock truthy)."""
        if getattr(effect, "is_mine", False) is True:
            return True
        if getattr(effect, "is_enemy", False) is True:
            return False
        alliance = getattr(effect, "alliance", None)
        if alliance is None:
            return False
        value = alliance if isinstance(alliance, int) and not isinstance(alliance, bool) else getattr(alliance, "value", None)
        try:
            return int(value) in (1, 2)
        except (TypeError, ValueError):
            name = str(getattr(alliance, "name", alliance)).upper()
            return name in ("SELF", "ALLY")

    def _resolve_threat_effect_owner(self, eid, effect) -> bool:
        """Allied vs enemy for bile / storm / nuke (nearest caster, else proto alliance)."""
        caster_map = None
        max_dist = 16.0
        if eid in _BILE_EFFECT_IDS:
            caster_map = self._ravager_positions
        elif eid in _STORM_EFFECT_IDS:
            caster_map = self._ht_positions
        elif eid in _NUKE_EFFECT_IDS:
            caster_map = self._ghost_positions
            max_dist = 128.0
        else:
            return False
        owner = self._nearest_caster_is_ally(effect.positions, caster_map, max_dist)
        if owner is not None:
            return owner
        # Proto alliance / is_mine is the P1-client view. After debug_control_enemy
        # that is not perspective-relative, so leftover storms/bile/nukes with no
        # live caster must not paint as allied.
        return False

    def _draw_effects(self, effects, layers):
        for effect in effects:
            eid = getattr(effect.id, "value", effect.id)
            fake_name = eid.upper() if isinstance(eid, str) else None
            dps, radius, hits_air, hits_ground, is_ally = 0.0, 0.0, False, True, False
            stamp_splash = False
            splash_intensity = 0.0
            if fake_name in ("PARASITICBOMB", "PARASITICBOMBDUMMY"):
                # python-sc2 converts PARASITICBOMBDUMMY → fake effect id "PARASITICBOMB".
                # Air-only, no friendly fire: always enemy DPS + splash (cloud follows the target).
                dps, radius = PARASITIC_BOMB_DPS, PARASITIC_BOMB_RADIUS_GU
                hits_air, hits_ground, is_ally = True, False, False
                stamp_splash = True
                splash_intensity = PARASITIC_BOMB_DAMAGE
            elif eid in _LIBERATOR_ZONE_EFFECT_IDS:
                # Concord Cannon is single-target; zone is a DPS threat disc, not splash.
                dps, radius = LIBERATOR_ZONE_DAMAGE, LIBERATOR_ZONE_RADIUS_GU
                owner = self._nearest_caster_is_ally(
                    effect.positions, self._liberator_positions, 16.0
                )
                is_ally = bool(owner)
            elif eid in _THREAT_EFFECT_SPECS:
                dps, radius, hits_air = _THREAT_EFFECT_SPECS[eid]
                is_ally = self._resolve_threat_effect_owner(eid, effect)
            elif eid in _SCAN_EFFECT_IDS:
                idx = self.current_layer_map.get(LayerConstants.ENEMY_DETECTION)
                if idx is not None:
                    for pos in effect.positions:
                        px, py = self._get_pixel_coords(pos)
                        if px is not None: self._draw_circle_vectorized(layers, idx, px, py, 13.0, None, 1.0, False)
                continue
            if dps > 0:
                p = "allied" if is_ally else "enemy"
                layer_indices = self._effect_dps_layer_indices(
                    p, hits_air=hits_air, hits_ground=hits_ground
                )
                for idx in layer_indices:
                    for pos in effect.positions:
                        px, py = self._get_pixel_coords(pos)
                        if px is not None:
                            self._draw_circle_vectorized(
                                layers, idx, px, py, radius, None, dps, True
                            )
                if stamp_splash:
                    self._stamp_effect_splash(
                        layers, is_ally, effect.positions, radius, splash_intensity
                    )

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
            cargo_max = int(getattr(unit, "cargo_max", 0) or 0)
            if cargo_max > 0:
                r = int(unit.radius * self.scale) + 1
                cargo_used = int(getattr(unit, "cargo_used", 0) or 0)
                if idx_slots:
                    self._draw_circle_vectorized(
                        layers, idx_slots, px, py, 0, r,
                        0.5 if cargo_used < cargo_max else 1.0, False,
                    )
                if idx_pass and cargo_used > 0:
                    self._draw_circle_vectorized(
                        layers, idx_pass, px, py, 0, r,
                        min(float(cargo_used) / 8.0, 1.0), True,
                    )

    def get_config(self):
        return {"map_size": (self.feature_map_width, self.feature_map_height), "scale": self.scale}
