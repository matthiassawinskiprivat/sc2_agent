"""Assembles squad pointer Dict observations for CTF."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit

from src.agent.squad_constants import (
    MAX_SQUAD_UNITS,
    SQUAD_COMMAND_DIM,
    SQUAD_RACE_DIM,
    encode_race_vector,
    include_race_enabled,
)
from src.agent.unit_scalar_constants import UnitScalarConstants as USC
from src.utils.perspective_transformer import PerspectiveTransformer
from src.utils.unit_feature_helpers import (
    is_force_melee_unit,
    weapon_range_upgrade_deltas,
)
from src.transformers.observation_transformer import ObservationTransformer

_F = USC.FEATURES
_MAX_RANGE = 10.0       # normalisation cap for weapon range (game units)
_MAX_SPEED = 6.0        # normalisation cap for movement speed
_MAX_HEALTH = 400.0     # tanh normalisation reference for max-health tankiness
_CD_FRAMES = 24.0       # approximate max weapon cooldown (frames)
_MELEE_RANGE_MAX = 1.5  # matches CaptureTheFlagBot._RANGED_WEAPON_RANGE_MIN


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Coerce to float, returning default for None or non-numeric types."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class SquadObservationBuilder:
    """Build global map + per-unit scalars + single hard-anchor tactical crop."""

    def __init__(self, training_conf: Dict[str, Any]):
        self.training_conf = training_conf
        self.active_layers: List[str] = list(
            training_conf.get("active_layers") or []
        )
        self.active_tactical_layers: List[str] = list(
            training_conf.get("active_tactical_layers") or self.active_layers
        )
        # Config order IS feature order (mirrors active_layers). Empty → full catalog.
        self.active_unit_scalars: List[str] = USC.resolve_active_features(
            training_conf.get("active_unit_scalars")
        )
        self.unit_feat_dim: int = len(self.active_unit_scalars)
        self._transformer = ObservationTransformer(
            map_size=(128, 128),
            training_conf={
                **training_conf,
                "active_layers": self.active_layers,
                "active_tactical_layers": self.active_tactical_layers,
                "agent_architecture": "universal",
            },
        )
        self._include_race = include_race_enabled(training_conf)
        self._global_obs_cache: Dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        # Static-per-type feature cache: cache_key → full catalog vector (USC.get_dim(),).
        # Active subset is gathered afterward so cache stays valid across configs.
        self._static_feat_cache: Dict[Any, np.ndarray] = {}

    def clear_global_cache(self) -> None:
        """Drop cached global map/command tensors (call once per env physics step)."""
        self._global_obs_cache.clear()
        self._static_feat_cache.clear()

    def zero_observation(self) -> Dict[str, np.ndarray]:
        n_g = len(self.active_layers) or 1
        n_t = len(self.active_tactical_layers) or n_g
        obs = {
            "map": np.zeros((n_g, 128, 128), dtype=np.float32),
            "command": np.zeros(SQUAD_COMMAND_DIM, dtype=np.float32),
            "anchor_tactical": np.zeros((n_t, 32, 32), dtype=np.float32),
            "unit_scalars": np.zeros(
                (MAX_SQUAD_UNITS, self.unit_feat_dim), dtype=np.float32
            ),
            "unit_mask": np.zeros(MAX_SQUAD_UNITS, dtype=np.float32),
        }
        if self._include_race:
            obs["race"] = np.zeros(SQUAD_RACE_DIM, dtype=np.float32)
        return obs

    @staticmethod
    def resolve_anchor_tag(
        sorted_tags: Sequence[int],
        alive_tags: Sequence[int],
        preferred_tag: Optional[int] = None,
    ) -> Optional[int]:
        """Pick hard-anchor tag: preferred if alive, else first sorted alive."""
        alive = {int(t) for t in alive_tags}
        ordered = [int(t) for t in sorted_tags if int(t) in alive]
        if not ordered:
            return None
        if preferred_tag is not None and int(preferred_tag) in alive:
            return int(preferred_tag)
        return ordered[0]

    def build(
        self,
        bot,
        *,
        perspective_player_id: int,
        squad_unit_tags: Sequence[int],
        pool_high_level_command: str,
        playable_size: tuple[float, float] = (256.0, 256.0),
        anchor_unit_tag: Optional[int] = None,
        own_race: Optional[str] = None,
        enemy_race: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        sorted_tags = [int(t) for t in sorted(squad_unit_tags)]
        alive_tags: List[int] = []
        for tag in sorted_tags[:MAX_SQUAD_UNITS]:
            if bot.all_units.find_by_tag(tag) is not None:
                alive_tags.append(tag)
        resolved_anchor = self.resolve_anchor_tag(
            sorted_tags, alive_tags, preferred_tag=anchor_unit_tag
        )
        raw_obs = self._build_raw_obs(
            bot,
            perspective_player_id=perspective_player_id,
            sorted_tags=sorted_tags,
            pool_high_level_command=pool_high_level_command,
            anchor_unit_tag=resolved_anchor,
        )
        cache_key = (int(perspective_player_id), str(pool_high_level_command).lower())
        cached = self._global_obs_cache.get(cache_key)
        if cached is None:
            global_result = self._transformer.transform(raw_obs)
            if not isinstance(global_result, dict):
                raise TypeError("SquadObservationBuilder requires universal dict transform output")
            map_obs = np.asarray(global_result["map"], dtype=np.float32)
            command_obs = np.asarray(global_result["command"], dtype=np.float32)
            if command_obs.shape[0] != SQUAD_COMMAND_DIM:
                padded = np.zeros(SQUAD_COMMAND_DIM, dtype=np.float32)
                n = min(SQUAD_COMMAND_DIM, command_obs.shape[0])
                padded[:n] = command_obs[:n]
                command_obs = padded
            self._global_obs_cache[cache_key] = (map_obs, command_obs)
        else:
            map_obs, command_obs = cached

        pw, ph = playable_size

        # --- Contextual quantities used by relational features ---
        # Own-side upgrades (only available for the perspective player)
        upgrades: frozenset = frozenset()
        if perspective_player_id == bot.player_id:
            raw_upg = getattr(getattr(bot, "state", None), "upgrades", None)
            if isinstance(raw_upg, (set, frozenset, list, tuple)):
                upgrades = frozenset(raw_upg)

        # Enemy center-of-mass (perspective-correct enemy_units)
        if perspective_player_id == bot.player_id:
            vis_enemies = bot.enemy_units
        else:
            vis_enemies = bot.units
        enemy_com = _center_of_mass(vis_enemies)

        # Allied squad center-of-mass from live unit positions
        squad_positions = []
        for tag in sorted_tags[:MAX_SQUAD_UNITS]:
            u = bot.all_units.find_by_tag(tag)
            if u is not None:
                squad_positions.append((u.position.x, u.position.y))
        allied_com = (
            (
                sum(p[0] for p in squad_positions) / len(squad_positions),
                sum(p[1] for p in squad_positions) / len(squad_positions),
            )
            if squad_positions
            else None
        )

        n_t = len(self.active_tactical_layers) or map_obs.shape[0]
        anchor_tactical = np.zeros((n_t, 32, 32), dtype=np.float32)
        unit_scalars = np.zeros((MAX_SQUAD_UNITS, self.unit_feat_dim), dtype=np.float32)
        unit_mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)

        for slot, tag in enumerate(sorted_tags[:MAX_SQUAD_UNITS]):
            unit = bot.all_units.find_by_tag(tag)
            if unit is None:
                continue
            unit_mask[slot] = 1.0
            objective = bot.unit_target_map.get(tag)
            unit_scalars[slot] = self._unit_scalars(
                unit,
                pool_high_level_command,
                playable_w=pw,
                playable_h=ph,
                upgrades=upgrades,
                objective=objective,
                enemy_com=enemy_com,
                allied_com=allied_com,
            )

        # Single hard-anchor crop (Option A): O(1) vs previous O(N) transform_tactical.
        if resolved_anchor is not None:
            anchor_unit = bot.all_units.find_by_tag(resolved_anchor)
            if anchor_unit is not None:
                tactical = self._transformer.transform_tactical(raw_obs, anchor_unit)
                anchor_tactical = np.asarray(tactical, dtype=np.float32)

        obs = {
            "map": map_obs,
            "command": command_obs,
            "anchor_tactical": anchor_tactical,
            "unit_scalars": unit_scalars,
            "unit_mask": unit_mask,
        }
        if self._include_race:
            obs["race"] = encode_race_vector(own_race, enemy_race)
        if perspective_player_id == 2:
            obs = self._flip_obs(obs)
        return obs

    # ------------------------------------------------------------------
    # Static (per-type) feature helpers
    # ------------------------------------------------------------------

    def _unit_static_feats(self, unit: Unit, upgrades: frozenset) -> np.ndarray:
        """Compute Groups C+D features that are constant per unit type within an episode.

        Cached by (type_id, upgrades_hash) so the lookup is O(1) after the first call
        for each unit type.  Cache is cleared at the start of each physics step via
        clear_global_cache().
        """
        type_key = getattr(unit.type_id, "value", unit.type_id)
        cache_key = (type_key, upgrades)
        cached = self._static_feat_cache.get(cache_key)
        if cached is not None:
            return cached

        feats = np.zeros(USC.get_dim(), dtype=np.float32)

        # --- Group C: Static combat profile ---
        g_delta, a_delta = weapon_range_upgrade_deltas(unit, upgrades)
        g_range = _safe_float(getattr(unit, "ground_range", 0.0)) + g_delta
        a_range = _safe_float(getattr(unit, "air_range", 0.0)) + a_delta
        g_dps   = _safe_float(getattr(unit, "ground_dps", 0.0))
        feats[_F[USC.GROUND_RANGE_NORM]] = min(g_range / _MAX_RANGE, 1.0)
        feats[_F[USC.AIR_RANGE_NORM]]    = min(a_range / _MAX_RANGE, 1.0)
        feats[_F[USC.GROUND_DPS_NORM]]   = float(np.tanh(g_dps / 20.0))
        feats[_F[USC.CAN_ATTACK_AIR]]    = float(_safe_float(getattr(unit, "air_dps", 0.0)) > 0.0)
        feats[_F[USC.IS_MELEE]]          = float(
            is_force_melee_unit(unit) or g_range < _MELEE_RANGE_MAX
        )
        h_max  = _safe_float(getattr(unit, "health_max", 0.0))
        sh_max = _safe_float(getattr(unit, "shield_max", 0.0))
        feats[_F[USC.MAX_HEALTH_NORM]] = float(np.tanh((h_max + sh_max) / _MAX_HEALTH))

        # --- Group D: Unit type attribute flags ---
        feats[_F[USC.IS_BIOLOGICAL]] = float(bool(getattr(unit, "is_biological", False)))
        feats[_F[USC.IS_MECHANICAL]] = float(bool(getattr(unit, "is_mechanical", False)))
        feats[_F[USC.IS_ARMORED]]    = float(bool(getattr(unit, "is_armored", False)))
        feats[_F[USC.IS_MASSIVE]]    = float(bool(getattr(unit, "is_massive", False)))
        feats[_F[USC.IS_FLYING]]     = float(bool(getattr(unit, "is_flying", False)))
        feats[_F[USC.HAS_SHIELD]]    = float(sh_max > 0.0)

        self._static_feat_cache[cache_key] = feats
        return feats

    # ------------------------------------------------------------------
    # Per-step scalar builder
    # ------------------------------------------------------------------

    def _unit_scalars(
        self,
        unit: Unit,
        pool_command: str,
        *,
        playable_w: float,
        playable_h: float,
        upgrades: frozenset,
        objective,
        enemy_com,
        allied_com,
    ) -> np.ndarray:
        # Fill the full catalog buffer (catalog index order), then gather into
        # active_unit_scalars config order — same pattern as active_layers slicing.
        catalog = self._unit_static_feats(unit, upgrades).copy()

        # --- Group A: Vitals ---
        h     = _safe_float(unit.health)
        h_max = max(_safe_float(getattr(unit, "health_max", 1.0)), 1e-3)
        sh    = _safe_float(getattr(unit, "shield", 0.0))
        sh_max = max(_safe_float(getattr(unit, "shield_max", 0.0)), 1e-3)
        catalog[_F[USC.HEALTH_PCT]]        = h / h_max
        catalog[_F[USC.SHIELD_PCT]]        = sh / sh_max if sh_max > 1e-3 else 0.0
        catalog[_F[USC.COMBINED_VITALITY]] = (h + sh) / (h_max + sh_max)

        raw_cd = _safe_float(getattr(unit, "weapon_cooldown", -1.0), default=-1.0)
        norm_cd = max(0.0, raw_cd) / _CD_FRAMES if raw_cd > 0.0 else 0.0
        catalog[_F[USC.WEAPON_COOLDOWN_NORM]] = min(norm_cd, 1.0)
        catalog[_F[USC.WEAPON_READY]]          = float(bool(getattr(unit, "weapon_ready", True)))

        # --- Group B: Kiting / mobility ---
        try:
            speed = float(unit.calculate_speed(upgrades)) if upgrades else _safe_float(
                getattr(unit, "movement_speed", 0.0)
            )
        except Exception:
            speed = 0.0
        catalog[_F[USC.MOVEMENT_SPEED_NORM]] = float(np.tanh(speed / _MAX_SPEED))

        try:
            dwr = _safe_float(unit.distance_to_weapon_ready)
        except Exception:
            dwr = 0.0
        catalog[_F[USC.DIST_TO_WEAPON_READY_NORM]] = min(dwr / max(playable_w, 1.0), 1.0)

        e_max = max(_safe_float(getattr(unit, "energy_max", 0.0)), 1e-3)
        e     = _safe_float(getattr(unit, "energy", 0.0))
        catalog[_F[USC.ENERGY_PCT]] = (e / e_max) if e_max > 1e-3 else 0.0

        # --- Group E: Positional / relational ---
        px = float(unit.position.x)
        py = float(unit.position.y)
        catalog[_F[USC.X_NORM]] = px / max(playable_w, 1.0)
        catalog[_F[USC.Y_NORM]] = py / max(playable_h, 1.0)

        if objective is not None:
            ox = float(getattr(objective, "x", getattr(objective, "_proto", None) and
                               getattr(objective._proto, "x", px) or px))
            oy = float(getattr(objective, "y", getattr(objective, "_proto", None) and
                               getattr(objective._proto, "y", py) or py))
            d_obj = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
            catalog[_F[USC.DIST_TO_OBJECTIVE_NORM]] = min(d_obj / max(playable_w, 1.0), 1.0)
        else:
            catalog[_F[USC.DIST_TO_OBJECTIVE_NORM]] = 1.0  # unknown → max distance

        if enemy_com is not None:
            ex, ey = enemy_com
            d_en = math.sqrt((px - ex) ** 2 + (py - ey) ** 2)
            catalog[_F[USC.DIST_TO_ENEMY_COM_NORM]] = min(d_en / max(playable_w, 1.0), 1.0)

        if allied_com is not None:
            ax, ay = allied_com
            d_al = math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
            catalog[_F[USC.DIST_TO_ALLIED_COM_NORM]] = min(d_al / max(playable_w, 1.0), 1.0)

        # --- Group F: Engagement / concealment state ---
        catalog[_F[USC.IS_ATTACKING]] = float(bool(getattr(unit, "is_attacking", False)))
        catalog[_F[USC.IS_MOVING]]    = float(bool(getattr(unit, "is_moving", False)))
        is_burrowed = bool(getattr(unit, "is_burrowed", False))
        is_cloaked  = bool(getattr(unit, "is_cloaked", False))
        catalog[_F[USC.IS_CONCEALED]] = float(is_burrowed or is_cloaked)
        catalog[_F[USC.IS_DETECTED]]  = float(bool(getattr(unit, "is_detected", False)))

        # --- Group G: Role ---
        role = str(pool_command).lower()
        catalog[_F[USC.ROLE_ATTACK]] = 1.0 if role == "attack" else 0.0
        catalog[_F[USC.ROLE_DEFEND]] = 1.0 if role == "defend" else 0.0

        return self._gather_active(catalog)

    def _gather_active(self, catalog: np.ndarray) -> np.ndarray:
        """Slice the full catalog into active_unit_scalars config order."""
        out = np.zeros(self.unit_feat_dim, dtype=np.float32)
        for i, name in enumerate(self.active_unit_scalars):
            out[i] = catalog[_F[name]]
        return out

    # ------------------------------------------------------------------
    # Perspective flip
    # ------------------------------------------------------------------

    def _flip_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Flip spatial maps for player-2 perspective; leave scalars/mask unchanged."""
        flipped = dict(obs)
        for key in ("map", "anchor_tactical"):
            val = flipped.get(key)
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                flipped[key] = PerspectiveTransformer.flip_observation(val)
        return flipped

    # ------------------------------------------------------------------
    # Raw obs assembly
    # ------------------------------------------------------------------

    def _build_raw_obs(
        self,
        bot,
        *,
        perspective_player_id: int,
        sorted_tags: List[int],
        pool_high_level_command: str,
        anchor_unit_tag: Optional[int] = None,
    ) -> Dict[str, Any]:
        p_id = perspective_player_id
        if p_id == bot.player_id:
            units = bot.units
            enemy_units = bot.enemy_units
            structures = bot.structures
            enemy_structures = bot.enemy_structures
        else:
            units = bot.enemy_units
            enemy_units = bot.units
            structures = bot.enemy_structures
            enemy_structures = bot.structures

        active_unit: Optional[Unit] = None
        selected_units: List[Unit] = []
        for tag in sorted_tags[:MAX_SQUAD_UNITS]:
            unit = bot.all_units.find_by_tag(tag)
            if unit is None:
                continue
            selected_units.append(unit)
        if anchor_unit_tag is not None:
            active_unit = bot.all_units.find_by_tag(int(anchor_unit_tag))
        if active_unit is None and selected_units:
            active_unit = selected_units[0]
        if active_unit is None and units.exists:
            active_unit = units.first
            selected_units = [active_unit]

        target_pos = None
        if active_unit is not None and active_unit.tag in bot.unit_target_map:
            target_pos = bot.unit_target_map[active_unit.tag]
        elif active_unit is not None:
            role = bot.unit_command_map.get(active_unit.tag, pool_high_level_command)
            if role == "defend":
                own_pylon = (
                    structures(UnitTypeId.PYLON).first
                    if structures(UnitTypeId.PYLON).exists
                    else None
                )
                target_pos = own_pylon.position if own_pylon else None
            else:
                target_pos = bot.enemy_pylon_pos.get(p_id)

        from src.utils.command_constants import CommandConstants

        high_cmd = CommandConstants.get_index(pool_high_level_command)

        return {
            "units": units,
            "enemy_units": enemy_units,
            "enemy_structures": enemy_structures,
            "structures": structures,
            "effects": getattr(bot.state, "effects", []),
            "game_info": bot.game_info,
            "game_data": bot.game_data,
            "creep": getattr(bot.state, "creep", None),
            "command_target": target_pos,
            "selected_units": selected_units,
            "high_level_command": high_cmd,
        }


# ------------------------------------------------------------------
# Module-level utility
# ------------------------------------------------------------------

def _center_of_mass(units_collection) -> Optional[tuple[float, float]]:
    """Return the (x, y) centroid of a python-sc2 Units collection, or None."""
    try:
        positions = [(u.position.x, u.position.y) for u in units_collection]
    except Exception:
        return None
    if not positions:
        return None
    n = len(positions)
    return (sum(p[0] for p in positions) / n, sum(p[1] for p in positions) / n)
