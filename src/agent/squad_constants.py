"""Constants for the CTF squad pointer training stack."""

import numpy as np

from src.agent.universal_constants import (
    FIXED_UNIVERSAL_COMMAND_DIM,
    UNUSED_COMMAND_HEAD,
)
from src.agent.unit_scalar_constants import UnitScalarConstants

MAX_SQUAD_UNITS = 24
# Master catalog size (fallback). Live obs dim is len(active_unit_scalars).
SQUAD_UNIT_FEAT_DIM = UnitScalarConstants.get_dim()
SQUAD_COMMAND_DIM = 10
# Own + enemy race one-hots (terran/zerg/protoss each); default on, opt out via include_race: false.
SQUAD_RACE_DIM = 6
SQUAD_RACE_ORDER = ("terran", "zerg", "protoss")
# Own + enemy armor/shield upgrade cache (ground, air, shields each); gated by include_upgrades.
SQUAD_UPGRADES_DIM = 6
SQUAD_SPATIAL_RESOLUTION = 32
SQUAD_SPATIAL_CELLS = SQUAD_SPATIAL_RESOLUTION * SQUAD_SPATIAL_RESOLUTION
SQUAD_POINTER_DIM = MAX_SQUAD_UNITS

# --- Unit-target pointer (AlphaStar-style terminal target) -------------------
# One padded target slot: allied squad slots, then visible-enemy slots, then a
# trailing NULL slot used by spatial/categorical commands.
MAX_SQUAD_ENEMIES = 24
SQUAD_TARGET_SLOTS = MAX_SQUAD_UNITS + MAX_SQUAD_ENEMIES + 1
SQUAD_NULL_TARGET_SLOT = SQUAD_TARGET_SLOTS - 1

# --- Flat action-mask layout ------------------------------------------------
# [actor bits (2 per slot) | command | target slot | spatial cell]
SQUAD_MASK_ACTOR_OFFSET = 0
SQUAD_MASK_COMMAND_OFFSET = MAX_SQUAD_UNITS * 2
SQUAD_MASK_TARGET_OFFSET = SQUAD_MASK_COMMAND_OFFSET + FIXED_UNIVERSAL_COMMAND_DIM
SQUAD_MASK_SPATIAL_OFFSET = SQUAD_MASK_TARGET_OFFSET + SQUAD_TARGET_SLOTS
SQUAD_FLAT_ACTION_DIM = SQUAD_MASK_SPATIAL_OFFSET + SQUAD_SPATIAL_CELLS

# --- Sampled action vector layout ------------------------------------------
# [actor bit per slot | command | target slot | spatial cell]
SQUAD_ACTION_COMMAND_IDX = MAX_SQUAD_UNITS
SQUAD_ACTION_TARGET_IDX = MAX_SQUAD_UNITS + 1
SQUAD_ACTION_SPATIAL_IDX = MAX_SQUAD_UNITS + 2
SQUAD_ACTION_DIM = MAX_SQUAD_UNITS + 3

# --- Command target modes ---------------------------------------------------
# How the terminal target of a command is expressed. The policy samples the
# command first, then narrows actor/target/spatial legality with this mode.
CMD_MODE_SPATIAL = 0       # world cell (move)
CMD_MODE_ALLY_TARGET = 1   # allied unit pointer (heal_unit)
CMD_MODE_ENEMY_TARGET = 2  # hostile unit pointer (attack_unit)
CMD_MODE_NONE = 3          # no target at all (padding / unknown categoricals)
CMD_MODE_LOAD_TARGET = 4   # allied unit pointer (load_unit: transport↔passenger)
CMD_MODE_AOE = 5           # world cell; only ready AoE casters act
CMD_MODE_UNLOAD = 6        # world cell; only loaded transports act
CMD_MODE_CLOAK = 7         # no target; only units that can conceal act
CMD_MODE_UNCLOAK = 8       # no target; only currently concealed units act
CMD_MODE_ATTACK_LOCAL = 9  # world cell; weapon-ready actors + enemy in tactical fovea
SQUAD_COMMAND_MODE_DIM = FIXED_UNIVERSAL_COMMAND_DIM
# Inclusive upper bound for the ``command_modes`` observation Box.
SQUAD_COMMAND_MODE_MAX = float(CMD_MODE_ATTACK_LOCAL)

# Head names whose terminal target is a unit pointer rather than a world cell.
ALLY_TARGET_COMMAND_HEADS = ("heal_unit",)
LOAD_TARGET_COMMAND_HEADS = ("load_unit",)
ENEMY_TARGET_COMMAND_HEADS = ("attack_unit",)
AOE_COMMAND_HEADS = ("aoe_local",)
UNLOAD_COMMAND_HEADS = ("unload_local",)
CLOAK_COMMAND_HEADS = ("cloak_burrow",)
UNCLOAK_COMMAND_HEADS = ("uncloak_unburrow",)
ATTACK_LOCAL_COMMAND_HEADS = ("attack_local",)

# Modes that decode a world spatial cell (vs unit pointer / null).
SPATIAL_COMMAND_MODES = frozenset(
    {CMD_MODE_SPATIAL, CMD_MODE_AOE, CMD_MODE_UNLOAD, CMD_MODE_ATTACK_LOCAL}
)

DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_MIN_SELECTED_UNITS = 1


def command_target_mode(head_name: str, *, is_categorical: bool = False) -> int:
    """Target mode for one command head name."""
    name = str(head_name or "").lower().strip()
    if not name or name == UNUSED_COMMAND_HEAD:
        return CMD_MODE_NONE
    if name in CLOAK_COMMAND_HEADS:
        return CMD_MODE_CLOAK
    if name in UNCLOAK_COMMAND_HEADS:
        return CMD_MODE_UNCLOAK
    if is_categorical:
        return CMD_MODE_NONE
    if name in ALLY_TARGET_COMMAND_HEADS:
        return CMD_MODE_ALLY_TARGET
    if name in LOAD_TARGET_COMMAND_HEADS:
        return CMD_MODE_LOAD_TARGET
    if name in ENEMY_TARGET_COMMAND_HEADS:
        return CMD_MODE_ENEMY_TARGET
    if name in ATTACK_LOCAL_COMMAND_HEADS:
        return CMD_MODE_ATTACK_LOCAL
    if name in AOE_COMMAND_HEADS:
        return CMD_MODE_AOE
    if name in UNLOAD_COMMAND_HEADS:
        return CMD_MODE_UNLOAD
    return CMD_MODE_SPATIAL


def command_target_modes(
    action_heads, categorical_heads=None
) -> np.ndarray:
    """Length-``FIXED_UNIVERSAL_COMMAND_DIM`` mode vector in command-index order.

    Rides in the squad observation (`command_modes`) so the learner narrows
    actor/target/spatial legality with exactly the table the collector used.
    """
    modes = np.full(FIXED_UNIVERSAL_COMMAND_DIM, float(CMD_MODE_NONE), dtype=np.float32)
    spatial = [str(h).lower() for h in (action_heads or [])]
    cats = [str(h).lower() for h in (categorical_heads or [])]
    for idx, name in enumerate(spatial):
        if idx >= FIXED_UNIVERSAL_COMMAND_DIM:
            break
        modes[idx] = float(command_target_mode(name))
    for offset, name in enumerate(cats):
        idx = len(spatial) + offset
        if idx >= FIXED_UNIVERSAL_COMMAND_DIM:
            break
        modes[idx] = float(command_target_mode(name, is_categorical=True))
    return modes


def ally_target_slot(ally_index: int) -> int:
    """Target slot for an allied squad slot index."""
    if not 0 <= int(ally_index) < MAX_SQUAD_UNITS:
        raise ValueError(f"ally_index {ally_index} out of range")
    return int(ally_index)


def enemy_target_slot(enemy_index: int) -> int:
    """Target slot for a visible-enemy slot index."""
    if not 0 <= int(enemy_index) < MAX_SQUAD_ENEMIES:
        raise ValueError(f"enemy_index {enemy_index} out of range")
    return MAX_SQUAD_UNITS + int(enemy_index)


def target_slot_kind(slot: int) -> int:
    """``CMD_MODE_ALLY_TARGET`` / ``CMD_MODE_ENEMY_TARGET`` / ``CMD_MODE_NONE``."""
    s = int(slot)
    if 0 <= s < MAX_SQUAD_UNITS:
        return CMD_MODE_ALLY_TARGET
    if MAX_SQUAD_UNITS <= s < MAX_SQUAD_UNITS + MAX_SQUAD_ENEMIES:
        return CMD_MODE_ENEMY_TARGET
    return CMD_MODE_NONE


def target_slot_local_index(slot: int) -> int:
    """Ally slot index, or enemy slot index, for a non-null target slot."""
    s = int(slot)
    if 0 <= s < MAX_SQUAD_UNITS:
        return s
    if MAX_SQUAD_UNITS <= s < MAX_SQUAD_UNITS + MAX_SQUAD_ENEMIES:
        return s - MAX_SQUAD_UNITS
    return -1


def encode_race_vector(own_race: str | None, enemy_race: str | None) -> np.ndarray:
    """Own + enemy one-hots in fixed ``SQUAD_RACE_ORDER`` (terran, zerg, protoss)."""
    vec = np.zeros(SQUAD_RACE_DIM, dtype=np.float32)
    own = str(own_race or "").lower().strip()
    enemy = str(enemy_race or "").lower().strip()
    n = len(SQUAD_RACE_ORDER)
    for i, name in enumerate(SQUAD_RACE_ORDER):
        if own == name:
            vec[i] = 1.0
        if enemy == name:
            vec[n + i] = 1.0
    return vec


def include_race_enabled(training_conf: dict) -> bool:
    """Whether squad obs include the 6-dim own/enemy race one-hot.

    Defaults to True so CTF/scenario/AoE levels keep a stable 144-wide
    ``query_linear`` after the first graft. Only ``ctf_level_01`` opts out
    (``include_race: false``) so the universal→squad graft stays 138-wide.
    """
    env_conf = training_conf.get("env", {}) if isinstance(training_conf.get("env"), dict) else {}
    if "include_race" in training_conf:
        return bool(training_conf["include_race"])
    return bool(env_conf.get("include_race", True))


def include_upgrades_enabled(training_conf: dict) -> bool:
    """Whether squad obs include the 6-dim own/enemy armor+shield upgrade cache."""
    env_conf = training_conf.get("env", {}) if isinstance(training_conf.get("env"), dict) else {}
    if "include_upgrades" in training_conf:
        return bool(training_conf["include_upgrades"])
    return bool(env_conf.get("include_upgrades", False))


def encode_upgrades_vector(
    own_ground: float,
    own_air: float,
    own_shields: float,
    enemy_ground: float,
    enemy_air: float,
    enemy_shields: float,
) -> np.ndarray:
    """Pack normalized armor/shield levels (already in [0, 1])."""
    return np.asarray(
        [own_ground, own_air, own_shields, enemy_ground, enemy_air, enemy_shields],
        dtype=np.float32,
    )


def update_armor_shield_cache(
    cache: np.ndarray,
    *,
    is_enemy: bool,
    is_flying: bool,
    armor_upgrade_level: int,
    shield_upgrade_level: int,
) -> None:
    """In-place max-update of a length-6 upgrades cache from one sighted unit."""
    base = 3 if is_enemy else 0
    armor_slot = base + (1 if is_flying else 0)
    shield_slot = base + 2
    armor_norm = min(max(int(armor_upgrade_level), 0), 3) / 3.0
    shield_norm = min(max(int(shield_upgrade_level), 0), 3) / 3.0
    cache[armor_slot] = max(float(cache[armor_slot]), armor_norm)
    cache[shield_slot] = max(float(cache[shield_slot]), shield_norm)


def squad_action_space_nvec():
    """MultiDiscrete nvec: actor bit per slot + command + target slot + spatial."""
    return [2] * MAX_SQUAD_UNITS + [
        FIXED_UNIVERSAL_COMMAND_DIM,
        SQUAD_TARGET_SLOTS,
        SQUAD_SPATIAL_CELLS,
    ]


def squad_fallback_action(unit_mask: np.ndarray | None = None) -> np.ndarray:
    """Safe fallback squad action: one alive unit, first head, null target, center cell."""
    from src.agent.universal_constants import universal_fallback_action

    uni = universal_fallback_action()
    action = np.zeros(SQUAD_ACTION_DIM, dtype=np.int64)
    mask = np.asarray(unit_mask if unit_mask is not None else np.ones(MAX_SQUAD_UNITS)).reshape(-1)
    for i in range(min(MAX_SQUAD_UNITS, len(mask))):
        if mask[i] > 0.5:
            action[i] = 1
            break
    else:
        action[0] = 1
    action[SQUAD_ACTION_COMMAND_IDX] = int(uni[0])
    action[SQUAD_ACTION_TARGET_IDX] = SQUAD_NULL_TARGET_SLOT
    action[SQUAD_ACTION_SPATIAL_IDX] = int(uni[1])
    return action
