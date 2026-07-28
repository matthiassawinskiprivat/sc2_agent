"""Constants for the CTF squad pointer training stack."""

import numpy as np

from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from src.agent.unit_scalar_constants import UnitScalarConstants

MAX_SQUAD_UNITS = 24
# Master catalog size (fallback). Live obs dim is len(active_unit_scalars).
SQUAD_UNIT_FEAT_DIM = UnitScalarConstants.get_dim()
SQUAD_COMMAND_DIM = 10
# Own + enemy race one-hots (terran/zerg/protoss each); gated by env.include_race.
SQUAD_RACE_DIM = 6
SQUAD_RACE_ORDER = ("terran", "zerg", "protoss")
SQUAD_SPATIAL_RESOLUTION = 32
SQUAD_SPATIAL_CELLS = SQUAD_SPATIAL_RESOLUTION * SQUAD_SPATIAL_RESOLUTION
SQUAD_POINTER_DIM = MAX_SQUAD_UNITS
SQUAD_FLAT_ACTION_DIM = (
    SQUAD_POINTER_DIM * 2 + FIXED_UNIVERSAL_COMMAND_DIM + SQUAD_SPATIAL_CELLS
)
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_MIN_SELECTED_UNITS = 1


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
    """Whether squad obs include the 6-dim own/enemy race one-hot."""
    env_conf = training_conf.get("env", {}) if isinstance(training_conf.get("env"), dict) else {}
    if "include_race" in training_conf:
        return bool(training_conf["include_race"])
    return bool(env_conf.get("include_race", False))


def squad_action_space_nvec():
    """MultiDiscrete nvec: binary pointer slot per unit + command + spatial."""
    return [2] * MAX_SQUAD_UNITS + [FIXED_UNIVERSAL_COMMAND_DIM, SQUAD_SPATIAL_CELLS]


def squad_fallback_action(unit_mask: np.ndarray | None = None) -> np.ndarray:
    """Safe fallback squad action: one alive unit, first command head, center spatial."""
    from src.agent.universal_constants import universal_fallback_action

    uni = universal_fallback_action()
    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    mask = np.asarray(unit_mask if unit_mask is not None else np.ones(MAX_SQUAD_UNITS)).reshape(-1)
    for i in range(min(MAX_SQUAD_UNITS, len(mask))):
        if mask[i] > 0.5:
            action[i] = 1
            break
    else:
        action[0] = 1
    action[MAX_SQUAD_UNITS] = int(uni[0])
    action[MAX_SQUAD_UNITS + 1] = int(uni[1])
    return action
