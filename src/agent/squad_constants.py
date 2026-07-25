"""Constants for the CTF squad pointer training stack."""

import numpy as np

from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from src.agent.unit_scalar_constants import UnitScalarConstants

MAX_SQUAD_UNITS = 24
# Master catalog size (fallback). Live obs dim is len(active_unit_scalars).
SQUAD_UNIT_FEAT_DIM = UnitScalarConstants.get_dim()
SQUAD_COMMAND_DIM = 10
SQUAD_SPATIAL_RESOLUTION = 32
SQUAD_SPATIAL_CELLS = SQUAD_SPATIAL_RESOLUTION * SQUAD_SPATIAL_RESOLUTION
SQUAD_POINTER_DIM = MAX_SQUAD_UNITS
SQUAD_FLAT_ACTION_DIM = (
    SQUAD_POINTER_DIM * 2 + FIXED_UNIVERSAL_COMMAND_DIM + SQUAD_SPATIAL_CELLS
)
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_MIN_SELECTED_UNITS = 1


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
