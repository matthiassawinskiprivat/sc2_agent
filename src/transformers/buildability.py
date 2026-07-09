"""Race-specific build placement rules for terrain_buildable rendering."""

from typing import Optional, Set

import numpy as np
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId

ZERG_OFF_CREEP_STRUCTURES: Set[UnitTypeId] = {
    UnitTypeId.HATCHERY,
    UnitTypeId.LAIR,
    UnitTypeId.HIVE,
    UnitTypeId.EXTRACTOR,
}

ZERG_GEYSER_STRUCTURES: Set[UnitTypeId] = {
    UnitTypeId.EXTRACTOR,
}


def compute_buildable_mask(
    race: Race,
    structure_type: Optional[UnitTypeId],
    path_data: np.ndarray,
    creep_data: Optional[np.ndarray] = None,
    placement_data: Optional[np.ndarray] = None,
    geyser_mask: Optional[np.ndarray] = None,
    power_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns float32 mask (H, W) in SC2 grid coordinates."""
    pathable = path_data > 0
    creep = creep_data > 0 if creep_data is not None else np.zeros_like(pathable, dtype=bool)
    placement = placement_data > 0 if placement_data is not None else pathable
    geysers = geyser_mask > 0 if geyser_mask is not None else np.zeros_like(pathable, dtype=bool)
    power = power_mask > 0 if power_mask is not None else np.ones_like(pathable, dtype=bool)

    if race == Race.Zerg:
        if structure_type in ZERG_GEYSER_STRUCTURES:
            mask = geysers
        elif structure_type in ZERG_OFF_CREEP_STRUCTURES:
            mask = pathable
        else:
            mask = pathable & creep
    elif race == Race.Terran:
        mask = pathable & placement
    elif race == Race.Protoss:
        if structure_type == UnitTypeId.PYLON:
            mask = pathable
        else:
            mask = pathable & power
    else:
        mask = pathable

    return mask.astype(np.float32)
