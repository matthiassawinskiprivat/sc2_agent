import numpy as np
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId

from src.transformers.buildability import compute_buildable_mask


def test_zerg_extractor_requires_geyser():
    path = np.ones((4, 4), dtype=np.uint8)
    creep = np.zeros((4, 4), dtype=np.uint8)
    geysers = np.zeros((4, 4), dtype=np.float32)
    geysers[2, 2] = 1.0

    mask = compute_buildable_mask(
        Race.Zerg,
        UnitTypeId.EXTRACTOR,
        path,
        creep_data=creep,
        geyser_mask=geysers,
    )
    assert mask[2, 2] == 1.0
    assert mask[0, 0] == 0.0


def test_zerg_pool_requires_creep():
    path = np.ones((4, 4), dtype=np.uint8)
    creep = np.zeros((4, 4), dtype=np.uint8)
    creep[1, 1] = 1.0

    mask = compute_buildable_mask(
        Race.Zerg,
        UnitTypeId.SPAWNINGPOOL,
        path,
        creep_data=creep,
    )
    assert mask[1, 1] == 1.0
    assert mask[0, 0] == 0.0


def test_zerg_hatchery_off_creep_ok():
    path = np.ones((4, 4), dtype=np.uint8)
    creep = np.zeros((4, 4), dtype=np.uint8)

    mask = compute_buildable_mask(
        Race.Zerg,
        UnitTypeId.HATCHERY,
        path,
        creep_data=creep,
    )
    assert mask[1, 1] == 1.0
