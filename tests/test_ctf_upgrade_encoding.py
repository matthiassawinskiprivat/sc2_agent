"""Armor/shield upgrade cache and DPS attack-upgrade scaling."""
from unittest.mock import MagicMock

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId

from src.agent.squad_constants import (
    SQUAD_UPGRADES_DIM,
    update_armor_shield_cache,
)
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants


def test_update_armor_shield_cache_max_and_slots():
    cache = np.zeros(SQUAD_UPGRADES_DIM, dtype=np.float32)
    update_armor_shield_cache(
        cache,
        is_enemy=True,
        is_flying=True,
        armor_upgrade_level=3,
        shield_upgrade_level=0,
    )
    assert cache[4] == 1.0  # enemy air
    assert cache[3] == 0.0  # enemy ground unseen
    update_armor_shield_cache(
        cache,
        is_enemy=True,
        is_flying=True,
        armor_upgrade_level=1,
        shield_upgrade_level=0,
    )
    assert cache[4] == 1.0  # max kept
    update_armor_shield_cache(
        cache,
        is_enemy=False,
        is_flying=False,
        armor_upgrade_level=2,
        shield_upgrade_level=3,
    )
    assert abs(cache[0] - 2 / 3) < 1e-6
    assert cache[2] == 1.0


def test_dps_scales_with_attack_upgrade_level():
    """Catalog DPS intensity rises with attack_upgrade_level × upgrade_bonus."""
    conf = {
        "active_layers": ["enemy_g_dps_base"],
        "agent_architecture": "universal",
    }
    tr = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    layers0 = np.zeros((LayerConstants.get_num_layers(), 128, 128), dtype=np.float32)
    layers2 = np.zeros_like(layers0)

    def _marine(level: int):
        u = MagicMock()
        u.type_id = UnitTypeId.MARINE
        u.attack_upgrade_level = level
        u.position = MagicMock(x=64.0, y=64.0)
        u.radius = 0.375
        u.is_flying = False
        u.is_hallucination = False
        u.owner_id = 2
        u.buffs = []
        return u

    tr.current_layer_map = dict(LayerConstants.LAYERS)
    tr.active_indices = set(LayerConstants.LAYERS.values())
    tr.scale = 1.0
    tr.offset_x = 0
    tr.offset_y = 0
    tr.crop_x = 0
    tr.crop_y = 0
    tr.feature_map_width = 128
    tr.feature_map_height = 128
    tr._grid_x = np.arange(128).reshape(1, -1)
    tr._grid_y = np.arange(128).reshape(-1, 1)
    tr._unit_is_combat_ready = lambda u: True

    tr._draw_unit_dps(_marine(0), layers0, True, 64, 64, None)
    tr._draw_unit_dps(_marine(2), layers2, True, 64, 64, None)
    idx = LayerConstants.LAYERS[LayerConstants.ENEMY_G_DPS_BASE]
    assert float(np.max(layers2[idx])) > float(np.max(layers0[idx]))
