import numpy as np
import pytest
from unittest.mock import MagicMock
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
from tests.obs_transformer_test_utils import get_layer, layer_sum, transform_map


def create_mock_unit(unit_type, position, is_enemy=False, tag=0):
    unit = MagicMock(spec=Unit)
    unit.type_id = unit_type
    unit.position = position
    unit.radius = 1.0
    unit.is_structure = False
    unit.is_biological = False
    unit.is_mechanical = True
    unit.is_armored = False
    unit.is_massive = False
    unit.is_flying = False
    unit.is_cloaked = False
    unit.is_blip = False
    unit.is_ready = True
    unit.is_hallucination = False
    unit.health = 100
    unit.health_max = 100
    unit.shield = 0
    unit.shield_max = 0
    unit.energy = 0
    unit.energy_max = 0
    unit.detect_range = 0
    unit.tag = tag

    type_id_mock = MagicMock()
    type_id_mock.value = unit_type.value if hasattr(unit_type, "value") else unit_type
    type_id_mock.name = unit_type.name
    unit.type_id = type_id_mock

    return unit


def _mock_game_info():
    game_info = MagicMock()
    game_info.map_size.width = 128
    game_info.map_size.height = 128
    game_info.playable_area.x = 0
    game_info.playable_area.y = 0
    game_info.playable_area.width = 128
    game_info.playable_area.height = 128
    game_info.pathing_grid.data_numpy = np.ones((128, 128))
    game_info.terrain_height.data_numpy = np.zeros((128, 128))
    return game_info


def test_beacon_visible_with_default_full_layers():
    """Beacons in enemy_units render on ENEMY_UNIT_DENSITY when that layer is active."""
    transformer = ObservationTransformer(map_size=(128, 128))
    beacon = create_mock_unit(UnitTypeId.BEACON_TERRANSMALL, Point2((64, 64)), is_enemy=True, tag=100)
    raw_obs = {
        "units": [],
        "enemy_units": [beacon],
        "effects": [],
        "game_info": _mock_game_info(),
        "game_data": MagicMock(),
        "creep": None,
        "command_target": None,
    }
    map_data = transform_map(transformer, raw_obs)
    enemy_density = get_layer(map_data, LayerConstants.ENEMY_UNIT_DENSITY)
    assert np.max(enemy_density) > 0.0


def test_beacon_invisible_when_enemy_density_pruned():
    """Pruned configs that omit enemy_unit_density must not render beacons."""
    active_layers = [
        LayerConstants.ALLIED_UNIT_DENSITY,
        LayerConstants.TERRAIN_PATHABLE,
        LayerConstants.COMMAND_OBJECTIVE,
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={"active_layers": active_layers},
    )
    beacon = create_mock_unit(UnitTypeId.BEACON_TERRANSMALL, Point2((64, 64)), is_enemy=True, tag=100)
    raw_obs = {
        "units": [],
        "enemy_units": [beacon],
        "effects": [],
        "game_info": _mock_game_info(),
        "game_data": MagicMock(),
        "creep": None,
        "command_target": None,
    }
    map_data = transform_map(transformer, raw_obs)
    assert map_data.shape[0] == len(active_layers)
    with pytest.raises(KeyError):
        get_layer(map_data, LayerConstants.ENEMY_UNIT_DENSITY, active_layers=active_layers)
