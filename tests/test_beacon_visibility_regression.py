import numpy as np
from unittest.mock import MagicMock
from sc2.position import Point2
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
from tests.obs_transformer_test_utils import get_layer, layer_sum, transform_map


class MockUnitTypeId:
    def __init__(self, name, value=0):
        self.name = name
        self.value = value

    def __eq__(self, other):
        if isinstance(other, MockUnitTypeId):
            return self.value == other.value
        return self.value == other


class MockUnit:
    def __init__(self, type_name, x, y, radius=1, is_enemy=True):
        self.type_id = MockUnitTypeId(type_name, 100)
        self.position = Point2((x, y))
        self.radius = radius
        self.is_structure = False
        self.is_enemy = is_enemy
        self.is_cloaked = False
        self.is_detected = True
        self.is_snapshot = False
        self.is_biological = False
        self.is_mechanical = False
        self.is_armored = False
        self.is_massive = False
        self.is_flying = False
        self.health = 100
        self.health_max = 100
        self.shield = 0
        self.shield_max = 0
        self.energy = 0
        self.energy_max = 0
        self.build_progress = 1.0
        self.detect_range = 0
        self.is_ready = True
        self.is_blip = False
        self.is_hallucination = False


def create_mock_obs(units, map_size=(64, 64)):
    mock_game_info = MagicMock()
    mock_game_info.map_size.width = map_size[0]
    mock_game_info.map_size.height = map_size[1]
    mock_playable_area = MagicMock()
    mock_playable_area.x = 0
    mock_playable_area.y = 0
    mock_playable_area.width = map_size[0]
    mock_playable_area.height = map_size[1]
    mock_game_info.playable_area = mock_playable_area
    mock_game_info.pathing_grid.data_numpy = np.ones(map_size, dtype=np.uint8)
    mock_game_info.terrain_height.data_numpy = np.zeros(map_size, dtype=np.float32)

    mock_creep = MagicMock()
    mock_creep.data_numpy = np.zeros(map_size, dtype=np.uint8)

    return {
        "units": [u for u in units if not u.is_enemy],
        "enemy_units": [u for u in units if u.is_enemy],
        "effects": [],
        "game_info": mock_game_info,
        "game_data": MagicMock(),
        "creep": mock_creep,
    }


def test_beacon_visible_with_enemy_density_layer():
    transformer = ObservationTransformer(map_size=(64, 64))
    beacon = MockUnit("BEACON_TERRANSMALL", 32, 32, is_enemy=True)
    map_data = transform_map(transformer, create_mock_obs([beacon]))
    enemy_density = get_layer(map_data, LayerConstants.ENEMY_UNIT_DENSITY)
    assert np.max(enemy_density) > 0


def test_beacon_invisible_when_layer_pruned():
    active_layers = [
        LayerConstants.ALLIED_UNIT_DENSITY,
        LayerConstants.TERRAIN_PATHABLE,
    ]
    transformer = ObservationTransformer(
        map_size=(64, 64),
        training_conf={"active_layers": active_layers},
    )
    beacon = MockUnit("BEACON_TERRANSMALL", 32, 32, is_enemy=True)
    map_data = transform_map(transformer, create_mock_obs([beacon]))
    assert map_data.shape[0] == len(active_layers)


def test_beacon_rendered_as_enemy_even_if_neutral():
    transformer = ObservationTransformer(map_size=(64, 64))
    beacon = MockUnit("BEACON_TERRANSMALL", 10, 10, is_enemy=True)
    map_data = transform_map(transformer, create_mock_obs([beacon]))
    assert layer_sum(map_data, LayerConstants.ENEMY_UNIT_DENSITY) > 0
    assert layer_sum(map_data, LayerConstants.ALLIED_UNIT_DENSITY) == 0
