import pytest
import numpy as np
from unittest.mock import MagicMock
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2, Rect
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants

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
        self.type_id = MockUnitTypeId(type_name, 100) # Arbitrary value
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
    
    # Mock playable area to match map size for simplicity
    mock_playable_area = MagicMock()
    mock_playable_area.x = 0
    mock_playable_area.y = 0
    mock_playable_area.width = map_size[0]
    mock_playable_area.height = map_size[1]
    mock_game_info.playable_area = mock_playable_area
    
    # Mock pathing grid (all playable)
    mock_game_info.pathing_grid.data_numpy = np.ones(map_size, dtype=np.uint8)
    mock_game_info.terrain_height.data_numpy = np.zeros(map_size, dtype=np.float32)

    mock_creep = MagicMock()
    mock_creep.data_numpy = np.zeros(map_size, dtype=np.uint8)

    return {
        'units': [u for u in units if not u.is_enemy],
        'enemy_units': [u for u in units if u.is_enemy],
        'effects': [],
        'game_info': mock_game_info,
        'game_data': MagicMock(), # Mock game data
        'creep': mock_creep
    }

def test_beacon_visible_with_config():
    """
    Verifies that the Beacon is rendered on ENEMY_UNIT_DENSITY when render_beacon=True.
    """
    # Setup
    transformer = ObservationTransformer(
        map_size=(64, 64), 
        training_conf={'render_beacon': True}
    )
    
    # Create a Beacon unit (Neutral/Enemy)
    beacon = MockUnit("BEACON_TERRANSMALL", 32, 32, is_enemy=True)
    
    # Create observation
    obs = create_mock_obs([beacon])
    
    # Transform
    layers = transformer.transform(obs)
    
    # Assert
    enemy_density_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]
    enemy_density_layer = layers[enemy_density_idx]
    
    # Check if there is ANY signal in the middle of the map
    # We allow some tolerance for coordinate mapping
    assert np.max(enemy_density_layer) > 0, "Beacon should be visible on ENEMY_UNIT_DENSITY layer"
    
    # Verify exact position roughly
    # Coordinate system: (0,0) is bottom-left in SC2, top-left in numpy array usually, 
    # but transformer handles this.
    # px, py calculation in transformer:
    # py = int(self.feature_map_height - 1 - (rel_y * self.scale + self.offset_y))
    # With 64x64 map and 64x64 feature map, scale should be 1.0.
    # x=32, y=32 -> px=32, py=64-1-32 = 31.
    
    # Note: The transformer uses a drawing radius.
    assert enemy_density_layer[31, 32] > 0 or enemy_density_layer[32, 32] > 0, \
        "Beacon signal should be at expected coordinates"

def test_beacon_invisible_without_config():
    """
    Verifies that the Beacon is NOT rendered when render_beacon is missing or False.
    """
    # Setup
    transformer = ObservationTransformer(
        map_size=(64, 64), 
        training_conf={}
    )
    
    # Create a Beacon unit
    beacon = MockUnit("BEACON_TERRANSMALL", 32, 32, is_enemy=True)
    
    # Create observation
    obs = create_mock_obs([beacon])
    
    # Transform
    layers = transformer.transform(obs)
    
    # Assert
    enemy_density_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]
    enemy_density_layer = layers[enemy_density_idx]
    
    assert np.max(enemy_density_layer) == 0, "Beacon should NOT be visible without config"

def test_beacon_rendered_as_enemy_even_if_neutral():
    """
    In the environment, the beacon might be owned by neutral (16) or enemy (2).
    The logic in sc2_bot.py puts it in 'enemy_units' list manually.
    The transformer relies on the 'is_enemy' flag passed to _process_single_unit.
    This test verifies that if we pass it as enemy, it renders on enemy layer.
    """
    transformer = ObservationTransformer(
        map_size=(64, 64), 
        training_conf={'render_beacon': True}
    )
    
    beacon = MockUnit("BEACON_TERRANSMALL", 10, 10, is_enemy=True)
    obs = create_mock_obs([beacon])
    
    layers = transformer.transform(obs)
    
    enemy_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]
    allied_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]
    
    assert np.max(layers[enemy_idx]) > 0, "Should appear on enemy layer"
    assert np.max(layers[allied_idx]) == 0, "Should NOT appear on allied layer"
