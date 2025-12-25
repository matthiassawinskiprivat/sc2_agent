import pytest
import numpy as np
from unittest.mock import MagicMock
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants

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
    
    # Add name attribute to type_id for the check "BEACON" in u_name.upper()
    # In python-sc2, unit.type_id is an enum, so we mock its behavior
    type_id_mock = MagicMock()
    type_id_mock.value = unit_type.value if hasattr(unit_type, "value") else unit_type
    type_id_mock.name = unit_type.name # "BEACON_TERRANSMALL"
    unit.type_id = type_id_mock
    
    return unit

def test_beacon_visibility_default_config():
    """
    Test that the Beacon is NOT visible when training_conf is missing (default behavior).
    This reproduces the bug where the agent is blind.
    """
    # 1. Initialize Transformer without training_conf (simulating current bug)
    transformer = ObservationTransformer(map_size=(128, 128))
    
    # 2. Create a Mock Observation with a Beacon
    beacon_pos = Point2((64, 64))
    beacon = create_mock_unit(UnitTypeId.BEACON_TERRANSMALL, beacon_pos, is_enemy=True, tag=100)
    
    raw_obs = {
        'units': [],
        'enemy_units': [beacon],
        'effects': [],
        'game_info': MagicMock(), # Mock game info if needed
        'game_data': MagicMock(),
        'creep': None,
        'command_target': None
    }
    
    # Mock Game Info details needed for coordinate mapping
    raw_obs['game_info'].map_size.width = 128
    raw_obs['game_info'].map_size.height = 128
    raw_obs['game_info'].playable_area.x = 0
    raw_obs['game_info'].playable_area.y = 0
    raw_obs['game_info'].playable_area.width = 128
    raw_obs['game_info'].playable_area.height = 128
    raw_obs['game_info'].pathing_grid.data_numpy = np.ones((128, 128))
    raw_obs['game_info'].terrain_height.data_numpy = np.zeros((128, 128))
    
    # 3. Transform
    layers = transformer.transform(raw_obs)
    
    # 4. Assert Beacon is MISSING (This confirms the bug exists)
    enemy_density_layer = layers[LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]]
    
    # The beacon should be rendered at (64, 64) if it were visible.
    # Since we expect it to be invisible with default config, max value should be 0.
    assert np.max(enemy_density_layer) == 0.0, "Beacon should be invisible with default config (reproducing bug)"

def test_beacon_visibility_with_config():
    """
    Test that the Beacon IS visible when render_beacon=True is passed.
    This validates the fix.
    """
    # 1. Initialize Transformer WITH training_conf
    training_conf = {'render_beacon': True}
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=training_conf)
    
    # 2. Create a Mock Observation with a Beacon
    beacon_pos = Point2((64, 64))
    beacon = create_mock_unit(UnitTypeId.BEACON_TERRANSMALL, beacon_pos, is_enemy=True, tag=100)
    
    raw_obs = {
        'units': [],
        'enemy_units': [beacon],
        'effects': [],
        'game_info': MagicMock(),
        'game_data': MagicMock(),
        'creep': None,
        'command_target': None
    }
    
    # Mock Game Info
    raw_obs['game_info'].map_size.width = 128
    raw_obs['game_info'].map_size.height = 128
    raw_obs['game_info'].playable_area.x = 0
    raw_obs['game_info'].playable_area.y = 0
    raw_obs['game_info'].playable_area.width = 128
    raw_obs['game_info'].playable_area.height = 128
    raw_obs['game_info'].pathing_grid.data_numpy = np.ones((128, 128))
    raw_obs['game_info'].terrain_height.data_numpy = np.zeros((128, 128))
    
    # 3. Transform
    layers = transformer.transform(raw_obs)
    
    # 4. Assert Beacon is VISIBLE
    enemy_density_layer = layers[LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]]
    
    assert np.max(enemy_density_layer) > 0.0, "Beacon should be visible when render_beacon=True"
