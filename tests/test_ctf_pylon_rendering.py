import pytest
import numpy as np
from unittest.mock import MagicMock
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
from tests.obs_transformer_test_utils import get_layer, layer_val, layer_sum, transform_map
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

def test_pylon_rendering_on_multiple_layers():
    # 1. Setup Transformer
    transformer = ObservationTransformer(map_size=(128, 128))
    
    # 2. Mock Game Info
    game_info = MagicMock()
    game_info.playable_area.x = 0
    game_info.playable_area.y = 0
    game_info.playable_area.width = 100
    game_info.playable_area.height = 100
    game_info.map_size.width = 100
    game_info.map_size.height = 100
    game_info.map_name = "TestMap"
    
    # Mock Grids (empty for now)
    game_info.pathing_grid = np.ones((100, 100), dtype=np.uint8)
    game_info.terrain_height = np.zeros((100, 100), dtype=np.uint8)
    
    # 3. Mock Allied Pylon
    pylon = MagicMock()
    pylon.type_id = UnitTypeId.PYLON
    pylon.position = Point2((50, 50))
    pylon.radius = 1.125
    pylon.health = 10.0
    pylon.health_max = 200.0
    pylon.shield = 10.0
    pylon.shield_max = 200.0
    pylon.energy = 0
    pylon.energy_max = 0
    pylon.weapon_cooldown = 0
    pylon.is_biological = False
    pylon.is_mechanical = True
    pylon.is_armored = True
    pylon.is_massive = False
    pylon.is_flying = False
    pylon.is_cloaked = False
    pylon.is_structure = True
    pylon.is_blip = False
    pylon.detect_range = 0
    pylon.build_progress = 1.0
    pylon.is_ready = True
    pylon.is_detected = True
    pylon.is_hallucination = False
    pylon.is_snapshot = False

    # 4. Mock Raw Obs
    raw_obs = {
        'units': [],
        'enemy_units': [],
        'structures': [pylon],
        'enemy_structures': [],
        'game_info': game_info,
        'game_data': MagicMock(),
        'command_target': Point2((50, 50)) # Objective target is the pylon
    }
    
    # 5. Transform
    obs = transform_map(transformer, raw_obs)

    assert layer_sum(obs, LayerConstants.ALLIED_UNIT_DENSITY) > 0
    assert layer_sum(obs, LayerConstants.ALLIED_UNIT_MECHANICAL) > 0
    assert layer_sum(obs, LayerConstants.ALLIED_HEALTH) > 0  # damaged pylon (50% HP)
    assert layer_sum(obs, LayerConstants.COMMAND_OBJECTIVE) > 0

def test_enemy_pylon_rendering():
    transformer = ObservationTransformer(map_size=(128, 128))
    
    game_info = MagicMock()
    game_info.playable_area.x = 0
    game_info.playable_area.y = 0
    game_info.playable_area.width = 100
    game_info.playable_area.height = 100
    game_info.map_size.width = 100
    game_info.map_size.height = 100
    game_info.pathing_grid = np.ones((100, 100), dtype=np.uint8)
    game_info.terrain_height = np.zeros((100, 100), dtype=np.uint8)
    
    # Mock Enemy Pylon
    pylon = MagicMock()
    pylon.type_id = UnitTypeId.PYLON
    pylon.position = Point2((25, 25))
    pylon.radius = 1.125
    pylon.health = 100.0
    pylon.health_max = 200.0
    pylon.shield = 100.0
    pylon.shield_max = 200.0
    pylon.energy = 0
    pylon.energy_max = 0
    pylon.weapon_cooldown = 0
    pylon.is_biological = False
    pylon.is_mechanical = True
    pylon.is_armored = True
    pylon.is_massive = False
    pylon.is_flying = False
    pylon.is_cloaked = False
    pylon.is_structure = True
    pylon.is_blip = False
    pylon.detect_range = 0
    pylon.build_progress = 1.0
    pylon.is_ready = True
    pylon.is_detected = True
    pylon.is_hallucination = False
    pylon.is_snapshot = False

    raw_obs = {
        'units': [],
        'enemy_units': [],
        'structures': [],
        'enemy_structures': [pylon],
        'game_info': game_info,
        'game_data': MagicMock(),
        'command_target': Point2((25, 25))
    }
    
    obs = transform_map(transformer, raw_obs)

    assert layer_sum(obs, LayerConstants.ENEMY_UNIT_DENSITY) > 0
    assert layer_sum(obs, LayerConstants.ENEMY_HEALTH) > 0
    assert layer_sum(obs, LayerConstants.COMMAND_OBJECTIVE) > 0