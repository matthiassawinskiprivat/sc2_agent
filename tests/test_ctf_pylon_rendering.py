import pytest
import numpy as np
from unittest.mock import MagicMock
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
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
    pylon.energy_max = 0
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
    obs = transformer.transform(raw_obs)
    
    # 6. Verify Layers
    
    # Layer 0: allied_unit_density
    idx_den = LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]
    assert obs[idx_den, 63, 64] > 0 # (50, 50) maps to center of 128x128
    
    # Layer 2: allied_unit_mechanical
    idx_mech = LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_MECHANICAL]
    assert obs[idx_mech, 63, 64] > 0
    
    # Layer 14: allied_health
    idx_hp = LayerConstants.LAYERS[LayerConstants.ALLIED_HEALTH]
    assert obs[idx_hp, 63, 64] > 0
    
    # Layer 41: command_objective
    idx_obj = LayerConstants.LAYERS[LayerConstants.COMMAND_OBJECTIVE]
    assert obs[idx_obj, 63, 64] > 0

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
    pylon.health = 200.0
    pylon.health_max = 200.0
    pylon.shield = 200.0
    pylon.shield_max = 200.0
    pylon.energy_max = 0
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
    
    raw_obs = {
        'units': [],
        'enemy_units': [],
        'structures': [],
        'enemy_structures': [pylon],
        'game_info': game_info,
        'game_data': MagicMock(),
        'command_target': Point2((25, 25))
    }
    
    obs = transformer.transform(raw_obs)
    
    # Verify Enemy Layers
    idx_den = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]
    # (25, 25) on 100x100 -> (32, 95) roughly in pixel coords? 
    # Let's check where it is non-zero
    assert np.any(obs[idx_den] > 0)
    
    idx_hp = LayerConstants.LAYERS[LayerConstants.ENEMY_HEALTH]
    assert np.any(obs[idx_hp] > 0)
    
    idx_obj = LayerConstants.LAYERS[LayerConstants.COMMAND_OBJECTIVE]
    assert np.any(obs[idx_obj] > 0)