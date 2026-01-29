import pytest
import numpy as np
from unittest.mock import MagicMock
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants

@pytest.fixture
def transformer():
    return ObservationTransformer(map_size=(64, 64))

def create_mock_unit(unit_type, pos, cargo_used=0, cargo_max=0):
    u = MagicMock(spec=Unit)
    u.type_id = unit_type
    u.position = pos
    u.cargo_used = cargo_used
    u.cargo_max = cargo_max
    u.radius = 1.0
    u.is_cloaked = False
    u.is_blip = False
    u.is_structure = False
    u.is_ready = True
    u.health = 100
    u.health_max = 100
    u.shield = 0
    u.shield_max = 0
    u.energy = 0
    u.energy_max = 0
    # Add minimal dps attributes to avoid crashes in dps drawing
    u.weapon_cooldown = 0
    u.attack_upgrade_level = 0
    u.detect_range = 0 # Default detection range
    return u

def test_transport_layers_rendering(transformer):
    """Test that Medivacs are drawn on TRANSPORT_SLOTS and PASSENGER_DENSITY."""
    
    # 1. Setup Mock Units
    # Empty Medivac at (10, 10)
    medivac_empty = create_mock_unit(
        UnitTypeId.MEDIVAC, Point2((10, 10)), 
        cargo_used=0, cargo_max=8
    )
    
    # Full Medivac at (20, 20) with 4 passengers
    medivac_full = create_mock_unit(
        UnitTypeId.MEDIVAC, Point2((20, 20)), 
        cargo_used=8, cargo_max=8
    )
    
    # Half-Full Medivac at (30, 30) with 2 passengers
    medivac_half = create_mock_unit(
        UnitTypeId.MEDIVAC, Point2((30, 30)), 
        cargo_used=2, cargo_max=8
    )
    
    # Marine (Not a transport)
    marine = create_mock_unit(
        UnitTypeId.MARINE, Point2((40, 40)),
        cargo_used=0, cargo_max=0
    )

    # Mock Game Info
    # Important: ObservationTransformer checks hasattr(game_info, 'playable_area')
    # Since MagicMock has everything, we must ensure it behaves as expected.
    game_info = MagicMock()
    # Mocking map_size
    game_info.map_size.width = 64
    game_info.map_size.height = 64
    # Ensure playable_area is treated as missing or raise AttributeError
    del game_info.playable_area
    # Disable terrain drawing to avoid shape errors
    del game_info.pathing_grid
    del game_info.terrain_height

    raw_obs = {
        'units': [medivac_empty, medivac_full, medivac_half, marine],
        'game_info': game_info,
        'game_data': None
    }

    # 2. Transform
    layers = transformer.transform(raw_obs)
    
    # 3. Assertions
    idx_slots = LayerConstants.LAYERS[LayerConstants.TRANSPORT_SLOTS]
    idx_pass = LayerConstants.LAYERS[LayerConstants.PASSENGER_DENSITY]
    
    # Coordinate Transform (Standard logic: y is inverted)
    # 64 height. (10, 10) -> px=10, py=63-10=53 (approx, depending on crop)
    py_10 = 64 - 1 - 10
    py_20 = 64 - 1 - 20
    py_30 = 64 - 1 - 30
    
    # Check Slots
    assert layers[idx_slots, py_10, 10] >= 120 # Empty (~0.5)
    assert layers[idx_slots, py_20, 20] == 255 # Full
    assert layers[idx_slots, py_30, 30] >= 120 # Available
    
    # Check Passengers
    # Medivac Full (8 used) -> 1.0 * 255 = 255
    assert layers[idx_pass, py_20, 20] == 255
    
    # Medivac Half (2 used) -> 2/8 = 0.25 -> 63.75
    val_half = layers[idx_pass, py_30, 30]
    assert 60 <= val_half <= 68
    
    # Medivac Empty -> 0
    assert layers[idx_pass, py_10, 10] == 0