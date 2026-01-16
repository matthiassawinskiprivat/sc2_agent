"""
Unit tests for UnitControlAgent.get_unload_mask.
"""

import numpy as np
import pytest
from src.agent.unit_control_agent import UnitControlAgent
from tests.masking_test_utils import MockGameInfo, MockUnit, create_mock_game_state, add_cliff_to_mock_state
from sc2.ids.unit_typeid import UnitTypeId

def test_get_unload_mask_terrain_erosion():
    """
    Test that the unload mask erodes pathable terrain.
    With a 2-pixel margin, a small pathable island should shrink or disappear.
    """
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=128, dilation_size=0)
    
    # Create 128x128 map, all unpathable
    pathing = np.zeros((128, 128), dtype=np.uint8)
    # Create a 10x10 pathable square in the middle
    pathing[60:70, 60:70] = 1
    game_info = MockGameInfo(pathing)
    
    agent.last_game_info = game_info
    
    # get_action_mask (move) should show the 10x10 area (potentially slightly different due to meshgrid/resize)
    move_mask = agent.get_action_mask()
    move_mask_2d = move_mask.reshape((128, 128))
    assert move_mask_2d[65, 65] == True
    assert np.sum(move_mask_2d) > 0
    
    # get_unload_mask should have eroded this area by 2 pixels on each side
    # Original: 60 to 69 inclusive. 
    # Eroded: 62 to 67 inclusive (approx).
    unload_mask = agent.get_unload_mask([], game_info)
    
    # Helper to check world pos
    def is_valid_at(x, y):
        # We need to find the index that corresponds to world (x, y)
        # For 128x128 map and 128x128 grid: index = (127-y)*128 + x
        idx = int((127 - y) * 128 + x)
        return unload_mask[idx]

    # The center should still be true if the island is large enough
    assert is_valid_at(65, 65) == True
    
    # The edges of the original island (60, 61, 68, 69) should be false due to 2px erosion
    assert is_valid_at(60, 65) == False
    assert is_valid_at(61, 65) == False
    assert is_valid_at(68, 65) == False
    assert is_valid_at(69, 65) == False

def test_get_unload_mask_building_exclusion():
    """
    Test that buildings are excluded with their exact footprint (no erosion).
    """
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=128, dilation_size=0)
    
    # Uniformly pathable map
    game_info = create_mock_game_state(grid_size=128, pathing_value=1)
    agent.last_game_info = game_info
    
    # Add a building at center (64, 64) with radius 5
    building = MockUnit(pos=(64, 64), radius=5, is_structure=True, type_id=UnitTypeId.COMMANDCENTER)
    
    unload_mask = agent.get_unload_mask([building], game_info)
    
    def is_valid_at(x, y):
        idx = int((127 - y) * 128 + x)
        return unload_mask[idx]

    # Center (where building is) must be False
    assert is_valid_at(64, 64) == False
    
    # Area just outside the building (radius 5) should be True
    # (64 + 6, 64) should be outside
    assert is_valid_at(70, 64) == True

def test_get_unload_mask_complex_logic():
    """
    Verify the logic: Fill buildings -> Erode terrain -> Mask buildings.
    If a building is at a cliff edge, erosion should happen as if the building wasn't there,
    but the building's footprint itself should be removed last.
    """
    # This test is more of a design validation once implemented.
    pass
