
import pytest
import numpy as np
from unittest.mock import MagicMock
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.unit_control_agent import UnitControlAgent

class MockUnit:
    def __init__(self, type_id, position, is_flying=False, is_structure=False):
        self.type_id = type_id
        self.position = position
        self.is_flying = is_flying
        self.is_structure = is_structure

def test_transport_mask_precision():
    """
    Verifies that the transport mask is precise (not dilated).
    We assert that a single unit results in a minimal number of masked pixels (ideally 1).
    """
    # 1. Setup Agent with default dilation
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=3)
    
    # 2. Mock Game Info
    game_info = MagicMock()
    game_info.map_size.width = 32
    game_info.map_size.height = 32
    game_info.pathing_grid.width = 32
    game_info.pathing_grid.height = 32
    game_info.pathing_grid.data_numpy = np.ones((32, 32), dtype=np.uint8) # All walkable
    game_info.playable_area.x = 0
    game_info.playable_area.y = 0
    game_info.playable_area.width = 32
    game_info.playable_area.height = 32

    # 3. Place a Unit at exact center of a grid cell
    # Grid cell size = 1.0 (32 map / 32 grid)
    # Center of cell (10, 10) is world (10.5, 10.5)
    unit = MockUnit(UnitTypeId.MEDIVAC, Point2((10.5, 10.5)))
    units = [unit]

    # 4. Generate Mask
    mask = agent.get_transport_mask(units, game_info, is_transport=False)
    
    # 5. Analyze Mask
    active_pixels = np.sum(mask)
    
    print(f"Active pixels in mask: {active_pixels}")
    
    # 6. Assert Precision
    # With dilation=3 (current default), this often produces > 1 pixel due to "halo".
    # With NO dilation (desired), this should produce exactly 1 pixel for a perfectly centered unit.
    assert active_pixels == 1, f"Mask is dilated or imprecise! Expected 1 active pixel, got {active_pixels}"

if __name__ == "__main__":
    try:
        test_transport_mask_precision()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
        sys.exit(1)
