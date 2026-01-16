import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock
from src.agent.unit_control_agent import UnitControlAgent
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

def create_mock_unit(unit_type, position, tag):
    unit = MagicMock()
    unit.type_id = unit_type
    unit.position = Point2(position)
    unit.tag = tag
    return unit

def test_transport_mask_generation():
    """
    Verifies that get_transport_mask generates a mask with 'True' values
    near transport units (Medivac, Overlord, WarpPrism) and 'False' elsewhere.
    """
    # 1. Setup Agent
    grid_size = 32 # Small grid for easier testing
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=grid_size, dilation_size=1)
    
    # 2. Setup Mock Game Info
    mock_game_info = MagicMock()
    mock_game_info.map_size.width = 100
    mock_game_info.map_size.height = 100
    mock_game_info.playable_area.x = 0
    mock_game_info.playable_area.y = 0
    mock_game_info.playable_area.width = 100
    mock_game_info.playable_area.height = 100
    
    # Mock pathing grid (needed for coordinate conversion internally, though not used for this mask logic directly in theory, 
    # but UnitControlAgent might rely on it for initialization)
    mock_game_info.pathing_grid.width = 100
    mock_game_info.pathing_grid.height = 100
    
    # 3. Create Transport Units
    # Medivac at (20, 20)
    medivac = create_mock_unit(UnitTypeId.MEDIVAC, (20, 20), 1)
    # Overlord at (80, 80)
    overlord = create_mock_unit(UnitTypeId.OVERLORD, (80, 80), 2)
    # WarpPrism at (50, 50)
    warp_prism = create_mock_unit(UnitTypeId.WARPPRISM, (50, 50), 3)
    # Marine (Non-transport) at (10, 10) - Should be ignored
    marine = create_mock_unit(UnitTypeId.MARINE, (10, 10), 4)
    
    units = [medivac, overlord, warp_prism, marine]
    
    # 4. Generate Mask
    # We pass the list of allied units
    mask = agent.get_transport_mask(units, mock_game_info)
    
    # 5. Verify Mask Shape
    assert mask.shape == (grid_size * grid_size,)
    
    # 6. Verify Content
    # Reshape to 2D for coordinate checking
    mask_2d = mask.reshape((grid_size, grid_size))
    
    # Convert world coordinates to grid coordinates
    # We use the agent's internal logic or approximation to find where (20,20) maps to.
    # Since map is 100x100 and grid is 32x32:
    # Scale = 32/100 = 0.32
    # (20, 20) -> (6.4, 6.4) -> index ~6
    
    # Helper to check if any True value exists in a window
    def check_window(grid, x, y, window=2):
        x_start = max(0, x - window)
        x_end = min(grid.shape[1], x + window + 1)
        y_start = max(0, y - window)
        y_end = min(grid.shape[0], y + window + 1)
        return np.any(grid[y_start:y_end, x_start:x_end])

    # Check Medivac region (20, 20)
    # X: 20/100 * 32 = 6.4 -> 6
    # Y: 20/100 -> 0.2 (World). Flipped for Image (0 at top): 0.8 * 32 = 25.6 -> 25
    assert check_window(mask_2d, 6, 25), "Medivac region should be masked True (X=6, Y=25)"
    
    # Check Overlord region (80, 80)
    # X: 80/100 * 32 = 25.6 -> 25
    # Y: 80/100 -> 0.8 (World). Flipped for Image: 0.2 * 32 = 6.4 -> 6
    assert check_window(mask_2d, 25, 6), "Overlord region should be masked True (X=25, Y=6)"
    
    # Check Marine region (10, 10) -> (3.2, 3.2) -> index ~3, Y ~28
    # Marine should NOT produce a mask
    # We can't strictly check for False here because mask dilations might overlap in dense maps,
    # but with these coordinates (10,10) vs (20,20), they are far enough (3 vs 6).
    assert not check_window(mask_2d, 3, 28, window=0), "Marine region should NOT be masked True"