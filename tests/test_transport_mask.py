import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock
from src.agent.unit_control_agent import UnitControlAgent
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

def create_mock_unit(unit_type, position, tag, is_flying=False, is_structure=False):
    unit = MagicMock()
    unit.type_id = unit_type
    unit.position = Point2(position)
    unit.tag = tag
    unit.is_flying = is_flying
    unit.is_structure = is_structure
    return unit

def test_transport_mask_generation_ground_actor():

    """

    Verifies that get_transport_mask (for ground actor) highlights transports.

    """

    # 1. Setup Agent

    grid_size = 32

    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=grid_size, dilation_size=1)

    

    # 2. Setup Mock Game Info

    mock_game_info = MagicMock()

    mock_game_info.map_size.width = 100

    mock_game_info.map_size.height = 100

    mock_game_info.playable_area.x = 0

    mock_game_info.playable_area.y = 0

    mock_game_info.playable_area.width = 100

    mock_game_info.playable_area.height = 100

    mock_game_info.pathing_grid.width = 100

    mock_game_info.pathing_grid.height = 100

    

    # 3. Create Units

    medivac = create_mock_unit(UnitTypeId.MEDIVAC, (20, 20), 1, is_flying=True)

    overlord = create_mock_unit(UnitTypeId.OVERLORD, (80, 80), 2, is_flying=True)

    marine = create_mock_unit(UnitTypeId.MARINE, (10, 10), 4, is_flying=False)

    

    units = [medivac, overlord, marine]

    

    # 4. Generate Mask (is_transport=False, actor is ground)

    mask = agent.get_transport_mask(units, mock_game_info, is_transport=False)

    mask_2d = mask.reshape((grid_size, grid_size))

    

    def check_near(grid, x, y):

        # 3x3 window check

        window = grid[max(0, y-1):min(32, y+2), max(0, x-1):min(32, x+2)]

        return np.any(window)



    # Medivac at (20, 20) -> (6, 25)

    assert check_near(mask_2d, 6, 25), "Ground unit should see Medivac"

    

    # Marine at (10, 10) -> (3, 28)

    assert not check_near(mask_2d, 3, 28), "Ground unit should NOT see Marine in load mask"



def test_transport_mask_generation_transport_actor():

    """

    Verifies that get_transport_mask (for transport actor) highlights ground units.

    """

    grid_size = 32

    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=grid_size, dilation_size=1)

    mock_game_info = MagicMock()

    mock_game_info.map_size.width = 100

    mock_game_info.map_size.height = 100

    mock_game_info.playable_area.x = 0

    mock_game_info.playable_area.y = 0

    mock_game_info.playable_area.width = 100

    mock_game_info.playable_area.height = 100

    mock_game_info.pathing_grid.width = 100

    mock_game_info.pathing_grid.height = 100

    

    medivac = create_mock_unit(UnitTypeId.MEDIVAC, (80, 80), 1, is_flying=True)

    marine = create_mock_unit(UnitTypeId.MARINE, (20, 20), 2, is_flying=False)

    

    # is_transport=True (actor is transport)

    mask = agent.get_transport_mask([medivac, marine], mock_game_info, is_transport=True)

    mask_2d = mask.reshape((grid_size, grid_size))

    

    def check_near(grid, x, y):

        window = grid[max(0, y-1):min(32, y+2), max(0, x-1):min(32, x+2)]

        return np.any(window)



    # Marine at (20, 20) -> (6, 25)

    assert check_near(mask_2d, 6, 25), "Transport should see Marine"

    

    # Medivac at (80, 80) -> (25, 6)

    assert not check_near(mask_2d, 25, 6), "Transport should NOT see other transport in load mask"
