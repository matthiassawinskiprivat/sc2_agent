import pytest
import numpy as np
from unittest.mock import MagicMock
from src.agent.unit_control_agent import UnitControlAgent
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

def create_mock_unit(unit_type, position, is_structure=False, is_flying=False):
    unit = MagicMock()
    unit.type_id = unit_type
    unit.position = Point2(position)
    unit.is_structure = is_structure
    unit.is_flying = is_flying
    unit.is_mineral_field = False
    unit.is_vespene_geyser = False
    unit.radius = 0.75
    if is_structure:
        unit.footprint_radius = 1.5
    return unit

def test_get_transport_mask_as_medivac():
    """
    TDD: When the actor is a Medivac, the load mask should highlight 
    ground units (Marines), NOT other Medivacs.
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

    # Actor: Medivac
    # Units on map: Marine at (20, 20), Another Medivac at (80, 80)
    marine = create_mock_unit(UnitTypeId.MARINE, (20, 20), is_flying=False)
    other_medivac = create_mock_unit(UnitTypeId.MEDIVAC, (80, 80), is_flying=True)
    
    # We need to pass the actor context. Currently get_transport_mask doesn't support this.
    # This call is expected to FAIL or behave incorrectly until we update the code.
    mask = agent.get_transport_mask([marine, other_medivac], mock_game_info, is_transport=True)
    
    mask_2d = mask.reshape((grid_size, grid_size))
    
    # Check Marine region (20, 20) -> should be TRUE
    # X: 20/100 * 32 = 6.4 -> 6
    # Y: 20/100 -> 0.2 (World). Flipped for Image: 0.8 * 32 = 25.6 -> 25
    idx_x, idx_y = 6, 25
    assert mask_2d[idx_y, idx_x] == True, "Medivac should see Marine in load mask"
    
    # Check Other Medivac region (80, 80) -> should be FALSE
    idx_x_m, idx_y_m = 25, 6
    assert mask_2d[idx_y_m, idx_x_m] == False, "Medivac should NOT see another Medivac in load mask"

def test_get_transport_mask_as_marine():
    """
    TDD: When the actor is a Marine, the load mask should highlight 
    transports (Medivacs), NOT other Marines.
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

    # Actor: Marine
    # Units on map: Medivac at (20, 20), Another Marine at (80, 80)
    medivac = create_mock_unit(UnitTypeId.MEDIVAC, (20, 20), is_flying=True)
    other_marine = create_mock_unit(UnitTypeId.MARINE, (80, 80), is_flying=False)
    
    mask = agent.get_transport_mask([medivac, other_marine], mock_game_info, is_transport=False)
    
    mask_2d = mask.reshape((grid_size, grid_size))
    
    # Check Medivac region -> should be TRUE
    assert mask_2d[25, 6] == True, "Marine should see Medivac in load mask"
    
    # Check Other Marine region -> should be FALSE
    assert mask_2d[6, 25] == False, "Marine should NOT see another Marine in load mask"

def test_get_unload_mask_kernel_and_enemies():
    """
    TDD: Verifies unload mask uses a 3x3 kernel (less aggressive) 
    and correctly masks out enemy buildings.
    """
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=128, dilation_size=0)
    
    # Create 128x128 map, all unpathable
    pathing = np.zeros((128, 128), dtype=np.uint8)
    # Create a small 6x6 pathable island (small island to test 3x3 vs 5x5)
    # A 5x5 erosion would destroy a 6x6 island (eroding 2px from all sides)
    # A 3x3 erosion (1px from all sides) should leave a 4x4 core.
    pathing[60:66, 60:66] = 1
    
    mock_game_info = MagicMock()
    mock_game_info.pathing_grid.data_numpy = pathing
    mock_game_info.pathing_grid.width = 128
    mock_game_info.pathing_grid.height = 128
    mock_game_info.map_size.width = 128
    mock_game_info.map_size.height = 128
    mock_game_info.playable_area.x = 0
    mock_game_info.playable_area.y = 0
    mock_game_info.playable_area.width = 128
    mock_game_info.playable_area.height = 128
    
    # Enemy building at center of island (63, 63)
    enemy_cc = create_mock_unit(UnitTypeId.COMMANDCENTER, (63, 63), is_structure=True)
    
    unload_mask = agent.get_unload_mask([enemy_cc], mock_game_info)
    unload_mask_2d = unload_mask.reshape((128, 128))
    
    # The center of the 6x6 island is (62.5, 62.5). Indices 60,61,62,63,64,65.
    # If it was 5x5 erosion (2px margin): 60+2=62, 65-2=63. Range [62, 63].
    # But wait, erosion removes pixels. 6x6 eroded by 2px -> 2x2 remains.
    # If 3x3 erosion (1px margin): 6x6 -> 4x4 remains.
    
    # Check if a 4x4 core exists
    # Index for Y=61, X=61 (1px in from edge)
    # In world (61, 61), index is (127-61)*128 + 61 = 66*128 + 61
    assert unload_mask_2d[127-61, 61] == True, "3x3 erosion should leave a 4x4 island from 6x6"
    
    # Check enemy building footprint is MASKED
    # CC has radius ~1.5 (footprint). 
    assert unload_mask_2d[127-63, 63] == False, "Enemy building should be masked out"
