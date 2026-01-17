import pytest
import numpy as np
from unittest.mock import MagicMock
from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

class MockUnit:
    def __init__(self, tag, type_id, pos, cargo_max=0, cargo_used=0, is_flying=False, is_structure=False):
        self.tag = tag
        self.type_id = type_id
        self.position = Point2(pos)
        self.cargo_max = int(cargo_max)
        self.cargo_used = int(cargo_used)
        self.is_flying = bool(is_flying)
        self.is_structure = bool(is_structure)
        self.radius = 1.5
        self.footprint_radius = 5.0 if is_structure else 0.0
        self.is_mineral_field = False
        self.is_vespene_geyser = False

    def distance_to(self, pos):
        if isinstance(pos, MagicMock): return 0.0 # simplified
        return ((self.position.x - pos.x)**2 + (self.position.y - pos.y)**2)**0.5

@pytest.mark.asyncio
async def test_drop_mechanics_full_flow():
    """
    Integration test: Verifies masking transitions during a full drop sequence.
    """
    mock_conn = MagicMock()
    bot = SC2Bot(conn=mock_conn, training_conf={
        "env": {"action_heads": ["move", "load", "unload"]}
    })
    bot.grid_size = 32
    
    # 1. Setup Game State
    medivac = MockUnit(tag=100, type_id=UnitTypeId.MEDIVAC, pos=(10, 10), cargo_max=8, is_flying=True)
    marine = MockUnit(tag=101, type_id=UnitTypeId.MARINE, pos=(12, 12), cargo_max=0, is_flying=False)
    enemy_cc = MockUnit(tag=200, type_id=UnitTypeId.COMMANDCENTER, pos=(90, 90), is_structure=True)
    
    bot.units = [medivac, marine]
    bot.all_units = [medivac, marine]
    bot.enemy_structures = [enemy_cc]
    
    bot.game_info = MagicMock()
    bot.game_info.map_size.width = 128
    bot.game_info.map_size.height = 128
    bot.game_info.playable_area.x = 0
    bot.game_info.playable_area.y = 0
    bot.game_info.playable_area.width = 128
    bot.game_info.playable_area.height = 128
    bot.game_info.pathing_grid.data_numpy = np.ones((128, 128), dtype=np.uint8)
    bot.game_info.pathing_grid.width = 128
    bot.game_info.pathing_grid.height = 128

    # Mock Hierarchical system and Agent
    mock_agent_system = MagicMock()
    from src.agent.unit_control_agent import UnitControlAgent
    # Real agent for realistic mask generation
    # Dilation 2 makes it more robust for testing
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=2)
    mock_agent_system.unit_agents = {"agent_0": agent}
    bot.agent_system = mock_agent_system

    def check_near(grid, x, y):
        # Check 3x3 window
        window = grid[max(0, y-1):min(32, y+2), max(0, x-1):min(32, x+2)]
        return np.any(window)

    # STEP 1: Medivac is active, needs to LOAD Marine
    bot.get_active_units = MagicMock(return_value=[medivac])
    bot.current_unit_index = 0
    
    masks_medivac = bot.action_masks()
    # heads: move (0-1023), load (1024-2047), unload (2048-3071)
    load_mask_medivac = masks_medivac[1024:2048].reshape((32, 32))
    
    # In 128x128 world, Marine is at (12, 12). 
    # X = 12/128 * 32 = 3. Y = 28 or 29 depending on flip.
    assert check_near(load_mask_medivac, 3, 28), "Medivac should see Marine in load mask near (3, 28)"
    assert load_mask_medivac[0, 0] == False, "Mask should not be all True"

    # STEP 2: Marine is active, needs to LOAD into Medivac
    bot.get_active_units = MagicMock(return_value=[marine])
    bot.current_unit_index = 0
    
    masks_marine = bot.action_masks()
    load_mask_marine = masks_marine[1024:2048].reshape((32, 32))
    
    # Medivac is at (10, 10). X = 10/128 * 32 = 2.5 -> 2. 
    assert check_near(load_mask_marine, 2, 29), "Marine should see Medivac in load mask near (2, 29)"

    # STEP 3: Medivac is active, carrying unit, needs to UNLOAD
    medivac.cargo_used = 2
    bot.get_active_units = MagicMock(return_value=[medivac])
    bot.current_unit_index = 0
    
    masks_unload = bot.action_masks()
    unload_mask = masks_unload[2048:3072].reshape((32, 32))
    
    # Enemy CC is at (90, 90). X = 90/128 * 32 = 22.5. Y = 9.
    assert unload_mask[9, 22] == False, "Unload mask should exclude enemy buildings"
    assert unload_mask[15, 15] == True, "Unload mask should allow clear terrain"