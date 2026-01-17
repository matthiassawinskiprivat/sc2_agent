import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_sc2bot_action_masks_uses_unload_mask_with_enemies():
    """
    Verifies that SC2Bot.action_masks() calls agent.get_unload_mask() 
    with a combined list of allied units and enemy structures.
    """
    # 1. Setup Bot
    mock_conn = MagicMock()
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "unload"]})
    bot.grid_size = 32
    
    # 2. Mock Agent System and Unit Control Agent
    mock_agent_system = MagicMock()
    mock_unit_agent = MagicMock()
    
    move_mask = np.ones(32 * 32, dtype=bool)
    mock_unit_agent.get_action_mask.return_value = move_mask
    
    unload_mask = np.zeros(32 * 32, dtype=bool)
    unload_mask[123] = True
    mock_unit_agent.get_unload_mask.return_value = unload_mask
    
    mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
    bot.agent_system = mock_agent_system
    
    # 3. Setup Units
    ally = MagicMock()
    ally.tag = 1
    ally.cargo_max = 0
    ally.is_flying = False
    
    enemy_struct = MagicMock()
    enemy_struct.tag = 2
    enemy_struct.is_structure = True
    
    bot.all_units = [ally]
    bot.units = [ally]
    bot.enemy_structures = [enemy_struct]
    bot.game_info = MagicMock()
    
    bot.get_active_units = MagicMock(return_value=[ally])
    bot.current_unit_index = 0
    
    # 4. Call action_masks
    final_mask = bot.action_masks()
    
    # 5. Verify Results
    assert np.array_equal(final_mask[1024:], unload_mask)
    
    # Verify the argument to get_unload_mask contains both ally and enemy
    args, kwargs = mock_unit_agent.get_unload_mask.call_args
    passed_units = list(args[0])
    assert ally in passed_units
    assert enemy_struct in passed_units
    assert len(passed_units) == 2