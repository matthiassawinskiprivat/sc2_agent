import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_sc2bot_action_masks_uses_load_mask():
    """
    Verifies that SC2Bot.action_masks() uses UnitControlAgent.get_transport_mask
    for the 'load' action head.
    """
    # 1. Setup Bot
    mock_conn = MagicMock()
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "load"]})
    bot.grid_size = 32
    
    # 2. Mock Agent System and Unit Control Agent
    mock_agent_system = MagicMock()
    mock_unit_agent = MagicMock()
    
    # move_mask (all ones for simplicity)
    move_mask = np.ones(32 * 32, dtype=bool)
    mock_unit_agent.get_action_mask.return_value = move_mask
    
    # transport_mask (distinct pattern)
    transport_mask = np.zeros(32 * 32, dtype=bool)
    transport_mask[500] = True # Mark one bit
    mock_unit_agent.get_transport_mask.return_value = transport_mask
    
    mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
    bot.agent_system = mock_agent_system
    
    # 3. Setup Units
    # Medivac at (20, 20)
    medivac = MagicMock()
    medivac.type_id = UnitTypeId.MEDIVAC
    bot.units = [medivac]
    
    # 4. Mock game_info
    bot.game_info = MagicMock()
    
    # 5. Call action_masks
    final_mask = bot.action_masks()
    
    # 6. Verify Results
    # final_mask should be concatenation of [move_mask, transport_mask]
    expected_length = 2 * 32 * 32
    assert len(final_mask) == expected_length
    
    # Check move part (first half)
    assert np.array_equal(final_mask[:1024], move_mask)
    
    # Check load part (second half)
    # This should FAIL currently because SC2Bot uses unrestricted_mask (all True) for load
    assert np.array_equal(final_mask[1024:], transport_mask), "Load head should use transport mask"
