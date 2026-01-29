import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_sc2bot_action_masks_uses_load_mask_context():
    """
    Verifies that SC2Bot.action_masks() uses UnitControlAgent.get_transport_mask
    with correct is_transport context.
    """
    # 1. Setup Bot
    mock_conn = MagicMock()
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "load"]})
    bot.grid_size = 32
    
    # 2. Mock Agent System and Unit Control Agent
    mock_agent_system = MagicMock()
    mock_unit_agent = MagicMock()
    
    move_mask = np.ones(32 * 32, dtype=bool)
    mock_unit_agent.get_action_mask.return_value = move_mask
    
    transport_mask_for_medivac = np.zeros(32 * 32, dtype=bool)
    transport_mask_for_medivac[100] = True
    
    transport_mask_for_marine = np.zeros(32 * 32, dtype=bool)
    transport_mask_for_marine[200] = True
    
    # Side effect to return different masks based on is_transport
    def get_transport_mask_side_effect(units, game_info, is_transport=False):
        if is_transport:
            return transport_mask_for_medivac
        return transport_mask_for_marine
        
    mock_unit_agent.get_transport_mask.side_effect = get_transport_mask_side_effect
    
    mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
    bot.agent_system = mock_agent_system
    
    # 3. Setup Units
    medivac = MagicMock()
    medivac.cargo_max = 8
    medivac.is_flying = True
    
    marine = MagicMock()
    marine.cargo_max = 0
    marine.is_flying = False
    
    bot.get_active_units = MagicMock(return_value=[medivac, marine])
    bot.units = [medivac, marine]
    bot.game_info = MagicMock()
    
    # CASE A: Active unit is Medivac (is_transport=True)
    bot.current_unit_index = 0
    final_mask_a = bot.action_masks()
    # Medivac part of load mask should be transport_mask_for_medivac
    assert np.array_equal(final_mask_a[1024:], transport_mask_for_medivac)
    mock_unit_agent.get_transport_mask.assert_called_with(bot.units, bot.game_info, is_transport=True)
    
    # CASE B: Active unit is Marine (is_transport=False)
    bot.current_unit_index = 1
    final_mask_b = bot.action_masks()
    # Marine part of load mask should be transport_mask_for_marine
    assert np.array_equal(final_mask_b[1024:], transport_mask_for_marine)
    mock_unit_agent.get_transport_mask.assert_called_with(bot.units, bot.game_info, is_transport=False)