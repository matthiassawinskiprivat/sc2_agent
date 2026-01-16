"""
Integration test for SC2Bot unload masking.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId

def test_sc2bot_action_masks_uses_unload_mask():
    """
    TDD: Verifies that SC2Bot.action_masks() calls agent.get_unload_mask() 
    for the 'unload' action head.
    """
    # Create a mock for SC2Bot that only has the necessary attributes and the action_masks method
    bot = MagicMock(spec=SC2Bot)
    bot.training_conf = {
        "action_heads": ["move", "unload"]
    }
    bot.action_heads = ["move", "unload"]
    bot.grid_size = 128
    
    # Mock Agent and System
    mock_agent_system = MagicMock()
    mock_unit_agent = MagicMock()
    mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
    bot.agent_system = mock_agent_system
    
    # Mock masks
    move_mask = np.ones(128 * 128, dtype=bool)
    unload_mask = np.zeros(128 * 128, dtype=bool)
    unload_mask[0] = True # Unique pattern
    
    mock_unit_agent.get_action_mask.return_value = move_mask
    mock_unit_agent.get_unload_mask.return_value = unload_mask
    
    # Mock units and game_info
    bot.units = MagicMock()
    bot.game_info = MagicMock()
    bot.get_active_units.return_value = []
    
    # We want to test the REAL action_masks implementation from SC2Bot
    # But we use our mock 'bot' as 'self'
    final_mask = SC2Bot.action_masks(bot)
    
    # Expected final_mask should be concatenation of move_mask and unload_mask
    assert len(final_mask) == 2 * 128 * 128
    
    # Verify that get_unload_mask was called
    mock_unit_agent.get_unload_mask.assert_called_once_with(bot.units, bot.game_info)
    
    # Verify the contents of the unload part of the mask
    assert np.array_equal(final_mask[16384:], unload_mask)

def test_sc2bot_unload_mask_no_flying_bypass():
    """
    Verifies that unlike the 'move' mask, the 'unload' mask is NOT 
    bypassed for flying units.
    """
    bot = MagicMock(spec=SC2Bot)
    bot.action_heads = ["move", "unload"]
    bot.grid_size = 128
    
    mock_agent_system = MagicMock()
    mock_unit_agent = MagicMock()
    mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
    bot.agent_system = mock_agent_system
    
    # Masks
    move_mask_pathable = np.zeros(128 * 128, dtype=bool)
    unload_mask_safe = np.zeros(128 * 128, dtype=bool)
    unload_mask_safe[10] = True
    
    mock_unit_agent.get_action_mask.return_value = move_mask_pathable
    mock_unit_agent.get_unload_mask.return_value = unload_mask_safe
    
    # Mock a flying unit
    flying_unit = MagicMock()
    flying_unit.is_flying = True
    bot.get_active_units.return_value = [flying_unit]
    bot.current_unit_index = 0
    bot.units = MagicMock()
    bot.game_info = MagicMock()
    
    final_mask = SC2Bot.action_masks(bot)
    
    # head 0 (move): Should be ALL TRUE (bypassed)
    assert np.all(final_mask[:16384] == True)
    
    # head 1 (unload): Should still be unload_mask_safe (NOT bypassed)
    assert np.array_equal(final_mask[16384:], unload_mask_safe)
    assert not np.all(final_mask[16384:] == True)
