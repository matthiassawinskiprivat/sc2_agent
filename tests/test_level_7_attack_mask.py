import pytest
import numpy as np
from unittest.mock import MagicMock
from src.level_7_bot import Level7Bot

def test_level_7_attack_mask_unmasked():
    # Setup
    training_conf = {
        "unit_type": "ROACH",
        "enemy_type": "ZERGLING",
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 128
    grid_area = 128 * 128
    
    # Mock agent_system to return a restricted move mask
    # All False (blocked)
    move_mask = np.zeros(grid_area, dtype=bool)
    
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {
        "agent_0": MagicMock()
    }
    bot.agent_system.unit_agents["agent_0"].get_action_mask.return_value = move_mask
    
    # Get combined mask
    combined_mask = bot.action_masks()
    
    # First half (Move) should be move_mask (All False)
    assert np.all(combined_mask[:grid_area] == False)
    
    # Second half (Attack) should be All True (Unmasked)
    # This should FAIL currently because Level7Bot.action_masks returns concatenate([move_mask, move_mask])
    assert np.all(combined_mask[grid_area:] == True)
