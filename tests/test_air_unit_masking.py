import pytest
import numpy as np
from unittest.mock import MagicMock
from src.level_7_bot import Level7Bot
from src.agent.ctf_bot import CaptureTheFlagBot

def test_level_7_air_unit_masking_unrestricted():
    """
    Verifies that for Level 7, if the active unit is an air unit, 
    both Move and Attack masks are fully open (All True).
    """
    # Setup
    training_conf = {
        "unit_type": "MEDIVAC", # Air unit
        "enemy_type": "ZERGLING",
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 128
    grid_area = 128 * 128
    
    # Mock active unit as FLYING
    mock_unit = MagicMock()
    mock_unit.is_flying = True
    bot.get_active_units = MagicMock(return_value=[mock_unit])
    bot.current_unit_index = 0
    
    # Mock agent_system to return a restricted move mask (should be ignored for air)
    restricted_move_mask = np.zeros(grid_area, dtype=bool)
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].get_action_mask.return_value = restricted_move_mask
    
    # In Level 7, the environment sets active_unit_tags and current_unit_index
    # The bot uses these to identify the "active" unit for which it provides masks.
    # Note: Level 7 bot doesn't strictly have a 'current_unit' property, 
    # but we will implement detection in action_masks.
    
    # Get combined mask
    combined_mask = bot.action_masks()
    
    # BOTH halves should be All True (Unmasked) for air units
    assert np.all(combined_mask == True), "Air units should have fully open action masks (Move + Attack)"

def test_ctf_air_unit_masking_unrestricted():
    """
    Verifies that for CTF, if the active unit is an air unit, 
    both Move and Attack masks are fully open (All True).
    """
    # Setup
    training_conf = {
        "active_layers_count": 23,
    }
    bot = CaptureTheFlagBot(player_id=1, training_conf=training_conf)
    bot.grid_size = 128
    grid_area = 128 * 128
    
    # Mock active unit as FLYING
    mock_unit = MagicMock()
    mock_unit.tag = 123
    mock_unit.is_flying = True
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"active_agent": MagicMock()}
    bot.active_unit_tag = 123
    
    # Mock self.all_units.find_by_tag(tag)
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = mock_unit
    
    # Store the tag in a way the bot can find it if needed
    # CaptureTheFlagBot current implementation of action_masks doesn't take tag, 
    # it assumes agent_system has the "active_agent".
    # We'll need to make sure the bot knows WHICH unit is active to check its is_flying status.
    
    # Mock restricted move mask
    restricted_move_mask = np.zeros(grid_area, dtype=bool)
    bot.agent_system.unit_agents["active_agent"].get_action_mask.return_value = restricted_move_mask
    
    # Get combined mask
    combined_mask = bot.action_masks()
    
    # BOTH halves should be All True (Unmasked) for air units
    assert np.all(combined_mask == True), "CTF Air units should have fully open action masks"

def test_ground_unit_masking_remains_restricted():
    """
    Ensures that ground units still have restricted move masks.
    """
    training_conf = {"unit_type": "MARINE"}
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 128
    grid_area = 128 * 128
    
    mock_unit = MagicMock()
    mock_unit.is_flying = False
    bot.get_active_units = MagicMock(return_value=[mock_unit])
    
    restricted_move_mask = np.zeros(grid_area, dtype=bool)
    # Put one True in the middle
    restricted_move_mask[grid_area // 2] = True
    
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].get_action_mask.return_value = restricted_move_mask
    
    combined_mask = bot.action_masks()
    
    # First half (Move) should match restricted_move_mask
    assert np.array_equal(combined_mask[:grid_area], restricted_move_mask)
    # Second half (Attack) should be All True (as per current Level 7 logic)
    assert np.all(combined_mask[grid_area:] == True)
