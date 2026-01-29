import pytest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, PropertyMock, patch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

def test_unload_masking_no_cargo():
    """Verifies that unload is masked when unit has no cargo."""
    training_conf = {
        "env": {
            "action_heads": ["move"],
            "categorical_heads": ["unload"]
        }
    }
    bot = SC2Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 128
    grid_area = 128 * 128
    
    # Mock Medivac with NO cargo
    mock_medivac = MagicMock()
    mock_medivac.tag = 1
    mock_medivac.cargo_max = 8
    mock_medivac.cargo_used = 0
    mock_medivac.is_flying = True
    
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = mock_medivac
    bot.active_unit_tag = 1
    
    # Mock agent_system
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].get_action_mask.return_value = np.ones(grid_area, dtype=bool)
    
    mask = bot.action_masks()
    
    assert mask[-1] == False, "Unload should be masked when cargo is empty"

def test_unload_masking_with_cargo_on_pathable():
    """Verifies that unload is UNMASKED when unit has cargo on pathable terrain."""
    training_conf = {
        "env": {
            "action_heads": ["move"],
            "categorical_heads": ["unload"]
        }
    }
    bot = SC2Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 128
    grid_area = 128 * 128
    
    # Mock Medivac WITH cargo
    mock_medivac = MagicMock()
    mock_medivac.tag = 1
    mock_medivac.cargo_max = 8
    mock_medivac.cargo_used = 2
    mock_medivac.is_flying = True
    mock_medivac.position = Point2((10, 10))
    
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = mock_medivac
    bot.active_unit_tag = 1
    bot.structures = MagicMock()
    bot.structures.__add__.return_value = []
    bot.mineral_field = []
    bot.vespene_geyser = []
    bot.state = MagicMock()
    bot.state.game_loop = 1 # Not divisible by 100 to avoid logger.debug
    
    # Mock game_info
    mock_game_info = MagicMock()
    mock_game_info.pathing_grid = MagicMock()
    mock_game_info.pathing_grid.__getitem__.return_value = 1
    
    with patch.object(SC2Bot, 'game_info', new_callable=PropertyMock, create=True) as mock_prop:
        mock_prop.return_value = mock_game_info
        
        # Mock agent_system
        bot.agent_system = MagicMock()
        bot.agent_system.unit_agents = {"agent_0": MagicMock()}
        bot.agent_system.unit_agents["agent_0"].get_action_mask.return_value = np.ones(grid_area, dtype=bool)
        
        mask = bot.action_masks()
        assert mask[-1] == True, "Unload should be allowed with cargo on pathable terrain"

def test_unload_masking_on_unpathable():
    """Verifies that unload is masked on unpathable terrain (cliffs/void)."""
    training_conf = {
        "env": {
            "action_heads": ["move"],
            "categorical_heads": ["unload"]
        }
    }
    bot = SC2Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 128
    grid_area = 128 * 128
    
    # Mock Medivac WITH cargo over CLIFF
    mock_medivac = MagicMock()
    mock_medivac.tag = 1
    mock_medivac.cargo_max = 8
    mock_medivac.cargo_used = 2
    mock_medivac.is_flying = True
    mock_medivac.position = Point2((10, 10))
    
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = mock_medivac
    bot.active_unit_tag = 1
    bot.structures = MagicMock()
    bot.structures.__add__.return_value = []
    bot.mineral_field = []
    bot.vespene_geyser = []
    bot.state = MagicMock()
    bot.state.game_loop = 1
    
    # Mock game_info
    mock_game_info = MagicMock()
    mock_game_info.pathing_grid = MagicMock()
    mock_game_info.pathing_grid.__getitem__.return_value = 0
    
    with patch.object(SC2Bot, 'game_info', new_callable=PropertyMock, create=True) as mock_prop:
        mock_prop.return_value = mock_game_info
        
        # Mock agent_system
        bot.agent_system = MagicMock()
        bot.agent_system.unit_agents = {"agent_0": MagicMock()}
        bot.agent_system.unit_agents["agent_0"].get_action_mask.return_value = np.ones(grid_area, dtype=bool)
        
        mask = bot.action_masks()
        assert mask[-1] == False, "Unload should be masked on unpathable terrain"

def test_unload_masking_blocked_by_structure():
    """Verifies that unload is masked when unit is over a building."""
    training_conf = {
        "env": {
            "action_heads": ["move"],
            "categorical_heads": ["unload"]
        }
    }
    bot = SC2Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 128
    grid_area = 128 * 128
    
    # Mock Medivac WITH cargo over a Command Center
    pos = Point2((10, 10))
    mock_medivac = MagicMock()
    mock_medivac.tag = 1
    mock_medivac.cargo_max = 8
    mock_medivac.cargo_used = 2
    mock_medivac.is_flying = True
    mock_medivac.position = pos
    
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = mock_medivac
    bot.active_unit_tag = 1
    bot.state = MagicMock()
    bot.state.game_loop = 1
    
    # Mock structure blocking
    mock_cc = MagicMock()
    mock_cc.position = pos
    mock_cc.radius = 2.5
    mock_cc.distance_to.return_value = 0.0
    bot.structures = MagicMock()
    bot.structures.__add__.return_value = [mock_cc]
    bot.mineral_field = []
    bot.vespene_geyser = []
    
    # Mock game_info
    mock_game_info = MagicMock()
    mock_game_info.pathing_grid = MagicMock()
    mock_game_info.pathing_grid.__getitem__.return_value = 1
    
    with patch.object(SC2Bot, 'game_info', new_callable=PropertyMock, create=True) as mock_prop:
        mock_prop.return_value = mock_game_info
        
        # Mock agent_system
        bot.agent_system = MagicMock()
        bot.agent_system.unit_agents = {"agent_0": MagicMock()}
        bot.agent_system.unit_agents["agent_0"].get_action_mask.return_value = np.ones(grid_area, dtype=bool)
        
        mask = bot.action_masks()
        assert mask[-1] == False, "Unload should be masked when over a structure"

def test_unit_tag_priority_in_masking():
    """Verifies that active_unit_tag overrides current_unit_index."""
    training_conf = {"env": {"action_heads": ["move"], "categorical_heads": []}}
    bot = SC2Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 128
    
    unit1 = MagicMock()
    unit1.tag = 101
    unit1.is_flying = False
    unit1.cargo_max = 0
    
    unit2 = MagicMock()
    unit2.tag = 202
    unit2.is_flying = True # Flying should result in all-True mask
    unit2.cargo_max = 0
    
    bot.all_units = MagicMock()
    def find_by_tag_mock(tag):
        if tag == 202: return unit2
        return None
    bot.all_units.find_by_tag.side_effect = find_by_tag_mock
    
    bot.active_unit_tag = 202
    bot.current_unit_index = 0
    bot.get_active_units = MagicMock(return_value=[unit1, unit2])
    
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].get_action_mask.return_value = np.zeros(128*128, dtype=bool)
    
    mask = bot.action_masks()
    assert np.all(mask == True), "Should have used unit2 (flying) via active_unit_tag"