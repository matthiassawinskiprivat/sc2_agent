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

def test_unload_masking_blocked_by_enemy_structure():
    """Verifies that unload is masked when unit is over an enemy building."""
    training_conf = {
        "env": {
            "action_heads": ["move"],
            "categorical_heads": ["unload"]
        }
    }
    bot = SC2Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 128
    grid_area = 128 * 128
    
    # Mock Medivac WITH cargo over an enemy Command Center
    pos = Point2((10, 10))
    mock_medivac = MagicMock()
    mock_medivac.tag = 1
    mock_medivac.cargo_max = 8
    mock_medivac.cargo_used = 2
    mock_medivac.is_flying = True
    mock_medivac.position = pos
    
    bot.all_units = MagicMock()
    def find_by_tag_mock(tag):
        if tag == 1: return mock_medivac
        return None
    bot.all_units.find_by_tag.side_effect = find_by_tag_mock
    bot.active_unit_tag = 1
    bot.state = MagicMock()
    bot.state.game_loop = 1
    
    # Mock enemy structure blocking
    mock_enemy_cc = MagicMock()
    mock_enemy_cc.position = pos
    mock_enemy_cc.radius = 2.5
    mock_enemy_cc.distance_to.return_value = 0.0
    
    # In SC2Bot.action_masks: blocking_entities = self.structures + self.enemy_structures + ...
    # We need to mock these lists so their sum contains the enemy structure
    bot.structures = []
    bot.enemy_structures = [mock_enemy_cc]
    bot.mineral_field = []
    bot.vespene_geyser = []
    
    # Mock game_info
    mock_game_info = MagicMock()
    mock_game_info.pathing_grid = MagicMock()
    mock_game_info.pathing_grid.__getitem__.return_value = 1 # Pathable
    
    with patch.object(SC2Bot, 'game_info', new_callable=PropertyMock, create=True) as mock_prop:
        mock_prop.return_value = mock_game_info
        
        # Mock agent_system
        bot.agent_system = MagicMock()
        bot.agent_system.unit_agents = {"agent_0": MagicMock()}
        bot.agent_system.unit_agents["agent_0"].get_action_mask.return_value = np.ones(grid_area, dtype=bool)
        
        mask = bot.action_masks()
        # The categorical unload mask should be the last element
        assert mask[-1] == False, "Unload should be masked when over an enemy structure"

if __name__ == "__main__":
    pytest.main([__file__])
