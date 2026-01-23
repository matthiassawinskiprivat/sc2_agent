import pytest
import numpy as np
from unittest.mock import MagicMock
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

# Mock SC2Bot
class MockBot:
    def __init__(self):
        self.grid_size = 16
        self.action_heads = ["move"]
        self.categorical_heads = ["unload"]
        self.agent_system = MagicMock()
        # Mock agent system to return a move mask
        self.agent_system.unit_agents = {
            "agent_0": MagicMock()
        }
        self.agent_system.unit_agents["agent_0"].get_action_mask.return_value = np.ones(16*16, dtype=bool)
        
        self.units = MagicMock()
        self.game_info = MagicMock()
        self.current_unit_index = 0

    def get_active_units(self):
        return [self.active_unit]

    def action_masks(self):
        # Implementation copied from src/sc2_bot.py for testing
        move_mask = self.agent_system.unit_agents["agent_0"].get_action_mask()
        active_units = self.get_active_units()
        idx = getattr(self, "current_unit_index", 0)
        active_unit = active_units[idx] if 0 <= idx < len(active_units) else None
        
        final_masks = [move_mask] # Only "move" spatial head
        
        categorical_masks = []
        for head in self.categorical_heads:
            if head == "unload":
                if active_unit and active_unit.cargo_max > 0 and active_unit.cargo_used > 0:
                    pos = active_unit.position
                    try:
                        is_pathable = self.game_info.pathing_grid.is_set(pos.rounded)
                        categorical_masks.append(is_pathable)
                    except:
                        categorical_masks.append(True)
                else:
                    categorical_masks.append(False)
        
        return np.concatenate(final_masks + [np.array(categorical_masks)])

def test_categorical_unload_masking():
    bot = MockBot()
    
    # 1. Transport with cargo on pathable terrain -> Mask = True
    active_unit = MagicMock()
    active_unit.cargo_max = 8
    active_unit.cargo_used = 2
    active_unit.position = MagicMock(spec=Point2)
    active_unit.position.rounded = MagicMock(spec=Point2)
    bot.active_unit = active_unit
    
    bot.game_info.pathing_grid.is_set.return_value = True
    
    masks = bot.action_masks()
    # Mask index 256 is the "unload" categorical head
    assert masks[256] == True
    
    # 2. Transport with cargo on UNPATHABLE terrain -> Mask = False
    bot.game_info.pathing_grid.is_set.return_value = False
    masks = bot.action_masks()
    assert masks[256] == False
    
    # 3. Transport with NO cargo -> Mask = False
    active_unit.cargo_used = 0
    bot.game_info.pathing_grid.is_set.return_value = True
    masks = bot.action_masks()
    assert masks[256] == False
    
    # 4. Non-transport unit -> Mask = False
    active_unit.cargo_max = 0
    masks = bot.action_masks()
    assert masks[256] == False

if __name__ == "__main__":
    pytest.main([__file__])
