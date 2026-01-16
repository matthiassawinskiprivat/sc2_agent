import pytest
from unittest.mock import MagicMock
from src.level_drop_bot import LevelDropBot
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_drop_reward_logic_reproduction():
    """
    Reproduces the bug where LevelDropBot fails to reward unloading if the enemy structure
    is not in self.structures (it should be looking in enemy_structures or using cached pos).
    """
    bot = LevelDropBot(conn=MagicMock())
    
    # 1. Setup Mock State
    # Objective is known (found previously)
    bot.command_center_tag = 999
    bot._objective_pos = Point2((20.0, 20.0))
    
    # Structures lists
    # self.structures contains ONLY allied structures
    bot.structures = MagicMock()
    bot.structures.find_by_tag.return_value = None # Enemy CC is NOT in allied structures
    
    # self.enemy_structures might contain it if visible, or not if FoW
    # Let's simulate FoW first (not in enemy_structures either), but we have the cached position
    bot.enemy_structures = MagicMock()
    bot.enemy_structures.find_by_tag.return_value = None
    
    # 2. Setup Allied Units (Marines) near the objective
    marine1 = MagicMock()
    marine1.tag = 101
    marine1.is_flying = False
    marine1.cargo_used = 0 # Not a transport
    marine1.distance_to.return_value = 5.0 # Close enough (< 15.0)
    
    # Mock bot.units to return our marine
    bot.units = [marine1]
    
    # Setup previous state (no units in zone previously)
    bot.previous_units_in_zone = set()
    
    # Mock rewards config
    bot.training_conf = {
        "rewards": {
            "unload_reward": 10.0
        }
    }
    
    # 3. Calculate Reward
    reward = bot.calculate_drop_reward()
    
    # 4. Assert Failure (Current Behavior)
    # Because it looks in self.structures, it finds None, so reward is 0.0
    # assert reward == 0.0, "Reward should be 0.0 because of the bug (looking in allied structures)"
    
    # 5. Assert Desired Behavior (Commented out until fix)
    assert reward == 10.0

@pytest.mark.asyncio
async def test_drop_reward_logic_with_visible_enemy():
    """
    Even if the enemy structure IS visible (in enemy_structures), the current code 
    checks self.structures, so it still fails.
    """
    bot = LevelDropBot(conn=MagicMock())
    bot.command_center_tag = 999
    bot._objective_pos = Point2((20.0, 20.0))
    
    # Enemy CC is in enemy_structures
    enemy_cc = MagicMock()
    enemy_cc.position = Point2((20.0, 20.0))
    bot.enemy_structures = MagicMock()
    bot.enemy_structures.find_by_tag.return_value = enemy_cc
    
    # But NOT in allied structures
    bot.structures = MagicMock()
    bot.structures.find_by_tag.return_value = None
    
    marine1 = MagicMock()
    marine1.tag = 101
    marine1.is_flying = False
    marine1.cargo_used = 0
    marine1.distance_to.side_effect = lambda target: 5.0 if target == enemy_cc or target == Point2((20.0, 20.0)) else 100.0
    
    bot.units = [marine1]
    bot.previous_units_in_zone = set()
    bot.training_conf = {"rewards": {"unload_reward": 10.0}}
    
    reward = bot.calculate_drop_reward()
    
    assert reward == 10.0, "Reward should be 10.0 now that logic is fixed"
