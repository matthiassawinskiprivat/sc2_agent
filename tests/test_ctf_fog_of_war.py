import pytest
from unittest.mock import MagicMock
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
from src.agent.ctf_bot import CaptureTheFlagBot

class MockUnit:
    def __init__(self, tag, pos, is_structure=False):
        self.tag = tag
        self.position = pos
        self.is_structure = is_structure
        self.health = 100
        self.shield = 50
        self.unit_type = UnitTypeId.ZERGLING

    def distance_to(self, other):
        if hasattr(other, 'position'):
            return self.position.distance_to(other.position)
        return self.position.distance_to(other)

@pytest.fixture
def bot():
    training_conf = {"rewards": {"distance_multiplier": 1.0, "pylon_damage_reward": 5.0, "pylon_damage_penalty": 5.0}}
    bot = CaptureTheFlagBot(player_id=1, training_conf=training_conf)
    
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0, 2: 0}
    bot.prev_own_units_health = {1: 0, 2: 0}
    
    # Mocking collections
    empty = MagicMock()
    empty.exists = False
    empty.first = None
    empty.__iter__.return_value = []
    
    # Own structures (Pylon alive)
    own_pylon = MockUnit(999, Point2((0,0)), is_structure=True)
    own_pylon.health = 10.0
    own_pylon.shield = 10.0 # Total 20
    
    own_structs = MagicMock()
    own_structs.exists = True
    own_structs.first = own_pylon
    own_structs.__iter__.return_value = [own_pylon]
    own_structs.return_value = own_structs # Handle filtering
    
    bot.structures = MagicMock(return_value=own_structs)
    
    # Enemy structures (Default empty/hidden)
    bot.enemy_structures = MagicMock(return_value=empty)
    
    bot.units = MagicMock(return_value=empty)
    bot.enemy_units = MagicMock(return_value=empty)
    bot.all_units = MagicMock()
    
    bot.is_visible = MagicMock(return_value=True) # Default
    return bot

def test_distance_reward_fog_of_war(bot):
    """
    Test distance reward works when pylon is hidden (via persistent position).
    EXPECTED TO FAIL on current codebase: The bot currently skips reward if pylon is not visible.
    """
    pylon_pos = Point2((50, 50))
    unit_tag = 101
    
    # 1. Setup Persistent Position (simulate previously seen)
    bot.enemy_pylon_pos[1] = pylon_pos
    
    # 2. Setup Enemy Pylon MISSING from observation (Fog)
    bot.enemy_structures.return_value.exists = False
    
    # Mock Not Visible (Fog)
    bot.is_visible.side_effect = lambda p: False if p == pylon_pos else True
    
    # 3. Setup Active Unit
    start_pos = Point2((10, 10))
    closer_pos = Point2((20, 20))
    unit = MockUnit(unit_tag, closer_pos)
    bot.active_unit_tag = unit_tag
    bot.all_units.find_by_tag.return_value = unit
    
    # 4. Setup Prev Distance
    bot.prev_dist_per_unit[unit_tag] = start_pos.distance_to(pylon_pos)
    
    # Execute
    reward = bot.calculate_reward()
    
    # Assert
    dist_change = start_pos.distance_to(pylon_pos) - closer_pos.distance_to(pylon_pos)
    assert reward == pytest.approx(dist_change * 1.0)

def test_hp_reward_fog_of_war(bot):
    """
    Test NO HP reward when pylon is missing but position is NOT visible (Hidden).
    EXPECTED TO FAIL on current codebase: The bot incorrectly thinks HP dropped to 0.
    """
    pylon_pos = Point2((50, 50))
    bot.enemy_pylon_pos[1] = pylon_pos
    bot.prev_pylon_health[2] = 20.0
    
    # Enemy Pylon Missing
    bot.enemy_structures.return_value.exists = False
    
    # Position NOT Visible
    bot.is_visible.side_effect = lambda p: False if p == pylon_pos else True
    
    reward = bot.calculate_reward()
    
    # Should maintain previous HP, so reward is 0. 
    # Current code gives (20-0)*5 = 100
    assert reward == 0.0
    assert bot.prev_pylon_health[2] == 20.0

def test_hp_reward_pylon_dead(bot):
    """
    Test HP reward triggers when pylon is missing AND position IS visible (Confirmed Dead).
    This should PASS on both versions, but ensures we don't break the valid destruction case.
    """
    pylon_pos = Point2((50, 50))
    bot.enemy_pylon_pos[1] = pylon_pos
    bot.prev_pylon_health[2] = 20.0
    
    # Enemy Pylon Missing
    bot.enemy_structures.return_value.exists = False
    
    # Position IS Visible
    bot.is_visible.return_value = True
    
    reward = bot.calculate_reward()
    
    # Should update HP to 0, reward = 20 * 5.0 = 100
    assert reward == 100.0
    assert bot.prev_pylon_health[2] == 0.0
