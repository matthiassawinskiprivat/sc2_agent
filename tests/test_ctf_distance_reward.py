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
        self.unit_type = UnitTypeId.ZERGLING  # Default

    def distance_to(self, other):
        if hasattr(other, 'position'):
            return self.position.distance_to(other.position)
        return self.position.distance_to(other)

@pytest.fixture
def bot():
    training_conf = {
        "rewards": {
            "distance_multiplier": 0.1
        }
    }
    bot = CaptureTheFlagBot(player_id=1, training_conf=training_conf)
    
    # Init internal state trackers usually set in on_step or similar
    bot.prev_pylon_health = {1: 150.0, 2: 150.0}
    bot.prev_enemy_units_health = {1: 100.0, 2: 100.0}
    bot.prev_own_units_health = {1: 100.0, 2: 100.0}
    bot.prev_dist_to_pylon = {1: None, 2: None}
    
    # Mocking internal structures
    # We use MagicMock for the Units collections
    bot.units = MagicMock()
    bot.enemy_structures = MagicMock()
    bot.all_units = MagicMock()
    bot.structures = MagicMock()
    bot.enemy_units = MagicMock()
    
    # Setup default "empty" behavior for these collections to avoid TypeErrors
    # When called (filter), return empty mock by default
    empty_units = MagicMock()
    empty_units.exists = False
    empty_units.first = None
    empty_units.amount = 0
    empty_units.__iter__.return_value = []
    
    bot.units.return_value = empty_units
    bot.enemy_structures.return_value = empty_units
    bot.structures.return_value = empty_units
    bot.enemy_units.return_value = empty_units
    
    return bot

def test_forward_progress_reward(bot):
    """Test that moving closer to the enemy pylon yields a positive reward."""
    unit_tag = 101
    start_pos = Point2((10, 10))
    closer_pos = Point2((20, 20))
    pylon_pos = Point2((50, 50))
    
    # 1. Setup Enemy Pylon
    enemy_pylon = MockUnit(999, pylon_pos, is_structure=True)
    enemy_pylon.unit_type = UnitTypeId.PYLON
    
    mock_enemy_structs = MagicMock()
    mock_enemy_structs.first = enemy_pylon
    mock_enemy_structs.exists = True
    bot.enemy_structures.return_value = mock_enemy_structs
    
    # 2. Setup Active Unit
    active_unit = MockUnit(unit_tag, start_pos)
    bot.active_unit_tag = unit_tag
    bot.all_units.find_by_tag.return_value = active_unit
    
    # 3. Setup Own Structures (Pylon) to avoid crash in pylon damage logic
    own_pylon = MockUnit(888, Point2((0,0)), is_structure=True)
    mock_own_structs = MagicMock()
    mock_own_structs.first = own_pylon
    mock_own_structs.exists = True
    bot.structures.return_value = mock_own_structs

    # Initialize Bot State for this test
    # We want to test the NEW logic which uses prev_dist_per_unit
    bot.prev_dist_per_unit = {}
    dist_start = start_pos.distance_to(pylon_pos)
    bot.prev_dist_per_unit[unit_tag] = dist_start
    
    # Move unit
    active_unit.position = closer_pos
    
    # Execute
    reward = bot.calculate_reward()
    
    # Assert
    dist_now = closer_pos.distance_to(pylon_pos)
    expected_diff = dist_start - dist_now # Positive
    expected_reward = expected_diff * 0.1
    
    # Note: calculate_reward accumulates other rewards too (like combat).
    # Since we didn't change health, other rewards should be 0.
    # But existing logic for "prev_dist_to_pylon" (singular) might interfere if not removed?
    # Actually, the spec says "Additive Integration". The old logic might duplicate this if not removed.
    # The existing code has "Progress Rewards (Distance to enemy pylon)" block. 
    # WE ARE REPLACING/IMPROVING THAT. The Spec says "implement ... similar to level 0".
    # If the old logic exists, we should probably replace it or ensure it doesn't double count.
    # The plan implies we are modifying calculate_reward. 
    
    assert reward == pytest.approx(expected_reward, abs=1e-5)
    assert bot.prev_dist_per_unit[unit_tag] == dist_now

def test_retreating_penalty(bot):
    unit_tag = 102
    start_pos = Point2((20, 20))
    farther_pos = Point2((10, 10))
    pylon_pos = Point2((50, 50))
    
    # Setup Pylons
    enemy_pylon = MockUnit(999, pylon_pos, is_structure=True)
    mock_enemy_structs = MagicMock()
    mock_enemy_structs.first = enemy_pylon
    mock_enemy_structs.exists = True
    bot.enemy_structures.return_value = mock_enemy_structs
    
    own_pylon = MockUnit(888, Point2((0,0)), is_structure=True)
    mock_own_structs = MagicMock()
    mock_own_structs.first = own_pylon
    mock_own_structs.exists = True
    bot.structures.return_value = mock_own_structs

    # Setup Unit
    active_unit = MockUnit(unit_tag, farther_pos)
    bot.active_unit_tag = unit_tag
    bot.all_units.find_by_tag.return_value = active_unit
    
    bot.prev_dist_per_unit = {}
    dist_start = start_pos.distance_to(pylon_pos)
    bot.prev_dist_per_unit[unit_tag] = dist_start
    
    reward = bot.calculate_reward()
    
    expected_diff = dist_start - farther_pos.distance_to(pylon_pos) # Negative
    expected_reward = expected_diff * 0.1
    
    assert reward == pytest.approx(expected_reward, abs=1e-5)

def test_multi_unit_isolation(bot):
    unit_a = 101
    unit_b = 102
    pylon_pos = Point2((50, 50))
    
    # Setup Pylons
    enemy_pylon = MockUnit(999, pylon_pos, is_structure=True)
    mock_enemy_structs = MagicMock()
    mock_enemy_structs.first = enemy_pylon
    mock_enemy_structs.exists = True
    bot.enemy_structures.return_value = mock_enemy_structs
    
    own_pylon = MockUnit(888, Point2((0,0)), is_structure=True)
    mock_own_structs = MagicMock()
    mock_own_structs.first = own_pylon
    mock_own_structs.exists = True
    bot.structures.return_value = mock_own_structs
    
    # Initial State
    bot.prev_dist_per_unit = {
        unit_a: 10.0,
        unit_b: 50.0
    }
    
    # Move Unit A
    bot.active_unit_tag = unit_a
    unit_a_obj = MockUnit(unit_a, Point2((45, 45))) # dist ~7.07
    bot.all_units.find_by_tag.return_value = unit_a_obj
    
    bot.calculate_reward()
    
    # Verify A updated, B stayed same
    assert bot.prev_dist_per_unit[unit_a] == pytest.approx(unit_a_obj.position.distance_to(pylon_pos), abs=1e-5)
    assert bot.prev_dist_per_unit[unit_b] == 50.0

def test_configuration_scaling(bot):
    # Re-init bot with multiplier 2.0
    training_conf = {"rewards": {"distance_multiplier": 2.0}}
    bot = CaptureTheFlagBot(player_id=1, training_conf=training_conf)
    
    # Mock internals again (simplified)
    bot.prev_pylon_health = {1: 150.0, 2: 150.0}
    bot.prev_enemy_units_health = {1: 100.0, 2: 100.0}
    bot.prev_own_units_health = {1: 100.0, 2: 100.0}
    
    bot.units = MagicMock()
    bot.enemy_structures = MagicMock()
    bot.all_units = MagicMock()
    bot.structures = MagicMock()
    bot.enemy_units = MagicMock()
    
    empty = MagicMock()
    empty.exists = False
    empty.first = None
    empty.__iter__.return_value = []
    bot.units.return_value = empty
    bot.enemy_structures.return_value = empty
    bot.structures.return_value = empty
    bot.enemy_units.return_value = empty
    
    unit_tag = 101
    pylon_pos = Point2((10, 0))
    start_pos = Point2((0, 0))
    end_pos = Point2((5, 0))
    
    # Setup Pylons
    enemy_pylon = MockUnit(999, pylon_pos, is_structure=True)
    mock_enemy_structs = MagicMock()
    mock_enemy_structs.first = enemy_pylon
    mock_enemy_structs.exists = True
    bot.enemy_structures.return_value = mock_enemy_structs
    
    own_pylon = MockUnit(888, Point2((0,0)), is_structure=True)
    mock_own_structs = MagicMock()
    mock_own_structs.first = own_pylon
    mock_own_structs.exists = True
    bot.structures.return_value = mock_own_structs
    
    bot.prev_dist_per_unit = {unit_tag: 10.0}
    
    bot.active_unit_tag = unit_tag
    unit_obj = MockUnit(unit_tag, end_pos)
    bot.all_units.find_by_tag.return_value = unit_obj
    
    reward = bot.calculate_reward()
    
    expected = (10.0 - 5.0) * 2.0
    assert reward == pytest.approx(expected, abs=1e-5)