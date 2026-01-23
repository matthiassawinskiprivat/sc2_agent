import pytest
from unittest.mock import MagicMock
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from src.level_drop_bot import LevelDropBot

class MockUnit:
    def __init__(self, tag, type_id, pos=(0, 0), health=100, health_max=100, cargo_max=0, cargo_used=0, is_flying=False, is_structure=False):
        self.tag = tag
        self.type_id = type_id
        self.position = MagicMock()
        self.position.x = pos[0]
        self.position.y = pos[1]
        self.position_tuple = (pos[0], pos[1])
        self.health = health
        self.health_max = health_max
        self.shield = 0
        self.shield_max = 0
        self.cargo_max = cargo_max
        self.cargo_used = cargo_used
        self.is_flying = is_flying
        self.is_structure = is_structure
        self.radius = 0.5
        self.is_biological = True

    def __getitem__(self, key):
        return self.position_tuple[key]

    def distance_to(self, other):
        other_pos = other
        if hasattr(other, "position"):
            other_pos = other.position
        
        # Ensure we return a real float for comparisons
        return float(((self.position.x - other_pos.x)**2 + (self.position.y - other_pos.y)**2)**0.5)

@pytest.fixture
def drop_bot():
    conn = MagicMock()
    conf = {
        "env": {
            "unit_type": "MARINE",
            "enemy_type": "COMMANDCENTER",
            "rewards": {
                "load_reward": 1.0,
                "unload_reward": 2.0,
                "distance_reward_scale": 0.1,
                "kill_penalty": -10.0,
                "step_penalty": 0.0
            }
        }
    }
    bot = LevelDropBot(conn, conf)
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.COMMANDCENTER
    bot._objective_pos = Point2((100, 100))
    bot.enemy_structures = MagicMock()
    bot.enemy_units = MagicMock()
    return bot

def test_fake_death_penalty_on_load(drop_bot):
    """Verifies that loading a unit does NOT trigger a kill penalty."""
    medivac = MockUnit(1, UnitTypeId.MEDIVAC, pos=(10, 10), cargo_max=8, cargo_used=0, is_flying=True)
    marine = MockUnit(2, UnitTypeId.MARINE, pos=(10, 10))
    
    from sc2.units import Units
    
    # Initial state
    drop_bot.units = Units([medivac, marine], drop_bot)
    drop_bot.previous_allied_health = {1: 100, 2: 100}
    
    # Simulate Load: Marine becomes passenger, disappears from 'units'
    medivac.cargo_used = 1
    medivac.passengers = [marine]
    
    # Update units to only contain Medivac
    drop_bot.units = Units([medivac], drop_bot)
    drop_bot.all_units = Units([medivac], drop_bot)
    
    penalty = drop_bot.calculate_allied_attrition_penalty()
    
    # FIXED BEHAVIOR: Marine is in passengers, so it's NOT considered dead
    assert penalty == 0.0

def test_infantry_distance_reward(drop_bot):
    """Verifies that infantry gets centroid guidance reward but not combat objective reward."""
    marine = MockUnit(2, UnitTypeId.MARINE, pos=(10, 10))
    medivac = MockUnit(1, UnitTypeId.MEDIVAC, pos=(20, 20))
    from sc2.units import Units
    drop_bot.units = Units([marine, medivac], drop_bot)
    drop_bot.all_units = Units([marine, medivac], drop_bot)
    
    # Setup meeting guidance
    drop_bot.update_reward_states()
    assert drop_bot._transport_centroid == Point2((20, 20))
    
    # Move marine closer to Medivac centroid
    marine.position.x = 11
    marine.position.y = 11
    
    # Distance to medivac decreased. should get meeting reward.
    # meeting_reward_scale = 0.1 * 0.5 = 0.05
    reward = drop_bot.calculate_drop_reward(2)
    assert reward > 0
    assert reward == pytest.approx(min(0.5, (drop_bot.previous_meeting_distances[2] - marine.distance_to(drop_bot._transport_centroid)) * 0.05))

def test_weighted_transport_distance_reward(drop_bot):
    """Verifies that transport distance reward is weighted by cargo."""
    medivac = MockUnit(1, UnitTypeId.MEDIVAC, pos=(10, 10), cargo_max=8, cargo_used=2, is_flying=True)
    from sc2.units import Units
    drop_bot.units = Units([medivac], drop_bot)
    drop_bot.all_units = Units([medivac], drop_bot)
    
    # Move it closer to (100, 100)
    # Start at 10,10. Objective at 100,100.
    dist_1 = ((10-100)**2 + (10-100)**2)**0.5 # ~127.28
    drop_bot.previous_transport_distances = {1: dist_1}
    
    medivac.position.x = 20
    medivac.position.y = 20
    dist_2 = medivac.distance_to(Point2((100, 100))) # ~113.14
    
    drop_bot.previous_transport_cargo = {1: 2} # Prevents load reward trigger
    
    reward = drop_bot.calculate_drop_reward(1)
    
    expected_diff = dist_1 - dist_2
    expected_shaping = expected_diff * 0.1 * 2 # diff * scale * cargo
    assert reward == pytest.approx(min(1.0, expected_shaping))

def test_structure_combat_reward(drop_bot):
    """Verifies that damage to structures yields rewards."""
    cc = MockUnit(100, UnitTypeId.COMMANDCENTER, pos=(100, 100), health=1500, is_structure=True)
    drop_bot.enemy_structures = [cc]
    drop_bot.enemy_units = []
    
    # Initial state tracking
    drop_bot.previous_health = {100: 1500}
    
    # Deal damage
    cc.health = 1400
    reward = drop_bot.calculate_combat_reward()
    
    # 100 damage * 0.1 reward scale = 10.0
    assert reward == 10.0
    
    # Destroy structure
    drop_bot.enemy_structures = []
    reward = drop_bot.calculate_combat_reward()
    
    # kill_reward = 10.0
    assert reward == 10.0

def test_strict_load_reward_attribution(drop_bot):
    """Verifies that loading a unit only rewards the involved units once and uses FLAT values."""
    medivac = MockUnit(1, UnitTypeId.MEDIVAC, pos=(10, 10), cargo_max=8, cargo_used=0, is_flying=True)
    marine = MockUnit(2, UnitTypeId.MARINE, pos=(10, 10))
    from sc2.units import Units
    drop_bot.units = Units([medivac, marine], drop_bot)
    drop_bot.all_units = Units([medivac, marine], drop_bot)
    
    # Initialize state
    drop_bot.update_reward_states()
    
    # SIMULATE LOAD
    medivac.cargo_used = 1
    medivac.passengers = [marine]
    drop_bot.units = Units([medivac], drop_bot) 
    drop_bot.all_units = Units([medivac], drop_bot) 
    
    # 1. Marine turn (Active Tag = 2)
    reward_marine = drop_bot.calculate_drop_reward(2)
    assert reward_marine == 1.0 # FLAT load_reward
    
    # 2. Medivac turn (Active Tag = 1)
    reward_medivac = drop_bot.calculate_drop_reward(1)
    assert reward_medivac == 1.0 # FLAT load_reward
    
    # 3. Repeat turns - Should NOT get reward again
    assert drop_bot.calculate_drop_reward(2) == 0.0
    assert drop_bot.calculate_drop_reward(1) == 0.0

def test_structure_initialization_no_reward(drop_bot):
    """Verifies that new structures don't give instant rewards if they start damaged."""
    cc = MockUnit(100, UnitTypeId.COMMANDCENTER, pos=(100, 100), health=1000, health_max=1500, is_structure=True)
    drop_bot.enemy_structures = [cc]
    drop_bot.enemy_units = []
    
    reward = drop_bot.calculate_combat_reward()
    assert reward == 0.0
    assert drop_bot.previous_health[100] == 1000
    
    cc.health = 900
    reward_2 = drop_bot.calculate_combat_reward()
    assert reward_2 == 10.0 # 100 damage * 0.1 scale

def test_centroid_pickup_guidance(drop_bot):
    """Verifies transport moves to infantry centroid."""
    medivac = MockUnit(1, UnitTypeId.MEDIVAC, pos=(0, 0), cargo_max=8, cargo_used=0, is_flying=True)
    marine1 = MockUnit(2, UnitTypeId.MARINE, pos=(10, 10))
    marine2 = MockUnit(3, UnitTypeId.MARINE, pos=(20, 20))
    from sc2.units import Units
    drop_bot.units = Units([medivac, marine1, marine2], drop_bot)
    drop_bot.all_units = Units([medivac, marine1, marine2], drop_bot)
    drop_bot._objective_pos = Point2((1000, 1000))
    
    # Centroid of (10,10) and (20,20) is (15,15)
    drop_bot.update_reward_states()
    assert drop_bot._infantry_centroid == Point2((15, 15))
    
    # Move Medivac closer to centroid
    medivac.position.x = 5
    medivac.position.y = 5
    
    reward = drop_bot.calculate_drop_reward(1)
    
    # prev_dist is ~21.21. curr_dist is ~14.14. Diff ~7.07
    # scale is 0.05. Expected ~0.35
    assert reward > 0
    assert reward == pytest.approx(min(0.5, (21.2132 - 14.1421) * 0.05), rel=1e-3)