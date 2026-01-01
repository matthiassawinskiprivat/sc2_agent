import pytest
import numpy as np
from unittest.mock import MagicMock
from src.level_3_bot import Level3Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

class MockUnit:
    def __init__(self, tag, pos):
        self.tag = tag
        self.position = pos
    def distance_to(self, other):
        return self.position.distance_to(other.position)

def test_level_4_optimal_assignment():
    """Verify that the optimal assignment logic handles 4 units correctly."""
    bot = Level3Bot(conn=MagicMock(), training_conf={"targets": 4})
    
    # 4 Units in a line
    units = [MockUnit(100 + i, Point2((0, i*10))) for i in range(4)]
    # 4 Beacons in a line, slightly offset
    beacons = [MockUnit(200 + i, Point2((2, i*10))) for i in range(4)]
    
    # Mix them up to ensure assignment isn't just accidental
    # U0 is closest to B0, U1 to B1, etc.
    
    all_units_list = units + beacons
    
    def mock_units_filter(unit_type):
        res = units if unit_type == UnitTypeId.ZERGLING else beacons
        m = MagicMock()
        m.__iter__.return_value = res
        m.exists = True
        m.__getitem__.side_effect = lambda idx: res[idx]
        return m

    bot.units = MagicMock(side_effect=mock_units_filter)
    bot.units.find_by_tag.side_effect = lambda tag: next((u for u in all_units_list if u.tag == tag), None)
    
    assignment = bot._calculate_optimal_assignment()
    
    # Verify unique assignments
    assigned_beacons = set(assignment.values())
    assert len(assigned_beacons) == 4, "Each unit should have a unique beacon"
    
    # Verify obvious mapping
    for i in range(4):
        assert assignment[100 + i] == 200 + i

def test_level_4_reward_scaling():
    """Verify reward scaling remains consistent with 4 units."""
    training_conf = {
        "targets": 4,
        "rewards": {"distance_multiplier": 1.0, "step_penalty": 0.0}
    }
    bot = Level3Bot(conn=MagicMock(), training_conf=training_conf)
    
    units = [MockUnit(100 + i, Point2((0, i*10))) for i in range(4)]
    beacons = [MockUnit(200 + i, Point2((2, i*10))) for i in range(4)]
    
    all_units_list = units + beacons
    bot.units = MagicMock(side_effect=lambda t: [units, beacons][0 if t == UnitTypeId.ZERGLING else 1])
    # Need to fix the mock above to work with get_active_zerglings and calculate_distance_reward
    
    def mock_units_filter(unit_type):
        res = units if unit_type == UnitTypeId.ZERGLING else beacons
        m = MagicMock()
        m.__iter__.return_value = res
        m.__len__.return_value = len(res)
        m.exists = True
        m.amount = len(res)
        m.__getitem__.side_effect = lambda idx: res[idx]
        return m

    bot.units = MagicMock(side_effect=mock_units_filter)
    bot.units.find_by_tag.side_effect = lambda tag: next((u for u in all_units_list if u.tag == tag), None)

    # Initialize
    bot.calculate_distance_reward()
    
    # Move ALL 4 units 1.0 unit closer to their respective beacons
    for u in units:
        u.position = Point2((u.position.x + 1.0, u.position.y))
        
    reward = bot.calculate_distance_reward()
    
    # Total change = 1.0 * 4 units = 4.0
    # Avg change = 4.0 / 4 units = 1.0
    # Reward = 1.0 * multiplier (1.0) = 1.0
    assert reward == pytest.approx(1.0)
