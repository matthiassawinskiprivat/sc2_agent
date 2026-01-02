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

def test_level_4_greedy_assignment():
    """Verify that the greedy assignment logic handles 4 units correctly."""
    bot = Level3Bot(conn=MagicMock(), training_conf={"targets": 4})
    
    # 4 Units in a line
    units = [MockUnit(100 + i, Point2((0, i*10))) for i in range(4)]
    # 4 Beacons in a line, slightly offset
    beacons = [MockUnit(200 + i, Point2((2, i*10))) for i in range(4)]
    
    all_units_list = units + beacons
    
    def mock_units_filter(unit_type):
        res = units if unit_type == UnitTypeId.ZERGLING else beacons
        m = MagicMock()
        m.__iter__.return_value = res
        m.exists = True
        m.closest_to = lambda u: min(beacons, key=lambda b: u.distance_to(b))
        m.__getitem__.side_effect = lambda idx: res[idx]
        return m

    bot.units = MagicMock(side_effect=mock_units_filter)
    bot.units.find_by_tag.side_effect = lambda tag: next((u for u in all_units_list if u.tag == tag), None)
    
    assignment = bot.get_closest_targets()
    
    # In this specific setup, each unit is closest to its "aligned" beacon
    for i in range(4):
        assert assignment[100 + i] == 200 + i

def test_level_4_reward_scaling():
    """Verify reward scaling matches the new individual/summed logic."""
    training_conf = {
        "targets": 4,
        "rewards": {"distance_multiplier": 1.0, "step_penalty": 0.0}
    }
    bot = Level3Bot(conn=MagicMock(), training_conf=training_conf)
    
    units = [MockUnit(100 + i, Point2((0, i*10))) for i in range(4)]
    beacons = [MockUnit(200 + i, Point2((2, i*10))) for i in range(4)]
    
    all_units_list = units + beacons
    
    def mock_units_filter(unit_type):
        res = units if unit_type == UnitTypeId.ZERGLING else beacons
        m = MagicMock()
        m.__iter__.return_value = res
        m.__len__.return_value = len(res)
        m.exists = True
        m.amount = len(res)
        m.closest_to = lambda u: min(beacons, key=lambda b: u.distance_to(b))
        m.__getitem__.side_effect = lambda idx: res[idx]
        return m

    bot.units = MagicMock(side_effect=mock_units_filter)
    bot.units.find_by_tag.side_effect = lambda tag: next((u for u in all_units_list if u.tag == tag), None)

    # Initialize
    bot.calculate_distance_reward()
    
    # Move Unit 0 exactly 1.0 unit closer to its beacon
    units[0].position = Point2((units[0].position.x + 1.0, units[0].position.y))
        
    # Test individual reward for Unit 0
    reward0 = bot.calculate_distance_reward(unit_tag=100)
    assert reward0 == pytest.approx(1.0)
    
    # Move Unit 1 exactly 2.0 units closer
    units[1].position = Point2((units[1].position.x + 2.0, units[1].position.y))
    reward1 = bot.calculate_distance_reward(unit_tag=101)
    assert reward1 == pytest.approx(2.0)