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

def test_optimal_assignment_integration():
    bot = Level3Bot(conn=MagicMock(), training_conf={"targets": 2})
    
    # Setup units in a way that greedy might fail or at least verify optimal logic
    u0 = MockUnit(100, Point2((0, 0)))
    u1 = MockUnit(101, Point2((0, 10)))
    b0 = MockUnit(200, Point2((0, 1)))
    b1 = MockUnit(201, Point2((0, 9)))
    
    # Mock bot.units
    def units_side_effect(unit_type):
        mock_coll = MagicMock()
        if unit_type == UnitTypeId.ZERGLING:
            units_list = [u0, u1]
            mock_coll.__iter__.return_value = units_list
            mock_coll.exists = True
            mock_coll.__getitem__.side_effect = lambda idx: units_list[idx]
        elif unit_type == UnitTypeId.BEACON_TERRANSMALL:
            beacons_list = [b0, b1]
            mock_coll.__iter__.return_value = beacons_list
            mock_coll.exists = True
            mock_coll.__getitem__.side_effect = lambda idx: beacons_list[idx]
        return mock_coll
        
    bot.units = MagicMock(side_effect=units_side_effect)
    
    # This method should be implemented in Level3Bot
    assignment = bot._calculate_optimal_assignment()
    
    # Expected: {100: 200, 101: 201}
    assert assignment[100] == 200
    assert assignment[101] == 201

def test_get_assigned_distances():
    bot = Level3Bot(conn=MagicMock(), training_conf={"targets": 2})
    
    # Unit 0 at (0,0), Unit 1 at (0,10)
    # Beacon 0 at (0,2), Beacon 1 at (0,8)
    
    # U0 is closer to B0 (dist 2)
    # U1 is closer to B1 (dist 2)
    
    u0 = MockUnit(100, Point2((0, 0)))
    u1 = MockUnit(101, Point2((0, 10)))
    b0 = MockUnit(200, Point2((0, 2)))
    b1 = MockUnit(201, Point2((0, 8)))
    
    all_units_list = [u0, u1, b0, b1]
    
    # Create the units collection object
    units_mock = MagicMock()
    
    def units_call_side_effect(unit_type):
        mock_coll = MagicMock()
        if unit_type == UnitTypeId.ZERGLING:
            units_list = [u0, u1]
            mock_coll.__iter__.return_value = units_list
            mock_coll.exists = True
            mock_coll.__getitem__.side_effect = lambda idx: units_list[idx]
        elif unit_type == UnitTypeId.BEACON_TERRANSMALL:
            beacons_list = [b0, b1]
            mock_coll.__iter__.return_value = beacons_list
            mock_coll.exists = True
            mock_coll.__getitem__.side_effect = lambda idx: beacons_list[idx]
        return mock_coll

    units_mock.side_effect = units_call_side_effect
    units_mock.find_by_tag.side_effect = lambda tag: next((u for u in all_units_list if u.tag == tag), None)
    
    bot.units = units_mock
    
    dists = bot.get_assigned_distances()
    
    # U0 (100) -> B0 (200), dist = 2.0
    # U1 (101) -> B1 (201), dist = 2.0
    assert dists[100] == 2.0
    assert dists[101] == 2.0
    
    # Now swap beacons such that crossing is optimal (though distance matrix remains same, assignment should be stable)
    # Actually let's test a case where greedy nearest unit would pick the "wrong" one.
    
    # U0 (0,0), U1 (0, 3)
    # B0 (0, 2), B1 (0, 10)
    
    # Greedy Nearest:
    # U1 is closest to B0 (dist 1). So U1 takes B0.
    # Then U0 must take B1 (dist 10).
    # Sum = 11.
    
    # Optimal:
    # U0 takes B0 (dist 2).
    # U1 takes B1 (dist 7).
    # Sum = 9. (Better!)
    
    u0.position = Point2((0, 0))
    u1.position = Point2((0, 3))
    b0.position = Point2((0, 2))
    b1.position = Point2((0, 10))
    
    dists = bot.get_assigned_distances()
    
    # We expect Optimal: U0->B0 (2.0), U1->B1 (7.0)
    assert dists[100] == 2.0
    assert dists[101] == 7.0

def test_calculate_distance_reward():
    training_conf = {
        "targets": 2,
        "rewards": {
            "distance_multiplier": 1.0,
            "step_penalty": 0.0 # Simplify
        }
    }
    bot = Level3Bot(conn=MagicMock(), training_conf=training_conf)
    
    u0 = MockUnit(100, Point2((0, 0)))
    u1 = MockUnit(101, Point2((0, 3)))
    b0 = MockUnit(200, Point2((0, 2)))
    b1 = MockUnit(201, Point2((0, 10)))
    
    all_units_list = [u0, u1, b0, b1]
    
    def mock_units_filter(unit_type):
        if unit_type == UnitTypeId.ZERGLING:
            res = [u0, u1]
        elif unit_type == UnitTypeId.BEACON_TERRANSMALL:
            res = [b0, b1]
        else:
            res = []
        
        m = MagicMock()
        m.__iter__.return_value = res
        m.__len__.return_value = len(res)
        m.exists = len(res) > 0
        m.amount = len(res)
        m.__getitem__.side_effect = lambda idx: res[idx]
        return m

    units_mock = MagicMock()
    units_mock.side_effect = mock_units_filter
    units_mock.find_by_tag.side_effect = lambda tag: next((u for u in all_units_list if u.tag == tag), None)
    
    bot.units = units_mock
    
    # First call to initialize previous_distances
    bot.calculate_distance_reward()
    assert bot.previous_distances[100] == 2.0 # U0->B0
    assert bot.previous_distances[101] == 7.0 # U1->B1
    
    # Move U0 closer to B0: (0, 0) -> (0, 1) [dist 2 -> 1]
    # Move U1 further from B1: (0, 3) -> (0, 0) [dist 7 -> 10]
    u0.position = Point2((0, 1))
    u1.position = Point2((0, 0))
    
    reward = bot.calculate_distance_reward()
    
    # Change U0: 2.0 - 1.0 = 1.0
    # Change U1: 7.0 - 10.0 = -3.0
    # Total change = -2.0
    # Avg change = -2.0 / 2 units = -1.0
    # Reward = (-1.0 * multiplier) + penalty = -1.0
    
    assert reward == -1.0


