import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from src.level_3_bot import Level3Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

@pytest.mark.asyncio
async def test_greedy_assignment_logic():
    """
    Verify that Level3Bot assigns the closest target to each unit greedily,
    allowing multiple units to target the same beacon.
    """
    training_conf = {"targets": 2}
    bot = Level3Bot(conn=MagicMock(), training_conf=training_conf)
    
    # Unit 1 at (10, 10), Unit 2 at (11, 11)
    # Beacon 1 at (0, 0), Beacon 2 at (100, 100)
    # Both units are closer to Beacon 1.
    
    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    
    u2 = MagicMock()
    u2.tag = 2
    u2.position = Point2((11, 11))
    
    b1 = MagicMock()
    b1.tag = 101
    b1.position = Point2((0, 0))
    
    b2 = MagicMock()
    b2.tag = 102
    b2.position = Point2((100, 100))
    
    # Mock for Units collection
    mock_units_zergling = MagicMock()
    mock_units_zergling.__iter__.return_value = [u1, u2]
    
    mock_units_beacon = MagicMock()
    mock_units_beacon.__iter__.return_value = [b1, b2]
    mock_units_beacon.closest_to.side_effect = lambda unit: b1
    
    def mock_units_callable(unit_type):
        if unit_type == UnitTypeId.ZERGLING:
            return mock_units_zergling
        elif unit_type == UnitTypeId.BEACON_TERRANSMALL:
            return mock_units_beacon
        return MagicMock()

    bot.units = MagicMock(side_effect=mock_units_callable)
    # Also mock find_by_tag on the same bot.units object
    def find_by_tag(tag):
        if tag == 1: return u1
        if tag == 2: return u2
        if tag == 101: return b1
        if tag == 102: return b2
        return None
    bot.units.find_by_tag.side_effect = find_by_tag
    
    bot.get_active_zerglings = MagicMock(return_value=[u1, u2])

    # We expect a new method or updated behavior
    assignment = bot.get_closest_targets()
    
    assert assignment[1] == 101
    assert assignment[2] == 101

@pytest.mark.asyncio
async def test_greedy_reward_calculation():
    """
    Verify reward is calculated as average change in distance to closest targets.
    """
    training_conf = {
        "targets": 2,
        "rewards": {
            "distance_multiplier": 1.0,
            "step_penalty": 0.0
        }
    }
    bot = Level3Bot(conn=MagicMock(), training_conf=training_conf)
    
    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    u1.distance_to.side_effect = lambda p: u1.position.distance_to(p)
    
    b1 = MagicMock()
    b1.tag = 101
    b1.position = Point2((0, 0))
    
    mock_units_zergling = MagicMock()
    mock_units_zergling.__iter__.return_value = [u1]
    
    mock_units_beacon = MagicMock()
    mock_units_beacon.__iter__.return_value = [b1]
    mock_units_beacon.closest_to.side_effect = lambda unit: b1

    def mock_units_callable(unit_type):
        if unit_type == UnitTypeId.ZERGLING:
            return mock_units_zergling
        elif unit_type == UnitTypeId.BEACON_TERRANSMALL:
            return mock_units_beacon
        return MagicMock()

    bot.units = MagicMock(side_effect=mock_units_callable)
    def find_by_tag(tag):
        if tag == 1: return u1
        if tag == 101: return b1
        return None
    bot.units.find_by_tag.side_effect = find_by_tag
    
    bot.get_active_zerglings = MagicMock(return_value=[u1])
    
    # Initial distance is sqrt(10^2 + 10^2) = 14.142
    bot.previous_distances = {1: 14.142}
    
    # Move unit closer to (5, 5) -> dist = 7.071
    u1.position = Point2((5, 5))
    
    reward = bot.calculate_distance_reward()
    
    # Expected reward = 14.142 - 7.071 = 7.071
    assert pytest.approx(reward, 0.01) == 7.071
