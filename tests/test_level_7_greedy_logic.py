import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

@pytest.mark.asyncio
async def test_greedy_assignment_logic_combat():
    """
    Verify that Level7Bot assigns the closest enemy to each allied unit greedily.
    """
    training_conf = {
        "targets": 2,
        "unit_type": "MARINE",
        "enemy_type": "ZERGLING"
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    
    # Ally 1 at (10, 10), Ally 2 at (11, 11)
    # Enemy 1 at (0, 0), Enemy 2 at (100, 100)
    # Both allies are closer to Enemy 1.
    
    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    
    u2 = MagicMock()
    u2.tag = 2
    u2.position = Point2((11, 11))
    
    e1 = MagicMock()
    e1.tag = 101
    e1.position = Point2((0, 0))
    
    e2 = MagicMock()
    e2.tag = 102
    e2.position = Point2((100, 100))
    
    mock_units_allied = MagicMock()
    mock_units_allied.__iter__.return_value = [u1, u2]
    
    mock_units_enemy = MagicMock()
    mock_units_enemy.__iter__.return_value = [e1, e2]
    mock_units_enemy.closest_to.side_effect = lambda unit: e1
    
    def mock_units_callable(unit_type):
        if unit_type == UnitTypeId.MARINE:
            return mock_units_allied
        return MagicMock()
        
    bot.units = MagicMock(side_effect=mock_units_callable)
    
    def mock_enemy_units_callable(unit_type):
        if unit_type == UnitTypeId.ZERGLING:
            return mock_units_enemy
        return MagicMock()
    bot.enemy_units = MagicMock(side_effect=mock_enemy_units_callable)
    
    def find_by_tag(tag):
        if tag == 1: return u1
        if tag == 2: return u2
        return None
    bot.units.find_by_tag.side_effect = find_by_tag
    
    def find_enemy_by_tag(tag):
        if tag == 101: return e1
        if tag == 102: return e2
        return None
    bot.enemy_units.find_by_tag.side_effect = find_enemy_by_tag
    
    bot.get_active_units = MagicMock(return_value=[u1, u2])

    # We expect a new method get_closest_targets_combat
    assignment = bot.get_closest_targets_combat()
    
    assert assignment[1] == 101
    assert assignment[2] == 101

@pytest.mark.asyncio
async def test_greedy_reward_calculation_combat():
    """
    Verify distance reward in Level 7 uses average change in distance to closest enemies.
    """
    training_conf = {
        "targets": 2,
        "unit_type": "MARINE",
        "enemy_type": "ZERGLING",
        "rewards": {
            "distance_multiplier": 1.0,
            "step_penalty": 0.0
        }
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    
    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    u1.distance_to.side_effect = lambda p: u1.position.distance_to(p)
    
    e1 = MagicMock()
    e1.tag = 101
    e1.position = Point2((0, 0))
    
    mock_units_allied = MagicMock()
    mock_units_allied.__iter__.return_value = [u1]
    
    mock_units_enemy = MagicMock()
    mock_units_enemy.__iter__.return_value = [e1]
    mock_units_enemy.closest_to.side_effect = lambda unit: e1

    def mock_units_callable(unit_type):
        if unit_type == UnitTypeId.MARINE:
            return mock_units_allied
        return MagicMock()
    bot.units = MagicMock(side_effect=mock_units_callable)
    bot.units.find_by_tag.side_effect = lambda tag: u1 if tag == 1 else None
    
    def mock_enemy_units_callable(unit_type):
        if unit_type == UnitTypeId.ZERGLING:
            return mock_units_enemy
        return MagicMock()
    bot.enemy_units = MagicMock(side_effect=mock_enemy_units_callable)
    bot.enemy_units.find_by_tag.side_effect = lambda tag: e1 if tag == 101 else None
    
    bot.get_active_units = MagicMock(return_value=[u1])
    
    # Initial distance is sqrt(10^2 + 10^2) = 14.142
    bot.previous_distances = {1: 14.142}
    
    # Move unit closer to (5, 5) -> dist = 7.071
    u1.position = Point2((5, 5))
    
    reward = bot.calculate_distance_reward()
    
    assert pytest.approx(reward, 0.01) == 7.071
