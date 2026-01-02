import pytest
import numpy as np
from unittest.mock import MagicMock
from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

def test_level_7_distance_reward_calculation():
    # Setup
    training_conf = {
        "unit_type": "ROACH",
        "enemy_type": "ZERGLING",
        "rewards": {
            "distance_multiplier": 0.05,
            "step_penalty": -0.01
        }
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    
    # Mock Allied Units
    a1 = MagicMock()
    a1.tag = 101
    a1.position = Point2((10, 10))
    
    # Mock Enemy Units
    e1 = MagicMock()
    e1.tag = 201
    e1.position = Point2((20, 10)) # Distance 10
    
    allies_list = [a1]
    enemies_list = [e1]

    # Setup bot.units and bot.enemy_units
    def units_side_effect(unit_type):
        mock_collection = MagicMock()
        if unit_type == bot.allied_unit_type:
            mock_collection.__iter__.return_value = iter(allies_list)
            mock_collection.__getitem__.side_effect = lambda i: allies_list[i]
            mock_collection.amount = 1
            mock_collection.__len__.return_value = 1
        return mock_collection

    def enemy_units_side_effect(unit_type):
        mock_collection = MagicMock()
        if unit_type == bot.enemy_unit_type:
            mock_collection.__iter__.return_value = iter(enemies_list)
            mock_collection.__getitem__.side_effect = lambda i: enemies_list[i]
            mock_collection.amount = 1
            mock_collection.__len__.return_value = 1
        return mock_collection
        
    bot.units = MagicMock()
    bot.units.side_effect = units_side_effect
    
    def find_by_tag_side_effect(tag):
        if tag == 101: return a1
        return None
    bot.units.find_by_tag.side_effect = find_by_tag_side_effect

    bot.enemy_units = MagicMock()
    bot.enemy_units.side_effect = enemy_units_side_effect
    
    def enemy_find_by_tag_side_effect(tag):
        if tag == 201: return e1
        return None
    bot.enemy_units.find_by_tag.side_effect = enemy_find_by_tag_side_effect
    
    # Also need to mock distance_to
    def dist_to(other):
        return a1.position.distance_to(other.position)
    a1.distance_to.side_effect = dist_to
    
    # Initial distance (simulated by calling get_assigned_distances)
    # Level7Bot inherits from Level3Bot, but it might not have the same 
    # internal state for previous_distances yet.
    
    # First call to establish baseline
    reward1 = bot.calculate_distance_reward()
    assert reward1 == -0.01 # Only step penalty as previous_distances is empty
    
    # Move ally closer
    a1.position = Point2((15, 10)) # Distance 5 (changed by 5)
    
    # Second call
    reward2 = bot.calculate_distance_reward()
    
    # Expected: (10 - 5) * 0.05 + (-0.01) = 5 * 0.05 - 0.01 = 0.25 - 0.01 = 0.24
    assert pytest.approx(reward2) == 0.24
