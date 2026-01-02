import pytest
import asyncio
from unittest.mock import MagicMock
from src.level_3_bot import Level3Bot
from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

@pytest.mark.asyncio
async def test_level3_greedy_empty_targets():
    training_conf = {"targets": 1}
    bot = Level3Bot(conn=MagicMock(), training_conf=training_conf)
    
    u1 = MagicMock()
    u1.tag = 1
    
    # Mock units collection
    mock_units_zergling = MagicMock()
    mock_units_zergling.__iter__.return_value = [u1]
    
    mock_units_beacon = MagicMock()
    mock_units_beacon.__iter__.return_value = []
    mock_units_beacon.exists = False # Crucial check
    
    def mock_units_callable(unit_type):
        if unit_type == UnitTypeId.ZERGLING:
            return mock_units_zergling
        elif unit_type == UnitTypeId.BEACON_TERRANSMALL:
            return mock_units_beacon
        return MagicMock()

    bot.units = MagicMock(side_effect=mock_units_callable)
    bot.get_active_zerglings = MagicMock(return_value=[u1])

    # Should not crash and return empty dict
    assignment = bot.get_closest_targets()
    assert assignment == {}

@pytest.mark.asyncio
async def test_level7_greedy_empty_targets():
    training_conf = {
        "targets": 1,
        "unit_type": "MARINE",
        "enemy_type": "ZERGLING"
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    
    u1 = MagicMock()
    u1.tag = 1
    
    mock_units_allied = MagicMock()
    mock_units_allied.__iter__.return_value = [u1]
    
    mock_units_enemy = MagicMock()
    mock_units_enemy.__iter__.return_value = []
    mock_units_enemy.exists = False
    
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
    
    bot.get_active_units = MagicMock(return_value=[u1])

    # Should not crash and return empty dict
    assignment = bot.get_closest_targets_combat()
    assert assignment == {}
