import pytest
from unittest.mock import MagicMock
from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId

def test_calculate_allied_attrition_penalty_damage():
    training_conf = {
        "unit_type": "ROACH",
        "rewards": {
            "damage_penalty": -0.5,
            "kill_penalty": -10.0
        }
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    bot.allied_unit_type = UnitTypeId.ROACH
    
    # Mock Allied Units
    u1 = MagicMock()
    u1.tag = 101
    u1.health = 100
    u1.shield = 0
    u1.health_max = 100
    u1.shield_max = 0
    
    def units_side_effect(unit_type):
        if unit_type == bot.allied_unit_type:
            mock_col = MagicMock()
            mock_col.__iter__.return_value = [u1]
            return mock_col
        return []
    
    bot.units = MagicMock()
    bot.units.side_effect = units_side_effect
    
    # Initialize health tracking
    # We call it once to set the baseline
    bot.calculate_allied_attrition_penalty()
    assert bot.previous_allied_health[101] == 100
    
    # Damage u1
    u1.health = 90
    penalty = bot.calculate_allied_attrition_penalty()
    
    # Expected: (100 - 90) * -0.5 = -5.0
    assert penalty == -5.0
    assert bot.previous_allied_health[101] == 90

def test_calculate_allied_attrition_penalty_kill():
    training_conf = {
        "unit_type": "ROACH",
        "rewards": {
            "damage_penalty": -0.5,
            "kill_penalty": -10.0
        }
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    bot.allied_unit_type = UnitTypeId.ROACH
    
    # Mock Allied Units
    u1 = MagicMock()
    u1.tag = 101
    u1.health = 100
    u1.shield = 0
    u1.health_max = 100
    u1.shield_max = 0
    
    bot.units = MagicMock()
    # Initial state with u1
    mock_col = MagicMock()
    mock_col.__iter__.return_value = [u1]
    bot.units.side_effect = lambda t: mock_col if t == bot.allied_unit_type else []
    
    # Baseline
    bot.calculate_allied_attrition_penalty()
    
    # u1 dies (remove from units)
    mock_col_empty = MagicMock()
    mock_col_empty.__iter__.return_value = []
    bot.units.side_effect = lambda t: mock_col_empty if t == bot.allied_unit_type else []
    
    penalty = bot.calculate_allied_attrition_penalty()
    
    # Expected: -10.0
    assert penalty == -10.0
    assert 101 not in bot.previous_allied_health
