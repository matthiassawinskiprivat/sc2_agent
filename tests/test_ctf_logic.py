import pytest
from unittest.mock import MagicMock
from src.agent.ctf_bot import CaptureTheFlagBot
from sc2.ids.unit_typeid import UnitTypeId

def test_ctf_reward_calculation():
    """Test CTF specific reward components."""
    bot = CaptureTheFlagBot(player_id=1)
    bot.training_conf = {
        "rewards": {
            "pylon_damage_reward": 5.0,
            "damage_reward": 0.1,
            "pylon_damage_penalty": 5.0
        }
    }
    
    # Mock pylons
    enemy_pylon = MagicMock(); enemy_pylon.health = 5.0; enemy_pylon.shield = 0.0
    own_pylon = MagicMock(); own_pylon.health = 10.0; own_pylon.shield = 0.0
    
    mock_enemy_units_list = MagicMock()
    mock_enemy_units_list.exists = True
    mock_enemy_units_list.first = enemy_pylon
    bot.enemy_structures = MagicMock(return_value=mock_enemy_units_list)
    
    mock_own_units_list = MagicMock()
    mock_own_units_list.exists = True
    mock_own_units_list.first = own_pylon
    bot.structures = MagicMock(return_value=mock_own_units_list)
    
    bot.enemy_units = []
    bot.units = []
    
    # Initial state
    bot.prev_pylon_health[2] = 10.0 # Enemy
    bot.prev_pylon_health[1] = 10.0 # Own
    
    reward = bot.calculate_reward()
    
    # (10.0 - 5.0) * 5.0 = 25.0
    assert reward == 25.0
    assert bot.prev_pylon_health[2] == 5.0

def test_ctf_unit_damage_reward():
    bot = CaptureTheFlagBot(player_id=1)
    bot.training_conf = {"rewards": {"damage_reward": 1.0, "pylon_damage_reward": 0.0, "pylon_damage_penalty": 0.0}}
    
    bot.prev_enemy_units_health = {1: 100.0, 2: 100.0}
    
    # Mock enemy units
    u1 = MagicMock(); u1.health = 40.0; u1.shield = 0.0; u1.is_structure = False
    u2 = MagicMock(); u2.health = 40.0; u2.shield = 0.0; u2.is_structure = False
    bot.enemy_units = [u1, u2] # Total 80.0
    

    
    # Mock structures to avoid None errors
    
    # We must ensure .first returns None or a valid object with numeric health
    
    mock_pylon = MagicMock()
    
    mock_pylon.health = 10.0
    
    mock_pylon.shield = 10.0
    
    
    
    mock_list = MagicMock()
    
    mock_list.exists = True
    
    mock_list.first = mock_pylon
    
    
    
    bot.enemy_structures = MagicMock(return_value=mock_list)
    
    bot.structures = MagicMock(return_value=mock_list)
    
    bot.units = []
    
    
    
    # Ensure prev matches current to avoid pylon rewards
    
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    

    
    reward = bot.calculate_reward()
    
    # (100.0 - 80.0) * 1.0 = 20.0
    assert reward == 20.0