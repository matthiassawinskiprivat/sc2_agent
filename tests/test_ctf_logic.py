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
    """Flat HP damage_reward was removed; see test_ctf_economic_death_reward /
    test_ctf_god_mode_combat_reward for the resource-weighted replacement."""
    pytest.skip("replaced by economic value combat rewards")


def test_melee_squad_damage_reward_and_penalty_scaling():
    pytest.skip("replaced by economic value combat rewards")


def test_ranged_squad_uses_base_damage_coefficients():
    pytest.skip("replaced by economic value combat rewards")
