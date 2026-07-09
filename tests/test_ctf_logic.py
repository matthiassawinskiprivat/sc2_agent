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


def test_melee_squad_damage_reward_and_penalty_scaling():
  from sc2.position import Point2

  bot = CaptureTheFlagBot(player_id=1)
  bot.training_conf = {
      "rewards": {
          "damage_reward": 0.2,
          "damage_penalty": 0.2,
          "melee_multiplier": 0.5,
          "pylon_damage_reward": 0.0,
          "pylon_damage_penalty": 0.0,
          "step_penalty": 0.0,
      }
  }
  bot.prev_pylon_health = {1: 20.0, 2: 20.0}
  bot.prev_enemy_units_health = {1: 100.0, 2: 100.0}
  bot.prev_own_units_health = {1: 100.0, 2: 100.0}
  bot.enemy_pylon_pos = {1: None, 2: None}
  bot.is_visible = MagicMock(return_value=False)
  bot.unit_command_map = {101: "attack"}

  melee_unit = MagicMock()
  melee_unit.tag = 101
  melee_unit.health = 100.0
  melee_unit.shield = 0.0
  melee_unit.is_structure = False
  melee_unit.position = Point2((0.0, 0.0))
  melee_unit.ground_range = 0.1
  melee_unit.air_range = 0.0
  melee_unit.is_flying = False
  melee_unit.type_id = UnitTypeId.ZEALOT

  enemy = MagicMock()
  enemy.health = 70.0
  enemy.shield = 0.0
  enemy.is_structure = False
  enemy.position = Point2((1.0, 0.0))
  enemy.ground_range = 5.0
  enemy.air_range = 0.0
  enemy.is_flying = False

  def _struct_col(pylon=None):
      col = MagicMock()
      col.exists = pylon is not None
      col.first = pylon
      return MagicMock(return_value=col)

  bot.structures = _struct_col(MagicMock(health=20.0, shield=0.0))
  bot.enemy_structures = _struct_col(None)
  bot.units = [melee_unit]
  bot.enemy_units = [enemy]
  bot.all_units = MagicMock()
  bot.all_units.find_by_tag = MagicMock(return_value=melee_unit)
  bot.unit_target_map = {}
  bot.prev_dist_per_squad = {}

  dealt_reward = bot.calculate_squad_step_reward([101], perspective_player_id=1)
  # (100 - 70) * 0.2 * (0.5 + 1) = 9.0
  assert dealt_reward == pytest.approx(9.0)

  bot.prev_enemy_units_health = {1: 70.0, 2: 100.0}
  bot.prev_own_units_health = {1: 100.0, 2: 100.0}
  melee_unit.health = 60.0
  taken_penalty = bot.calculate_squad_step_reward([101], perspective_player_id=1)
  # (100 - 60) * 0.2 * 0.5 = 4.0
  assert taken_penalty == pytest.approx(-4.0)


def test_ranged_squad_uses_base_damage_coefficients():
  from sc2.position import Point2

  bot = CaptureTheFlagBot(player_id=1)
  bot.training_conf = {
      "rewards": {
          "damage_reward": 0.2,
          "damage_penalty": 0.2,
          "melee_multiplier": 0.5,
          "pylon_damage_reward": 0.0,
          "pylon_damage_penalty": 0.0,
          "step_penalty": 0.0,
      }
  }
  bot.prev_pylon_health = {1: 20.0, 2: 20.0}
  bot.prev_enemy_units_health = {1: 100.0, 2: 100.0}
  bot.prev_own_units_health = {1: 100.0, 2: 100.0}
  bot.enemy_pylon_pos = {1: None, 2: None}
  bot.is_visible = MagicMock(return_value=False)
  bot.unit_command_map = {101: "attack"}

  ranged_unit = MagicMock()
  ranged_unit.tag = 101
  ranged_unit.health = 100.0
  ranged_unit.shield = 0.0
  ranged_unit.is_structure = False
  ranged_unit.position = Point2((0.0, 0.0))
  ranged_unit.ground_range = 5.0
  ranged_unit.air_range = 0.0
  ranged_unit.is_flying = False
  ranged_unit.type_id = UnitTypeId.MARINE

  enemy = MagicMock()
  enemy.health = 80.0
  enemy.shield = 0.0
  enemy.is_structure = False
  enemy.position = Point2((1.0, 0.0))
  enemy.ground_range = 0.1
  enemy.air_range = 0.0
  enemy.is_flying = False

  def _struct_col(pylon=None):
      col = MagicMock()
      col.exists = pylon is not None
      col.first = pylon
      return MagicMock(return_value=col)

  bot.structures = _struct_col(MagicMock(health=20.0, shield=0.0))
  bot.enemy_structures = _struct_col(MagicMock(health=20.0, shield=0.0))
  bot.units = [ranged_unit]
  bot.enemy_units = [enemy]
  bot.all_units = MagicMock()
  bot.all_units.find_by_tag = MagicMock(return_value=ranged_unit)

  reward = bot.calculate_squad_step_reward([101], perspective_player_id=1)
  # (100 - 80) * 0.2 = 4.0 without melee scaling
  assert reward == pytest.approx(4.0)