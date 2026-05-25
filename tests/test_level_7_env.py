import asyncio
import os
import yaml

import numpy as np
import pytest
from sc2.ids.unit_typeid import UnitTypeId
from unittest.mock import patch, MagicMock

from src.level_7_bot import Level7Bot
from train.envs.level_7_env import Level7Env

def test_level_7_env_action_space_is_doubled():
    # Grid size 16 -> 16*16 = 256. Level 7 -> 2 * 256 = 512.
    mock_cm = MagicMock()
    mock_cm.get_cumulative_active_layers.return_value = []
    mock_cm.get_cumulative_action_heads.return_value = ["move", "attack"]
    
    with patch('train.envs.level_0_env.ConfigManager', return_value=mock_cm), \
         patch('train.envs.level_0_env.Level0Env._discover_maps', return_value=['map1']):
        env = Level7Env(grid_size=16, level_name="level_7")
    
    assert env.action_space.n == 512

def test_level7_universal_apply_action_move_vs_attack_heads():
    """Universal Level7 path: MultiDiscrete head 0 -> move, head 1 -> attack (no full BotAI init)."""
    training_conf = {
        "agent_architecture": "universal",
        "action_heads": ["move_local", "attack_local"],
        "targets": 1,
    }
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.action_heads = ["move_local", "attack_local"]
    bot.last_action_was_move_by_unit = {}
    bot.last_action_was_attack_by_unit = {}
    bot.last_attack_target_tag_by_unit = {}
    bot._attack_order_weapon_cd_before = {}
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (10.0, 10.0)
    bot.do = MagicMock()
    unit = MagicMock()
    unit.tag = 1
    unit.weapon_cooldown = 0.0
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = unit
    enemies = MagicMock()
    enemies.exists = False
    enemies.__iter__.return_value = []
    bot.enemy_units = MagicMock(return_value=enemies)

    asyncio.run(bot.apply_action(1, [0, 0]))
    unit.move.assert_called_once()
    unit.attack.assert_not_called()

    unit.reset_mock()
    asyncio.run(bot.apply_action(1, [1, 0]))
    unit.attack.assert_called_once()
    unit.move.assert_not_called()


def test_level_7_bot_reward_logic():
    training_conf = {"rewards": {"damage_reward": 0.1, "kill_reward": 10.0}}
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.previous_health = {}
    bot.enemy_unit_type = UnitTypeId.ZERGLING

    enemy = MagicMock(tag=1, health=40, shield=0, health_max=50, shield_max=0)
    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [enemy]
    bot.enemy_units = MagicMock(return_value=enemies)
    bot.previous_health = {1: 50.0}

    reward = bot.calculate_combat_reward()
    assert reward == 1.0
    assert bot.previous_health[1] == 40

    enemies_kill = MagicMock()
    enemies_kill.exists = False
    enemies_kill.__iter__.return_value = []
    bot.enemy_units = MagicMock(return_value=enemies_kill)
    reward = bot.calculate_combat_reward()
    assert reward == 10.0
    assert 1 not in bot.previous_health


def test_level7_resolve_enemies_per_target_zergling_default():
    assert Level7Bot._resolve_enemies_per_target({}, UnitTypeId.ZERGLING) == 2


def test_level7_resolve_enemies_per_target_non_zergling_default():
    assert Level7Bot._resolve_enemies_per_target({}, UnitTypeId.ROACH) == 1


def test_level7_resolve_enemies_per_target_override_top_level():
    conf = {"enemies_per_target": 1}
    assert Level7Bot._resolve_enemies_per_target(conf, UnitTypeId.ZERGLING) == 1


def test_level7_resolve_enemies_per_target_override_nested_env():
    conf = {"env": {"enemies_per_target": 3}}
    assert Level7Bot._resolve_enemies_per_target(conf, UnitTypeId.ROACH) == 3


def test_level7_resolve_enemies_per_target_clamped_to_at_least_one():
    conf = {"enemies_per_target": 0}
    assert Level7Bot._resolve_enemies_per_target(conf, UnitTypeId.ZERGLING) == 1


def test_attack_level_configs_include_kiting_shaping_keys():
    """All attack level configs must have the kiting shaping parameters and new geometry rewards."""
    cfg_dir = os.path.join("d:/sc2_unit_agent", "config_universal")
    for i in range(7):
        level = f"level_attack_{i}.yaml"
        path = os.path.join(cfg_dir, level)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        rewards = data.get("env", {}).get("rewards", {})
        assert "kiting_simple_rewards" not in rewards, (
            f"{level}: kiting_simple_rewards flag should be removed"
        )
        for key in (
            "retreat_shaping_scale",
            "kiting_close_in_penalty",
            "weapon_ready_attack_bonus",
            "sweet_spot_bonus",
            "in_danger_penalty",
        ):
            assert key in rewards, f"{level} missing reward key: {key}"
            assert isinstance(rewards[key], (int, float))


def test_level7_proximity_gate_removed_no_class_constant():
    """
    Regression for fix 1: the hard-coded 10-unit proximity gate has been removed from
    Level7Env. ENEMY_PROXIMITY_RADIUS_FOR_DISTANCE_REWARD must no longer exist as a
    class attribute, confirming the dead zone is gone.
    """
    assert not hasattr(Level7Env, "ENEMY_PROXIMITY_RADIUS_FOR_DISTANCE_REWARD"), (
        "ENEMY_PROXIMITY_RADIUS_FOR_DISTANCE_REWARD still present — proximity gate not removed"
    )


def test_level7_proximity_gate_removed_no_instance_method():
    """
    Regression for fix 1: _is_unit_within_enemy_radius was the implementation of the
    proximity gate; it must no longer exist on Level7Env instances.
    """
    env = object.__new__(Level7Env)
    assert not hasattr(env, "_is_unit_within_enemy_radius"), (
        "_is_unit_within_enemy_radius still present — proximity gate not fully removed"
    )