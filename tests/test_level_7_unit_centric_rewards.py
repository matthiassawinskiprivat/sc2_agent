import asyncio

import pytest
from unittest.mock import MagicMock

from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId

from src.level_7_bot import Level7Bot
from train.envs.level_7_env import Level7Env


def _bare_bot(training_conf: dict) -> Level7Bot:
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.previous_health = {}
    bot.previous_allied_health = {}
    bot.previous_danger_counts = {}
    bot.last_attack_target_tag_by_unit = {}
    bot.last_action_was_attack_by_unit = {}
    bot._kite_damage_credit_ok = {}
    bot._micro_clean_attack_streak = {}
    bot._allied_took_damage_this_step = {}
    return bot


def test_unit_centric_combat_delta_attributes_single_target_damage():
    bot = _bare_bot({"rewards": {"damage_reward": 1.0, "kill_reward": 10.0}})

    # Acting unit attacked enemy 201.
    bot.last_action_was_attack_by_unit[101] = True
    bot.last_attack_target_tag_by_unit[101] = 201

    e = MagicMock()
    e.tag = 201
    e.health = 8
    e.shield = 0
    e.health_max = 10
    e.shield_max = 0

    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e]
    bot.enemy_units = MagicMock(return_value=enemies)

    # Baseline previous hp: 10 -> 8 => 2 damage -> reward 2.0 attributed.
    bot.previous_health = {201: 10.0}

    attributed, pooled = bot.calculate_unit_centric_combat_delta(101)
    assert attributed == pytest.approx(2.0)
    assert pooled == pytest.approx(0.0)


def test_unit_centric_combat_delta_attributes_target_and_pools_remaining_damage():
    bot = _bare_bot({"rewards": {"damage_reward": 1.0, "kill_reward": 10.0}})
    bot.last_action_was_attack_by_unit[101] = True
    bot.last_attack_target_tag_by_unit[101] = 201

    e1 = MagicMock(tag=201, health=8, shield=0, health_max=10, shield_max=0)
    e2 = MagicMock(tag=202, health=9, shield=0, health_max=10, shield_max=0)

    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e1, e2]
    bot.enemy_units = MagicMock(return_value=enemies)

    bot.previous_health = {201: 10.0, 202: 10.0}

    attributed, pooled = bot.calculate_unit_centric_combat_delta(101)
    assert attributed == pytest.approx(2.0)
    assert pooled == pytest.approx(1.0)


def test_unit_centric_combat_delta_attributes_single_target_kill():
    bot = _bare_bot({"rewards": {"damage_reward": 1.0, "kill_reward": 10.0}})
    bot.last_action_was_attack_by_unit[101] = True
    bot.last_attack_target_tag_by_unit[101] = 201

    enemies = MagicMock()
    enemies.exists = False
    enemies.__iter__.return_value = []
    bot.enemy_units = MagicMock(return_value=enemies)

    # Previously had enemy 201 alive.
    bot.previous_health = {201: 3.0}

    attributed, pooled = bot.calculate_unit_centric_combat_delta(101)
    assert attributed == pytest.approx(10.0)
    assert pooled == pytest.approx(0.0)


def test_apply_action_attack_records_nearest_enemy_tag_for_attribution():
    bot = _bare_bot(
        {
            "agent_architecture": "universal",
            "action_heads": ["move_local", "attack_local"],
        }
    )
    bot.action_heads = ["move_local", "attack_local"]
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (10.0, 10.0)
    bot.last_action_was_move_by_unit = {}
    bot.do = MagicMock()

    unit = MagicMock()
    unit.tag = 101
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = unit

    e1 = MagicMock()
    e1.tag = 201
    e1.position = Point2((8.0, 8.0))
    e2 = MagicMock()
    e2.tag = 202
    e2.position = Point2((40.0, 40.0))
    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e1, e2]
    bot.enemy_units = MagicMock(return_value=enemies)

    asyncio.run(bot.apply_action(101, [1, 0]))
    assert bot.last_attack_target_tag_by_unit[101] == 201
    unit.attack.assert_called_once()


def test_global_pool_flush_helper_flushes_only_on_done():
    env = object.__new__(Level7Env)
    env._global_reward_pool = 5.0

    r = env._maybe_flush_global_pool(1.0, cycle_completed=False, done=False)
    assert r == pytest.approx(1.0)
    assert env._global_reward_pool == pytest.approx(5.0)

    r = env._maybe_flush_global_pool(1.0, cycle_completed=True, done=False)
    assert r == pytest.approx(1.0)
    assert env._global_reward_pool == pytest.approx(5.0)

    env._global_reward_pool = 7.0
    r = env._maybe_flush_global_pool(2.0, cycle_completed=False, done=True)
    assert r == pytest.approx(9.0)
    assert env._global_reward_pool == pytest.approx(0.0)

