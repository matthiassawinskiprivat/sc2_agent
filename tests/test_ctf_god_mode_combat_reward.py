"""God-mode alliance buckets must not invert combat reward.

After ``debug_control_enemy``, both armies often sit in ``bot.units`` and
``enemy_units`` is empty. Reward used those buckets, so shooting the opponent
looked like own HP loss (damage_penalty) and enemy HP never dropped. That
teaches the policy to idle.
"""
from unittest.mock import MagicMock

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _unit(tag, owner_id, hp, *, is_structure=False, type_id=None, x=0.0, y=0.0):
    u = MagicMock()
    u.tag = tag
    u.owner_id = owner_id
    u.health = hp
    u.shield = 0.0
    u.is_structure = is_structure
    u.type_id = type_id
    u.position = Point2((x, y))
    u.passengers = []
    u.ground_range = 6.0
    u.air_range = 6.0
    u.is_flying = True
    return u


def _bot():
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={
            "agent_architecture": "squad",
            "rewards": {
                "damage_reward": 0.2,
                "damage_penalty": 0.2,
                "pylon_damage_reward": 0.0,
                "pylon_damage_penalty": 0.0,
                "heal_reward": 0.0,
                "kill_reward": 0.0,
                "kill_penalty": 0.0,
                "distance_multiplier": 0.0,
                "step_penalty": 0.0,
                "defender_step_reward": 0.0,
                "cooldown_retreat_scale": 0.0,
            },
        },
    )
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.prev_enemy_unit_count = {1: 0, 2: 0}
    bot.prev_own_unit_count = {1: 0, 2: 0}
    bot.prev_enemy_combat_snapshot = {1: {}, 2: {}}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.unit_command_map = {}
    bot.unit_target_map = {}
    bot.active_unit_tag = None
    bot.is_visible = MagicMock(return_value=False)
    bot.structures = []
    bot.enemy_structures = []
    return bot


def test_god_mode_units_bucket_credits_enemy_hp_loss_not_own_attrition():
    """P1+P2 both in bot.units: P2 HP drop is damage_reward, not damage_penalty."""
    p1 = _unit(1, 1, 100.0, type_id=UnitTypeId.TEMPEST)
    p2 = _unit(2, 2, 150.0, type_id=UnitTypeId.CORRUPTOR)
    bot = _bot()
    bot.units = [p1, p2]
    bot.enemy_units = []
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = lambda tag: {1: p1, 2: p2}.get(tag)
    bot.unit_command_map = {1: "attack", 2: "attack"}
    bot.active_unit_tag = 1
    bot.prev_own_units_health[1] = 100.0
    bot.prev_enemy_units_health[1] = 150.0
    p2.health = 100.0

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert reward == pytest.approx(50.0 * 0.2)
    assert bot.prev_own_units_health[1] == pytest.approx(100.0)
    assert bot.prev_enemy_units_health[1] == pytest.approx(100.0)


def test_god_mode_own_hp_loss_still_penalized():
    p1 = _unit(1, 1, 80.0, type_id=UnitTypeId.TEMPEST)
    p2 = _unit(2, 2, 150.0, type_id=UnitTypeId.CORRUPTOR)
    bot = _bot()
    bot.units = [p1, p2]
    bot.enemy_units = []
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = lambda tag: {1: p1, 2: p2}.get(tag)
    bot.unit_command_map = {1: "attack"}
    bot.active_unit_tag = 1
    bot.prev_own_units_health[1] = 100.0
    bot.prev_enemy_units_health[1] = 150.0

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert reward == pytest.approx(-20.0 * 0.2)


def test_god_mode_p2_trainee_sees_p1_as_enemy():
    p1 = _unit(1, 1, 100.0, type_id=UnitTypeId.TEMPEST)
    p2 = _unit(2, 2, 150.0, type_id=UnitTypeId.CORRUPTOR)
    bot = _bot()
    bot.units = [p1, p2]
    bot.enemy_units = []
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = lambda tag: {1: p1, 2: p2}.get(tag)
    bot.unit_command_map = {2: "attack"}
    bot.active_unit_tag = 2
    bot.prev_own_units_health[2] = 150.0
    bot.prev_enemy_units_health[2] = 100.0
    p1.health = 60.0

    reward = bot.calculate_squad_step_reward([2], perspective_player_id=2)
    assert reward == pytest.approx(40.0 * 0.2)


def test_god_mode_p2_own_hp_loss_when_p1_army_wiped():
    """Owner split must stick when the enemy list is empty (P1 dead).

    Falling back to alliance buckets would swap P2 leftovers into enemy_units
    and credit own attrition as damage_reward.
    """
    p2 = _unit(2, 2, 80.0, type_id=UnitTypeId.VOIDRAY)
    bot = _bot()
    bot.units = [p2]
    bot.enemy_units = []
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = lambda tag: {2: p2}.get(tag)
    bot.unit_command_map = {2: "attack"}
    bot.active_unit_tag = 2
    bot.prev_own_units_health[2] = 150.0
    bot.prev_enemy_units_health[2] = 0.0

    reward = bot.calculate_squad_step_reward([2], perspective_player_id=2)
    assert reward == pytest.approx(-70.0 * 0.2)
    assert bot.prev_own_units_health[2] == pytest.approx(80.0)
    assert bot.prev_enemy_units_health[2] == pytest.approx(0.0)


def test_prime_combat_baselines_prevents_spawn_heal():
    p1 = _unit(1, 1, 400.0, type_id=UnitTypeId.THOR)
    p2 = _unit(2, 2, 250.0, type_id=UnitTypeId.VOIDRAY)
    bot = _bot()
    bot.training_conf["rewards"]["heal_reward"] = 0.1
    bot.units = [p1, p2]
    bot.enemy_units = []
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = lambda tag: {1: p1, 2: p2}.get(tag)
    bot.unit_command_map = {1: "attack"}
    bot.active_unit_tag = 1

    bot.prime_combat_reward_baselines()
    assert bot.prev_own_units_health[1] == pytest.approx(400.0)
    assert bot.prev_enemy_units_health[1] == pytest.approx(250.0)
    assert bot.prev_own_unit_count[1] == 1

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert reward == pytest.approx(0.0)
