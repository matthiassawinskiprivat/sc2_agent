"""God-mode alliance buckets must not invert economic combat reward.

After ``debug_control_enemy``, both armies often sit in ``bot.units`` and
``enemy_units`` is empty. Reward used those buckets, so shooting the opponent
looked like own value loss (a penalty) and enemy value never dropped. That
teaches the policy to idle.

Reward is resource-weighted: damage pays ``fraction_of_value_removed × coeff``,
so these tests assert against each unit's real SC2 price rather than its hit
points. See ``src/agent/ctf_economic_rewards.py``.
"""
from unittest.mock import MagicMock

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot
from src.agent.ctf_economic_rewards import (
    DEFAULT_VALUE_NORM,
    DEFAULT_VESPENE_WEIGHT,
)

# Coefficients the fixture below uses, kept apart from level YAML on purpose.
ENEMY_COEFF = 3.0
OWN_COEFF = 1.0


def econ_value(minerals: float, vespene: float) -> float:
    """Normalised economic value, mirroring ``unit_economic_value``."""
    return (minerals + DEFAULT_VESPENE_WEIGHT * vespene) / DEFAULT_VALUE_NORM


# Prices of the units used here (minerals, vespene), from train/envs/ctf/game_data.
TEMPEST = econ_value(300, 200)
CORRUPTOR = econ_value(150, 100)
VOIDRAY = econ_value(250, 150)
THOR = econ_value(300, 200)


def _unit(tag, owner_id, hp, *, max_hp=None, is_structure=False, type_id=None,
          x=0.0, y=0.0):
    u = MagicMock()
    u.tag = tag
    u.owner_id = owner_id
    u.health = hp
    u.health_max = max_hp if max_hp is not None else hp
    u.shield = 0.0
    u.shield_max = 0.0
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
                "attacker_enemy_value_reward": ENEMY_COEFF,
                "attacker_own_value_penalty": OWN_COEFF,
                "defender_enemy_value_reward": ENEMY_COEFF,
                "defender_own_value_penalty": OWN_COEFF,
                "melee_multiplier": 0.0,
                "pylon_damage_reward": 0.0,
                "pylon_damage_penalty": 0.0,
                "heal_value_reward": 0.0,
                "defender_retained_value_bonus": 0.0,
                "distance_multiplier": 0.0,
                "step_penalty": 0.0,
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
    bot.prev_enemy_value_snapshot = {1: {}, 2: {}}
    bot.prev_own_value_snapshot = {1: {}, 2: {}}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.unit_command_map = {}
    bot.unit_target_map = {}
    bot.active_unit_tag = None
    bot.is_visible = MagicMock(return_value=False)
    bot.structures = []
    bot.enemy_structures = []
    return bot


def _wire(bot, units):
    """Put every unit in ``bot.units`` (the god-mode bucket) and index it by tag."""
    bot.units = list(units)
    bot.enemy_units = []
    by_tag = {u.tag: u for u in units}
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = by_tag.get


def test_god_mode_units_bucket_credits_enemy_value_loss_not_own_attrition():
    """P1+P2 both in bot.units: P2 value drop is a reward, not a penalty."""
    p1 = _unit(1, 1, 100.0, type_id=UnitTypeId.TEMPEST)
    p2 = _unit(2, 2, 150.0, type_id=UnitTypeId.CORRUPTOR)
    bot = _bot()
    _wire(bot, [p1, p2])
    bot.unit_command_map = {1: "attack", 2: "attack"}
    bot.active_unit_tag = 1
    bot.prime_combat_reward_baselines()

    p2.health = 100.0  # lost a third of its vitality
    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)

    assert reward == pytest.approx(ENEMY_COEFF * (1 / 3) * CORRUPTOR)


def test_god_mode_own_value_loss_still_penalized():
    p1 = _unit(1, 1, 100.0, type_id=UnitTypeId.TEMPEST)
    p2 = _unit(2, 2, 150.0, type_id=UnitTypeId.CORRUPTOR)
    bot = _bot()
    _wire(bot, [p1, p2])
    bot.unit_command_map = {1: "attack"}
    bot.active_unit_tag = 1
    bot.prime_combat_reward_baselines()

    p1.health = 80.0  # lost a fifth of its vitality
    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)

    assert reward == pytest.approx(-OWN_COEFF * 0.2 * TEMPEST)


def test_god_mode_p2_trainee_sees_p1_as_enemy():
    p1 = _unit(1, 1, 100.0, type_id=UnitTypeId.TEMPEST)
    p2 = _unit(2, 2, 150.0, type_id=UnitTypeId.CORRUPTOR)
    bot = _bot()
    _wire(bot, [p1, p2])
    bot.unit_command_map = {2: "attack"}
    bot.active_unit_tag = 2
    bot.prime_combat_reward_baselines()

    p1.health = 60.0  # P1 is the enemy from P2's perspective: -40%
    reward = bot.calculate_squad_step_reward([2], perspective_player_id=2)

    assert reward == pytest.approx(ENEMY_COEFF * 0.4 * TEMPEST)


def test_god_mode_p2_own_value_loss_when_p1_army_wiped():
    """Owner split must stick when the enemy list is empty (P1 dead).

    Falling back to alliance buckets would swap P2 leftovers into enemy_units
    and credit own attrition as an enemy-damage reward.
    """
    p2 = _unit(2, 2, 150.0, type_id=UnitTypeId.VOIDRAY)
    bot = _bot()
    _wire(bot, [p2])
    bot.unit_command_map = {2: "attack"}
    bot.active_unit_tag = 2
    bot.prime_combat_reward_baselines()

    p2.health = 80.0
    reward = bot.calculate_squad_step_reward([2], perspective_player_id=2)

    expected_fraction = (150.0 - 80.0) / 150.0
    assert reward == pytest.approx(-OWN_COEFF * expected_fraction * VOIDRAY)


def test_expensive_unit_damage_outweighs_cheap_unit_damage():
    """The whole point of economic shaping: price, not hit points, sets the payout.

    A Thor and a Zergling each lose half their vitality. The Thor is worth
    600 value to the Zergling's 25, so it must pay far more.
    """
    thor = _unit(1, 2, 400.0, type_id=UnitTypeId.THOR)
    ling = _unit(2, 2, 35.0, type_id=UnitTypeId.ZERGLING)
    own = _unit(3, 1, 100.0, type_id=UnitTypeId.TEMPEST)
    bot = _bot()
    _wire(bot, [own, thor, ling])
    bot.unit_command_map = {3: "attack"}
    bot.active_unit_tag = 3
    bot.prime_combat_reward_baselines()

    thor.health = 200.0
    thor_reward = bot.calculate_squad_step_reward([3], perspective_player_id=1)

    ling.health = 17.5
    ling_reward = bot.calculate_squad_step_reward([3], perspective_player_id=1)

    assert thor_reward == pytest.approx(ENEMY_COEFF * 0.5 * THOR)
    assert ling_reward == pytest.approx(ENEMY_COEFF * 0.5 * econ_value(25, 0))
    assert thor_reward > 20 * ling_reward


def test_death_pays_only_the_value_still_standing():
    """Per-step damage plus the death must sum to exactly the unit's full value.

    A flat kill bonus on top of damage shaping would double-count the health
    that was already paid for.
    """
    enemy = _unit(2, 2, 150.0, type_id=UnitTypeId.CORRUPTOR)
    own = _unit(1, 1, 100.0, type_id=UnitTypeId.TEMPEST)
    bot = _bot()
    _wire(bot, [own, enemy])
    bot.unit_command_map = {1: "attack"}
    bot.active_unit_tag = 1
    bot.prime_combat_reward_baselines()

    enemy.health = 50.0  # two thirds gone
    damage_reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)

    _wire(bot, [own])  # the remaining third dies
    death_reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)

    assert damage_reward + death_reward == pytest.approx(ENEMY_COEFF * CORRUPTOR)


def test_prime_combat_baselines_prevents_spawn_heal():
    p1 = _unit(1, 1, 400.0, type_id=UnitTypeId.THOR)
    p2 = _unit(2, 2, 250.0, type_id=UnitTypeId.VOIDRAY)
    bot = _bot()
    bot.training_conf["rewards"]["heal_value_reward"] = 4.0
    _wire(bot, [p1, p2])
    bot.unit_command_map = {1: "attack"}
    bot.active_unit_tag = 1

    bot.prime_combat_reward_baselines()
    assert bot.prev_own_value_snapshot[1].keys() == {1}
    assert bot.prev_enemy_value_snapshot[1].keys() == {2}

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert reward == pytest.approx(0.0)


def test_unpriced_unit_type_is_not_free_to_lose():
    """An unmodelled type must fall back to a nonzero price.

    A silent 0 would make the unit worthless to kill and free to lose, which is
    a far worse failure than a slightly wrong price.
    """
    own = _unit(1, 1, 100.0, type_id=None)
    bot = _bot()
    _wire(bot, [own])
    bot.unit_command_map = {1: "attack"}
    bot.active_unit_tag = 1
    bot.prime_combat_reward_baselines()

    own.health = 50.0
    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)

    assert reward < 0.0
