"""AoE splash land credit: economic value of hostiles in the caster radius."""
from unittest.mock import MagicMock

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot
from src.agent.ctf_economic_rewards import DEFAULT_VALUE_NORM, DEFAULT_VESPENE_WEIGHT


def econ_value(minerals: float, vespene: float = 0.0) -> float:
    return (minerals + DEFAULT_VESPENE_WEIGHT * vespene) / DEFAULT_VALUE_NORM


MARINE = econ_value(50, 0)


def _unit(tag, type_id, x, y, owner_id, *, is_flying=False, is_structure=False):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.owner_id = owner_id
    u.position = Point2((x, y))
    u.is_flying = is_flying
    u.is_structure = is_structure
    u.is_alive = True
    u.health = 45.0
    u.health_max = 45.0
    u.shield = 0.0
    u.shield_max = 0.0
    u.passengers = []
    u.ground_range = 5.0
    u.air_range = 0.0
    return u


def _bot():
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={
            "agent_architecture": "squad",
            "rewards": {
                "aoe_land_reward": 1.0,
                "attacker_enemy_value_reward": 0.0,
                "attacker_own_value_penalty": 0.0,
                "defender_enemy_value_reward": 0.0,
                "defender_own_value_penalty": 0.0,
                "heal_value_reward": 0.0,
                "pylon_value_scale": 0.0,
                "distance_multiplier": 0.0,
                "step_penalty": 0.0,
                "cooldown_retreat_scale": 0.0,
                "detection_reward": 0.0,
                "defender_zone_leave_penalty": 0.0,
            },
        },
    )
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.is_visible = MagicMock(return_value=False)
    bot.structures = []
    bot.enemy_structures = []
    return bot


def _wire(bot, own, enemies):
    mobiles = list(own) + list(enemies)
    bot.units = mobiles
    bot.enemy_units = []
    by_tag = {u.tag: u for u in mobiles}
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = by_tag.get
    bot.unit_command_map = {u.tag: "attack" for u in own}


def test_aoe_land_pays_economic_value_of_hostiles_in_radius():
    bot = _bot()
    caster = _unit(1, UnitTypeId.HIGHTEMPLAR, 50.0, 50.0, 1)
    near_a = _unit(2, UnitTypeId.MARINE, 50.5, 50.0, 2)
    near_b = _unit(3, UnitTypeId.MARINE, 50.0, 50.4, 2)
    far = _unit(4, UnitTypeId.MARINE, 70.0, 70.0, 2)
    _wire(bot, [caster], [near_a, near_b, far])
    bot.prime_combat_reward_baselines()

    CaptureTheFlagBot._credit_aoe_splash_value(bot, caster, Point2((50.0, 50.0)))
    assert bot._step_aoe_land_value == pytest.approx(2.0 * MARINE)

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert reward == pytest.approx(2.0 * MARINE)
    metrics = bot.pop_economic_metrics()
    assert metrics["aoe_land_value"] == pytest.approx(2.0 * MARINE)


def test_aoe_land_miss_pays_zero():
    bot = _bot()
    caster = _unit(1, UnitTypeId.HIGHTEMPLAR, 50.0, 50.0, 1)
    far = _unit(2, UnitTypeId.MARINE, 70.0, 70.0, 2)
    _wire(bot, [caster], [far])
    bot.prime_combat_reward_baselines()

    CaptureTheFlagBot._credit_aoe_splash_value(bot, caster, Point2((50.0, 50.0)))
    assert bot._step_aoe_land_value == pytest.approx(0.0)
    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert reward == pytest.approx(0.0)
