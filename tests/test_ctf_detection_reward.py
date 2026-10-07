"""Concealed hostiles in allied detector range pay each step, value-weighted."""
from unittest.mock import MagicMock

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot
from src.agent.ctf_economic_rewards import DEFAULT_VALUE_NORM, DEFAULT_VESPENE_WEIGHT


def econ_value(minerals: float, vespene: float = 0.0) -> float:
    return (minerals + DEFAULT_VESPENE_WEIGHT * vespene) / DEFAULT_VALUE_NORM


BANSHEE = econ_value(150, 100)


def _unit(
    tag,
    type_id,
    x,
    y,
    owner_id,
    *,
    is_cloaked=False,
    is_burrowed=False,
    is_detected=False,
    detect_range=0.0,
):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.owner_id = owner_id
    u.position = Point2((x, y))
    u.is_structure = False
    u.is_cloaked = is_cloaked
    u.is_burrowed = is_burrowed
    u.is_detected = is_detected
    u.detect_range = detect_range
    u.health = 140.0
    u.health_max = 140.0
    u.shield = 0.0
    u.shield_max = 0.0
    u.passengers = []
    u.ground_range = 6.0
    u.air_range = 6.0
    u.is_flying = type_id == UnitTypeId.BANSHEE
    return u


def _bot():
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={
            "agent_architecture": "squad",
            "rewards": {
                "detection_reward": 1.0,
                "attacker_enemy_value_reward": 0.0,
                "attacker_own_value_penalty": 0.0,
                "defender_enemy_value_reward": 0.0,
                "defender_own_value_penalty": 0.0,
                "heal_value_reward": 0.0,
                "pylon_damage_reward": 0.0,
                "pylon_damage_penalty": 0.0,
                "distance_multiplier": 0.0,
                "step_penalty": 0.0,
                "cooldown_retreat_scale": 0.0,
                "aoe_land_reward": 0.0,
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


def test_coverage_pays_every_step_while_concealed_units_stay_in_range():
    bot = _bot()
    raven = _unit(1, UnitTypeId.RAVEN, 0.0, 0.0, 1, detect_range=11.0)
    banshee = _unit(
        2,
        UnitTypeId.BANSHEE,
        5.0,
        0.0,
        2,
        is_cloaked=True,
    )
    _wire(bot, [raven], [banshee])
    bot.prime_combat_reward_baselines()

    first = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    second = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert first == pytest.approx(BANSHEE)
    assert second == pytest.approx(BANSHEE)


def test_two_concealed_in_range_pay_twice_one():
    bot = _bot()
    raven = _unit(1, UnitTypeId.RAVEN, 0.0, 0.0, 1, detect_range=11.0)
    a = _unit(2, UnitTypeId.BANSHEE, 3.0, 0.0, 2, is_cloaked=True)
    b = _unit(3, UnitTypeId.BANSHEE, 4.0, 0.0, 2, is_cloaked=True)
    _wire(bot, [raven], [a, b])
    bot.prime_combat_reward_baselines()

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert reward == pytest.approx(2.0 * BANSHEE)


def test_out_of_detector_range_does_not_pay_even_if_is_detected():
    bot = _bot()
    raven = _unit(1, UnitTypeId.RAVEN, 0.0, 0.0, 1, detect_range=11.0)
    banshee = _unit(
        2,
        UnitTypeId.BANSHEE,
        40.0,
        0.0,
        2,
        is_cloaked=True,
        is_detected=True,
    )
    _wire(bot, [raven], [banshee])
    bot.prime_combat_reward_baselines()

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert reward == pytest.approx(0.0)


def test_visible_uncloaked_does_not_pay():
    bot = _bot()
    raven = _unit(1, UnitTypeId.RAVEN, 0.0, 0.0, 1, detect_range=11.0)
    banshee = _unit(
        2,
        UnitTypeId.BANSHEE,
        5.0,
        0.0,
        2,
        is_cloaked=False,
        is_detected=True,
    )
    _wire(bot, [raven], [banshee])
    bot.prime_combat_reward_baselines()

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert reward == pytest.approx(0.0)
