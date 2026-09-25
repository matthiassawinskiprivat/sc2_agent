"""
Tests for CTF defender zone distance penalty (outside radius, moving away only).
"""
import pytest
from unittest.mock import MagicMock
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_unit(tag: int, x: float, y: float, owner_id: int = 1):
    u = MagicMock()
    u.tag = tag
    u.position = Point2((x, y))
    u.owner_id = owner_id
    u.is_structure = False
    u.health = 100.0
    u.shield = 0.0
    return u


def _make_bot(dist_mult: float = 0.5, zone_radius: float = 10.0) -> CaptureTheFlagBot:
    return CaptureTheFlagBot(
        player_id=1,
        training_conf={
            "rewards": {
                "distance_multiplier": dist_mult,
                "defender_zone_radius": zone_radius,
            }
        },
    )


def _wire_unit(bot: CaptureTheFlagBot, unit) -> None:
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit


def test_penalty_when_outside_zone_and_moving_away():
    bot = _make_bot(dist_mult=0.5)
    pylon = Point2((0.0, 0.0))
    unit = _make_unit(101, 14.0, 0.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    _wire_unit(bot, unit)

    bot._pre_action_dist_to_defend_pylon[101] = 12.0

    penalty = bot.calculate_defender_zone_penalty(101, pylon)
    assert penalty == pytest.approx(-0.5 * (14.0 - 12.0))


def test_no_penalty_inside_zone():
    bot = _make_bot()
    pylon = Point2((0.0, 0.0))
    unit = _make_unit(101, 8.0, 0.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    _wire_unit(bot, unit)

    bot._pre_action_dist_to_defend_pylon[101] = 9.0

    assert bot.calculate_defender_zone_penalty(101, pylon) == 0.0


def test_no_penalty_when_outside_but_moving_toward_pylon():
    bot = _make_bot()
    pylon = Point2((0.0, 0.0))
    unit = _make_unit(101, 12.0, 0.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    _wire_unit(bot, unit)

    bot._pre_action_dist_to_defend_pylon[101] = 14.0

    assert bot.calculate_defender_zone_penalty(101, pylon) == 0.0


def test_no_penalty_for_attackers():
    bot = _make_bot()
    pylon = Point2((0.0, 0.0))
    unit = _make_unit(101, 20.0, 0.0)
    bot.unit_command_map = {101: "attack"}
    _wire_unit(bot, unit)
    bot._pre_action_dist_to_defend_pylon[101] = 18.0

    assert bot.calculate_defender_zone_penalty(101, pylon) == 0.0


def test_no_penalty_on_first_step_without_baseline():
    bot = _make_bot()
    pylon = Point2((0.0, 0.0))
    unit = _make_unit(101, 15.0, 0.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    _wire_unit(bot, unit)

    assert bot.calculate_defender_zone_penalty(101, pylon) == 0.0


def test_record_pre_action_snapshots_defender_distance():
    bot = _make_bot()
    pylon = Point2((0.0, 0.0))
    unit = _make_unit(101, 11.5, 0.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    _wire_unit(bot, unit)
    bot.enemy_units = []
    bot.units = [unit]

    bot.record_pre_action_retreat_state(101)

    assert bot._pre_action_dist_to_defend_pylon[101] == pytest.approx(11.5)


def test_integrated_defender_penalty_in_calculate_reward():
    """Defenders pay the shared ``step_penalty`` now; the old free per-step
    ``defender_step_reward`` (farmable by hiding all episode) is gone — see
    ``src/agent/ctf_economic_rewards.py`` and ``defender_retained_value_bonus``.
    """
    bot = _make_bot(dist_mult=0.5)
    pylon_pos = Point2((0.0, 0.0))
    own_pylon = _make_unit(888, 0.0, 0.0)
    own_pylon.is_structure = True
    own_pylon.health = 20.0
    own_pylon.shield = 0.0
    enemy_pylon = _make_unit(999, 40.0, 0.0, owner_id=2)
    enemy_pylon.is_structure = True
    enemy_pylon.health = 20.0
    enemy_pylon.shield = 0.0

    def _struct_col(pylon):
        col = MagicMock()
        col.exists = pylon is not None
        col.first = pylon
        return MagicMock(return_value=col)

    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.is_visible = MagicMock(return_value=False)
    bot.structures = _struct_col(own_pylon)
    bot.enemy_structures = _struct_col(enemy_pylon)

    unit = _make_unit(101, 13.0, 0.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon_pos}
    bot.active_unit_tag = 101
    bot.units = [unit]
    bot.enemy_units = []
    _wire_unit(bot, unit)
    bot._pre_action_dist_to_defend_pylon[101] = 11.0

    reward = bot.calculate_reward(perspective_player_id=1)
    expected_penalty = -0.5 * (13.0 - 11.0)
    step_penalty = -0.05  # shared default; both roles pay it now
    assert reward == pytest.approx(expected_penalty + step_penalty)
