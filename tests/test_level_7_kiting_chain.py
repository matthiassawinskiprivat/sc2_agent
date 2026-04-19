"""Shoot → move away → shoot kiting chain (Level7 combat; always on)."""

import pytest
from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId

from src.level_7_bot import Level7Bot


def _bare_bot(training_conf: dict) -> Level7Bot:
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ROACH
    bot.previous_health = {}
    bot.previous_allied_health = {}
    bot.previous_danger_counts = {}
    bot.last_attack_target_tag_by_unit = {}
    bot.last_action_was_attack_by_unit = {}
    bot._kite_damage_credit_ok = {}
    bot._kite_phase = {}
    bot._kite_streak = {}
    bot._kiting_prev_dist = {}
    bot._kiting_chain_bonus_pending = {}
    bot._combat_reward_suppressed = {}
    return bot


def _simulate_attack_gating(bot: Level7Bot, unit_tag: int) -> None:
    """Mirror universal apply_action kiting gating for tests."""
    if not bot._kite_damage_credit_ok.get(unit_tag, True):
        bot._combat_reward_suppressed[unit_tag] = True
    else:
        bot._combat_reward_suppressed[unit_tag] = False
        bot._kite_damage_credit_ok[unit_tag] = False


def test_consecutive_attacks_suppress_damage_reward():
    bot = _bare_bot(
        {
            "rewards": {
                "damage_reward": 1.0,
                "kill_reward": 10.0,
            }
        }
    )
    bot.last_action_was_attack_by_unit[101] = True
    bot.last_attack_target_tag_by_unit[101] = 201

    e = MagicMock(tag=201, health=8, shield=0, health_max=10, shield_max=0)
    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e]
    bot.enemy_units = MagicMock(return_value=enemies)
    bot.previous_health = {201: 10.0}

    _simulate_attack_gating(bot, 101)
    a1, p1 = bot.calculate_unit_centric_combat_delta(101)
    assert (a1, p1) == (pytest.approx(2.0), pytest.approx(0.0))

    bot.previous_health[201] = 8.0
    e.health = 6

    bot.last_action_was_attack_by_unit[101] = True
    _simulate_attack_gating(bot, 101)
    a2, p2 = bot.calculate_unit_centric_combat_delta(101)
    assert (a2, p2) == (0.0, 0.0)


def test_shoot_move_away_shoot_micro_bonus_escalates():
    bot = _bare_bot(
        {
            "rewards": {"micro_bonus": 2.0},
        }
    )
    bot.last_action_was_attack_by_unit[101] = True
    bot.update_kiting_chain_after_step(101, None, None, None)
    assert bot._kite_phase[101] == "after_shoot_need_move"
    assert bot._kite_streak.get(101, 0) == 0
    assert bot._kiting_chain_bonus_pending.get(101, 0.0) == 0.0

    bot.last_action_was_attack_by_unit[101] = False
    bot.update_kiting_chain_after_step(101, None, 10.0, 11.0)
    assert bot._kite_phase[101] == "after_move_need_shoot"

    bot.last_action_was_attack_by_unit[101] = True
    bot.update_kiting_chain_after_step(101, None, None, None)
    assert bot._kiting_chain_bonus_pending.get(101, 0.0) == pytest.approx(2.0)
    assert bot._kite_streak[101] == 1

    bot.last_action_was_attack_by_unit[101] = False
    bot.update_kiting_chain_after_step(101, None, 11.0, 12.5)

    bot.last_action_was_attack_by_unit[101] = True
    bot.update_kiting_chain_after_step(101, None, None, None)
    assert bot._kiting_chain_bonus_pending.get(101, 0.0) == pytest.approx(2.0 + 4.0)


def test_move_closer_resets_kiting_streak():
    bot = _bare_bot(
        {"rewards": {"micro_bonus": 2.0}}
    )
    bot.last_action_was_attack_by_unit[101] = True
    bot.update_kiting_chain_after_step(101, None, None, None)
    bot.last_action_was_attack_by_unit[101] = False
    bot.update_kiting_chain_after_step(101, None, 10.0, 12.0)
    bot.last_action_was_attack_by_unit[101] = True
    bot.update_kiting_chain_after_step(101, None, None, None)
    assert bot._kite_streak[101] == 1

    bot.last_action_was_attack_by_unit[101] = False
    bot.update_kiting_chain_after_step(101, None, 12.0, 10.0)
    assert bot._kite_streak[101] == 0
    assert bot._kite_phase[101] == "idle"
