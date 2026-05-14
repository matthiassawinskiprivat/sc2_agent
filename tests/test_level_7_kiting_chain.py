"""Level7 combat: consecutive attacks keep combat reward signal."""

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
    bot._allied_took_damage_this_step = {}
    bot._pending_rewards = {}
    bot._enemy_last_known_pos = {}
    return bot


def test_consecutive_attacks_keep_damage_reward():
    """Consecutive attacks on the same direct-target enemy each earn their damage reward."""
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
    e.position = MagicMock()
    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e]
    bot.enemy_units = MagicMock(return_value=enemies)
    bot.previous_health = {201: 10.0}

    a1, p1 = bot.calculate_unit_centric_combat_delta(101)
    assert (a1, p1) == (pytest.approx(2.0), pytest.approx(0.0))

    bot.previous_health[201] = 8.0
    e.health = 6

    bot.last_action_was_attack_by_unit[101] = True
    a2, p2 = bot.calculate_unit_centric_combat_delta(101)
    assert (a2, p2) == (pytest.approx(2.0), pytest.approx(0.0))
