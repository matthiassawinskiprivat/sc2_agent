"""Universal Level7: mask attack command heads when no enemy appears in tactical fovea."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from sc2.position import Point2

from src.level_7_bot import Level7Bot
from src.sc2_bot import SC2Bot
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM


class _FakeEnemyGroup:
    def __init__(self, units):
        self._units = list(units)

    @property
    def exists(self):
        return len(self._units) > 0

    def __iter__(self):
        return iter(self._units)


def _noop_iv(self):
    pass


@pytest.fixture
def universal_bot():
    training_conf = {
        "agent_architecture": "universal",
        "action_heads": ["move_global", "move_local", "attack_local"],
        "unit_type": "COLOSSUS",
        "enemy_type": "ROACH",
    }
    with patch.object(SC2Bot, "_initialize_variables", _noop_iv, create=True):
        bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    bot.training_conf = training_conf
    bot.action_heads = training_conf["action_heads"]

    mock_agent = MagicMock()
    mock_agent.get_action_mask.return_value = np.ones(32 * 32, dtype=bool)
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": mock_agent}

    ally = MagicMock()
    ally.position = Point2((20.0, 24.0))
    ally.is_flying = False
    ally.tag = 99
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.active_unit_tag = ally.tag
    bot.zergling = ally

    return bot


def test_attack_unmasked_when_enemy_centered_in_tactical(universal_bot):
    bot = universal_bot
    enemy = MagicMock()
    enemy.position = Point2((20.0, 24.0))
    enemy.is_cloaked = False
    enemy.is_detected = True
    bot.enemy_units = MagicMock(return_value=_FakeEnemyGroup([enemy]))

    m = bot.action_masks()
    assert len(m) == FIXED_UNIVERSAL_COMMAND_DIM + 1024
    assert m[0] and m[1] and m[2]
    assert not m[3:FIXED_UNIVERSAL_COMMAND_DIM].any()
    assert m[FIXED_UNIVERSAL_COMMAND_DIM:].all()


def test_attack_masked_when_enemy_outside_tactical(universal_bot):
    bot = universal_bot
    enemy = MagicMock()
    enemy.position = Point2((60.0, 24.0))
    enemy.is_cloaked = False
    enemy.is_detected = True
    bot.enemy_units = MagicMock(return_value=_FakeEnemyGroup([enemy]))

    m = bot.action_masks()
    assert len(m) == FIXED_UNIVERSAL_COMMAND_DIM + 1024
    assert m[0] and m[1]
    assert not m[2]
    assert not m[3:FIXED_UNIVERSAL_COMMAND_DIM].any()


def test_attack_masked_when_cloaked_undetected(universal_bot):
    bot = universal_bot
    enemy = MagicMock()
    enemy.position = Point2((20.0, 24.0))
    enemy.is_cloaked = True
    enemy.is_detected = False
    enemy.is_snapshot = False
    bot.enemy_units = MagicMock(return_value=_FakeEnemyGroup([enemy]))

    m = bot.action_masks()
    assert not m[2]
    assert not m[3:FIXED_UNIVERSAL_COMMAND_DIM].any()


def test_attack_unmasked_for_cloaked_snapshot(universal_bot):
    bot = universal_bot
    enemy = MagicMock()
    enemy.position = Point2((20.0, 24.0))
    enemy.is_cloaked = True
    enemy.is_detected = False
    enemy.is_snapshot = True
    bot.enemy_units = MagicMock(return_value=_FakeEnemyGroup([enemy]))

    m = bot.action_masks()
    assert m[2]
    assert not m[3:FIXED_UNIVERSAL_COMMAND_DIM].any()
