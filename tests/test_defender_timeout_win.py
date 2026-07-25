"""Regression tests: defender-only episodes win on timeout when pylon survives.

The bug: _agent_attack_units_alive() returns 0 for a pure-defender (no attack
units), so every timeout resolves as draw_penalty even when the pylon is intact.
The fix: _p1_defended_on_timeout(r_conf) returns win_reward when
  rewards.defender_wins_on_timeout: true AND P1 owns a flag pylon AND pylon
  survived.
"""

import pytest
from unittest.mock import MagicMock
from sc2.ids.unit_typeid import UnitTypeId
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

WIN = 15.0
DRAW = -10.0
LOSS = -12.5

_REWARDS_DEFENDER = {
    "win_reward": WIN,
    "loss_penalty": LOSS,
    "draw_penalty": DRAW,
    "defender_wins_on_timeout": True,
}

_REWARDS_NO_FLAG = {
    "win_reward": WIN,
    "loss_penalty": LOSS,
    "draw_penalty": DRAW,
}


@pytest.fixture
def env():
    e = CaptureTheFlagSingleEnv(visualize=False, max_steps=100)
    e.bot = MagicMock()
    e.bot.structures = MagicMock()
    e.bot.enemy_structures = MagicMock()
    e.bot.unit_command_map = {}
    e.unit_pools_config = [
        {"agent": True, "high_level_command": "defend", "region": "B"},
        {"high_level_command": "attack", "region": "B"},
    ]
    e.player_units = {1: [10], 2: [20]}
    e.unit_pool_index = {}
    # Tag 10 is a defender — must be explicit so _agent_attack_units_alive returns 0.
    e.bot.unit_command_map = {10: "defend"}
    return e


def _pylon_alive(env):
    env.bot.structures(UnitTypeId.PYLON).amount = 1
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 0


def _pylon_destroyed(env):
    env.bot.structures(UnitTypeId.PYLON).amount = 0
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 0


# ---------------------------------------------------------------------------
# Unit tests for _p1_defended_on_timeout
# ---------------------------------------------------------------------------

class TestP1DefendedOnTimeout:
    def test_returns_true_when_flag_enabled_and_pylon_alive(self, env):
        _pylon_alive(env)
        assert env._p1_defended_on_timeout(_REWARDS_DEFENDER) is True

    def test_returns_false_when_flag_disabled(self, env):
        _pylon_alive(env)
        assert env._p1_defended_on_timeout(_REWARDS_NO_FLAG) is False

    def test_returns_false_when_pylon_destroyed(self, env):
        _pylon_destroyed(env)
        assert env._p1_defended_on_timeout(_REWARDS_DEFENDER) is False

    def test_returns_false_when_p1_has_no_defend_pool(self, env):
        """Attacker-only P1 should never receive defender win on timeout."""
        env.unit_pools_config = [
            {"agent": True, "high_level_command": "attack", "region": "A"},
            {"high_level_command": "defend", "region": "A"},
        ]
        _pylon_alive(env)
        assert env._p1_defended_on_timeout(_REWARDS_DEFENDER) is False


# ---------------------------------------------------------------------------
# _compute_p1_terminal_reward — timeout branch
# ---------------------------------------------------------------------------

class TestComputeP1TerminalRewardTimeout:
    def test_defender_pylon_intact_gets_win_reward(self, env):
        _pylon_alive(env)
        r = env._compute_p1_terminal_reward(
            p1_lost=False, p2_lost=False,
            p1_army_lost=False, p2_army_lost=False,
            all_units_gone=False, timeout=True,
            r_conf=_REWARDS_DEFENDER,
        )
        assert r == WIN

    def test_defender_pylon_destroyed_gets_loss_penalty(self, env):
        _pylon_destroyed(env)
        # p1_lost checked first → loss_penalty
        r = env._compute_p1_terminal_reward(
            p1_lost=True, p2_lost=False,
            p1_army_lost=False, p2_army_lost=False,
            all_units_gone=False, timeout=False,
            r_conf=_REWARDS_DEFENDER,
        )
        assert r == LOSS

    def test_defender_pylon_intact_flag_disabled_gets_draw(self, env):
        """Without the flag, timeout is still draw_penalty even if pylon alive."""
        _pylon_alive(env)
        r = env._compute_p1_terminal_reward(
            p1_lost=False, p2_lost=False,
            p1_army_lost=False, p2_army_lost=False,
            all_units_gone=False, timeout=True,
            r_conf=_REWARDS_NO_FLAG,
        )
        assert r == DRAW

    def test_attacker_with_alive_units_on_timeout_gets_loss(self, env):
        """P1 has live attack units at timeout → still a loss for the attacker."""
        env.unit_pools_config = [
            {"agent": True, "high_level_command": "attack", "region": "A"},
            {"high_level_command": "defend", "region": "A"},
        ]
        env.bot.unit_command_map = {10: "attack"}  # override fixture default
        _pylon_alive(env)
        r = env._compute_p1_terminal_reward(
            p1_lost=False, p2_lost=False,
            p1_army_lost=False, p2_army_lost=False,
            all_units_gone=False, timeout=True,
            r_conf=_REWARDS_DEFENDER,
        )
        assert r == LOSS

    def test_p2_army_killed_before_timeout_gives_win_not_timeout_path(self, env):
        """Killing all attackers triggers p2_army_lost path, not defender timeout."""
        _pylon_alive(env)
        r = env._compute_p1_terminal_reward(
            p1_lost=False, p2_lost=False,
            p1_army_lost=False, p2_army_lost=True,
            all_units_gone=False, timeout=False,
            r_conf=_REWARDS_DEFENDER,
        )
        assert r == WIN


# ---------------------------------------------------------------------------
# _all_trainee_pools_are_defenders — drives hard-reset opponent sync guard
# ---------------------------------------------------------------------------

class TestAllTraineePoolsAreDefenders:
    def test_pure_defender_returns_true(self, env):
        """Defender-only level: sync must be skipped so attacker model is kept."""
        assert env._all_trainee_pools_are_defenders() is True

    def test_attacker_only_returns_false(self, env):
        env.unit_pools_config = [
            {"agent": True, "high_level_command": "attack", "region": "A"},
            {"high_level_command": "defend", "region": "A"},
        ]
        assert env._all_trainee_pools_are_defenders() is False

    def test_mixed_attack_defend_returns_false(self, env):
        """Symmetric level has both attack and defend trainee pools."""
        env.unit_pools_config = [
            {"agent": True, "high_level_command": "attack", "region": "A"},
            {"agent": True, "high_level_command": "defend", "region": "B"},
            {"high_level_command": "defend", "region": "A"},
            {"high_level_command": "attack", "region": "B"},
        ]
        assert env._all_trainee_pools_are_defenders() is False

    def test_empty_pools_returns_false(self, env):
        env.unit_pools_config = []
        assert env._all_trainee_pools_are_defenders() is False
