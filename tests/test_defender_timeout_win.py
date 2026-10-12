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

    def test_p2_trainee_defender_timeout_win(self, env):
        """Trainee on SC2 player 2 still wins timeout when its pylon survives."""
        env._alternate_trainee_race = True
        env.episode_counter = 2
        env.unit_pools_config = [
            {"race": "terran", "high_level_command": "attack", "region": "B"},
            {"race": "zerg", "high_level_command": "defend", "region": "B"},
        ]
        env.player_units = {1: [10], 2: [20]}
        env.bot.unit_command_map = {20: "defend"}
        env.bot.structures(UnitTypeId.PYLON).amount = 1
        env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
        assert env._p1_defended_on_timeout(_REWARDS_DEFENDER) is True


def test_agent_attack_units_alive_counts_p2_trainee(env):
    env._alternate_trainee_race = True
    env.episode_counter = 2
    env.unit_pools_config = [
        {"race": "terran", "high_level_command": "defend", "region": "B"},
        {"race": "zerg", "high_level_command": "attack", "region": "B"},
    ]
    env.player_units = {1: [10], 2: [20, 21]}
    env.bot.unit_command_map = {20: "attack", 21: "attack"}
    assert env._agent_attack_units_alive() == 2


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

    def test_defender_pylon_destroyed_alone_does_not_end(self, env):
        """Flag pylons are anchors only; pylon loss alone is not a terminal loss."""
        _pylon_destroyed(env)
        env._has_combat_units = MagicMock(return_value=True)
        env._enemy_objective_zone_cleared = MagicMock(return_value=False)
        r = env._compute_p1_terminal_reward(
            p1_lost=True, p2_lost=False,
            p1_army_lost=False, p2_army_lost=False,
            all_units_gone=False, timeout=False,
            r_conf=_REWARDS_DEFENDER,
        )
        assert r == 0.0

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

    def test_objective_zone_cleared_gives_win(self, env):
        """Clearing all enemy units and buildings from the objective zone is a win."""
        _pylon_alive(env)
        env._enemy_objective_zone_cleared = MagicMock(return_value=True)
        r = env._compute_p1_terminal_reward(
            p1_lost=False, p2_lost=False,
            p1_army_lost=False, p2_army_lost=True,
            all_units_gone=False, timeout=False,
            r_conf=_REWARDS_DEFENDER,
        )
        assert r == WIN

    def test_opponent_army_killed_with_pylon_standing_is_not_win(self, env):
        """Army wipe alone does not win while the objective building still stands."""
        _pylon_alive(env)
        env._enemy_objective_zone_cleared = MagicMock(return_value=False)
        r = env._compute_p1_terminal_reward(
            p1_lost=False, p2_lost=False,
            p1_army_lost=False, p2_army_lost=True,
            all_units_gone=False, timeout=False,
            r_conf=_REWARDS_DEFENDER,
        )
        assert r == 0.0

    def test_own_army_killed_still_loss(self, env):
        """Losing the trainee's own combat units remains an immediate loss."""
        _pylon_alive(env)
        r = env._compute_p1_terminal_reward(
            p1_lost=False, p2_lost=False,
            p1_army_lost=True, p2_army_lost=False,
            all_units_gone=False, timeout=False,
            r_conf=_REWARDS_DEFENDER,
        )
        assert r == LOSS

    def test_enemy_pylon_destroyed_alone_does_not_win(self, env):
        """Flag pylon is the zone anchor only — destroying it is not a win trophy."""
        _pylon_alive(env)
        # Keep enemy army "alive" so zone-clear / wipe branches do not fire.
        env._has_combat_units = MagicMock(return_value=True)
        env._enemy_objective_zone_cleared = MagicMock(return_value=False)
        r = env._compute_p1_terminal_reward(
            p1_lost=False, p2_lost=True,
            p1_army_lost=False, p2_army_lost=False,
            all_units_gone=False, timeout=False,
            r_conf=_REWARDS_DEFENDER,
        )
        assert r == 0.0


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
