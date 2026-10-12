"""
Regression tests for the phantom terminal step fix.

Bug: when the CTF episode ends during an OPPONENT's physics step
(requires_policy_action=False), the terminal reward from P1's perspective
was never written to the training buffer.  The fix defers the terminal state
until a phantom agent step that carries the reward into the buffer.

Coverage:
  - _compute_p1_terminal_reward returns correct values for all termination conditions
  - _finalize_squad_physics_step defers termination when acting_player_id != 1
  - requires_policy_action returns True while deferred state is pending
  - _async_step phantom step returns deferred reward and resets state
  - Normal agent-step termination is NOT deferred (reward goes directly to buffer)
  - reset() clears deferred state from the previous episode
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from sc2.ids.unit_typeid import UnitTypeId

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    env = CaptureTheFlagSingleEnv(visualize=False, max_steps=100)

    env.bot = MagicMock()
    env.bot.actions = []
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.client.actions = AsyncMock()
    env.bot.apply_action = AsyncMock()
    env.bot.record_pre_action_retreat_state = MagicMock()
    env.bot.do = MagicMock()
    env.bot._unit_last_order = {}
    env.bot.calculate_squad_step_reward = MagicMock(return_value=0.0)
    env.bot.get_observation = MagicMock(return_value=MagicMock())
    # Unconfigured MagicMock() coerces via __float__ to 1.0, not 0.0 — pin the
    # terminal army-value bonus explicitly (see src/agent/ctf_economic_rewards.py).
    env.bot.terminal_retained_value_bonus = MagicMock(return_value=0.0)

    # Pylon / army state — default: both pylons alive, both armies alive
    pylon_group = MagicMock()
    pylon_group.amount = 1
    env.bot.structures = MagicMock(return_value=pylon_group)
    env.bot.enemy_structures = MagicMock(return_value=pylon_group)

    env._manager = MagicMock()
    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value=np.zeros((23, 128, 128)))
    env._maybe_promote_defender_to_attack = MagicMock()
    env._flush_bot_actions = AsyncMock()
    env._reissue_non_squad_orders = AsyncMock()
    env._advance_squad_index = AsyncMock()
    env._agent_attack_units_alive = MagicMock(return_value=1)

    # Opponent squad setup (acting_player_id = 2)
    env.unit_pools_config = [{"agent": False, "high_level_command": "attack"}]
    env.squad_unit_tags = [(2, [200])]
    env.current_squad_index = 0
    env.current_unit_index = 0
    env.active_player_id = 2
    env.player_units = {1: [100], 2: [200]}

    return env


def _make_empty_pylon_group():
    g = MagicMock()
    g.amount = 0
    return g


def _make_alive_pylon_group():
    g = MagicMock()
    g.amount = 1
    return g


# ---------------------------------------------------------------------------
# _compute_p1_terminal_reward
# ---------------------------------------------------------------------------

def test_p1_terminal_reward_pylon_loss_alone_is_not_terminal(env):
    env._has_combat_units = MagicMock(return_value=True)
    env._enemy_objective_zone_cleared = MagicMock(return_value=False)
    r_conf = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    result = env._compute_p1_terminal_reward(
        p1_lost=True, p2_lost=False, p1_army_lost=False, p2_army_lost=False,
        all_units_gone=False, timeout=False, r_conf=r_conf
    )
    assert result == 0.0, "Flag pylon loss alone no longer ends the episode"


def test_p1_terminal_reward_zone_clear_is_win(env):
    env._enemy_objective_zone_cleared = MagicMock(return_value=True)
    r_conf = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    result = env._compute_p1_terminal_reward(
        p1_lost=False, p2_lost=False, p1_army_lost=False, p2_army_lost=True,
        all_units_gone=False, timeout=False, r_conf=r_conf
    )
    assert result == 15.0, "P1 wins when the objective zone is fully cleared"


def test_p1_terminal_reward_army_draw(env):
    r_conf = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    result = env._compute_p1_terminal_reward(
        p1_lost=False, p2_lost=False, p1_army_lost=True, p2_army_lost=True,
        all_units_gone=False, timeout=False, r_conf=r_conf
    )
    assert result == -2.0, "Mutual army wipe is a draw"


def test_p1_terminal_reward_timeout_with_attackers(env):
    env._agent_attack_units_alive.return_value = 2
    r_conf = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    result = env._compute_p1_terminal_reward(
        p1_lost=False, p2_lost=False, p1_army_lost=False, p2_army_lost=False,
        all_units_gone=False, timeout=True, r_conf=r_conf
    )
    assert result == -5.0, "Timeout with P1 attackers alive → loss penalty for P1"


def test_p1_terminal_reward_timeout_no_attackers(env):
    env._agent_attack_units_alive.return_value = 0
    r_conf = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    result = env._compute_p1_terminal_reward(
        p1_lost=False, p2_lost=False, p1_army_lost=False, p2_army_lost=False,
        all_units_gone=False, timeout=True, r_conf=r_conf
    )
    assert result == -2.0, "Timeout without P1 attackers alive → draw penalty"


def test_p1_terminal_reward_enemy_pylon_alone_is_not_win(env):
    """Destroying the enemy flag pylon alone must not award win_reward."""
    env._has_combat_units = MagicMock(return_value=True)
    env._enemy_objective_zone_cleared = MagicMock(return_value=False)
    r_conf = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    result = env._compute_p1_terminal_reward(
        p1_lost=False, p2_lost=True, p1_army_lost=False, p2_army_lost=False,
        all_units_gone=False, timeout=False, r_conf=r_conf
    )
    assert result == 0.0


def test_p1_terminal_reward_own_army_wipe_is_loss(env):
    r_conf = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    result = env._compute_p1_terminal_reward(
        p1_lost=False, p2_lost=False, p1_army_lost=True, p2_army_lost=False,
        all_units_gone=False, timeout=False, r_conf=r_conf
    )
    assert result == -5.0


# ---------------------------------------------------------------------------
# Phantom step: opponent terminates episode
# ---------------------------------------------------------------------------

def test_deferred_state_set_when_p1_wins_during_opponent_step(env):
    """When the enemy army is wiped during the opponent's physics step, the
    terminal state should be deferred and the step should return done=False."""
    env.training_conf["rewards"] = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    env.unit_pools_config = [
        {"agent": True, "high_level_command": "attack"},
        {"agent": False, "high_level_command": "defend"},
    ]

    alive_group = _make_alive_pylon_group()
    env.bot.structures.return_value = alive_group
    env.bot.enemy_structures.return_value = alive_group
    # P1 army alive, P2 army wiped + zone cleared → attacker win.
    env._has_combat_units = MagicMock(side_effect=lambda owner_id: owner_id == 1)
    env._enemy_objective_zone_cleared = MagicMock(return_value=True)

    async def run():
        obs, reward, done, truncated, info = await env._finalize_squad_physics_step(
            squad_tags=[200], acting_player_id=2
        )
        return obs, reward, done, truncated, info

    obs, reward, done, truncated, info = asyncio.run(run())

    # The step itself should NOT be terminal (deferred)
    assert done is False
    assert truncated is False
    assert reward == 0.0
    assert info.get("deferred_terminal") is True

    # Deferred state should be set
    assert env._deferred_done is True
    assert env._deferred_p1_terminal_reward == 15.0  # P1 wins = win_reward


def test_deferred_state_set_when_p1_loses_during_opponent_step(env):
    """When P1's army is wiped during the opponent's physics step."""
    env.training_conf["rewards"] = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    env.unit_pools_config = [
        {"agent": True, "high_level_command": "defend"},
        {"agent": False, "high_level_command": "attack"},
    ]

    alive_group = _make_alive_pylon_group()
    env.bot.structures.return_value = alive_group
    env.bot.enemy_structures.return_value = alive_group
    # P1 army wiped, P2 army alive; zone clear lets the actor's step terminate
    # so the deferred trainee path can apply P1's army-wipe loss.
    env._has_combat_units = MagicMock(side_effect=lambda owner_id: owner_id == 2)
    env._enemy_objective_zone_cleared = MagicMock(return_value=True)

    async def run():
        return await env._finalize_squad_physics_step(
            squad_tags=[200], acting_player_id=2
        )

    obs, reward, done, truncated, info = asyncio.run(run())

    assert done is False
    assert truncated is False
    assert reward == 0.0
    assert env._deferred_done is True
    assert env._deferred_p1_terminal_reward == -5.0   # P1 loses = loss_penalty


def test_requires_policy_action_true_when_deferred(env):
    """requires_policy_action must return True while deferred terminal is pending."""
    # Put env in a normal "opponent squad" state that would return False
    env.unit_pools_config = [{"agent": False, "high_level_command": "attack"}]
    env.squad_unit_tags = [(2, [200])]
    env.current_squad_index = 0

    # Without deferred state, it should be False (opponent squad)
    assert env.requires_policy_action() is False

    # Set deferred state
    env._deferred_done = True
    env._deferred_p1_terminal_reward = 15.0

    # Now it must return True
    assert env.requires_policy_action() is True


def test_phantom_step_returns_p1_reward_and_clears_state(env):
    """_async_step phantom step returns stored reward and clears deferred state."""
    dummy_obs = np.ones((23, 128, 128), dtype=np.float32)
    env._deferred_done = True
    env._deferred_truncated = False
    env._deferred_p1_terminal_reward = 15.0
    env._deferred_terminal_obs = dummy_obs

    async def run():
        return await env._async_step(action_p_active=np.zeros(2, dtype=np.int64))

    obs, reward, done, truncated, info = asyncio.run(run())

    # Phantom step should return terminal state with P1 reward
    assert done is True
    assert truncated is False
    assert reward == 15.0
    np.testing.assert_array_equal(obs, dummy_obs)
    assert info.get("phantom_terminal") is True

    # Deferred state must be cleared
    assert env._deferred_done is False
    assert env._deferred_truncated is False
    assert env._deferred_p1_terminal_reward == 0.0
    assert env._deferred_terminal_obs is None


def test_no_deferral_when_agent_step_terminates(env):
    """When the game ends during an AGENT's physics step, no deferral occurs."""
    env.training_conf["rewards"] = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}

    # Switch to agent squad; opponent owns the defend-pool flag.
    env.unit_pools_config = [
        {"agent": True, "high_level_command": "attack"},
        {"high_level_command": "defend"},
    ]
    env.squad_unit_tags = [(1, [100])]
    env.active_player_id = 1
    # P2 army wiped + zone cleared → P1 wins.
    env._has_combat_units = MagicMock(side_effect=lambda owner_id: owner_id == 1)
    env._enemy_objective_zone_cleared = MagicMock(return_value=True)

    alive_group = _make_alive_pylon_group()
    env.bot.structures.return_value = alive_group
    env.bot.enemy_structures.return_value = alive_group

    async def run():
        return await env._finalize_squad_physics_step(
            squad_tags=[100], acting_player_id=1
        )

    obs, reward, done, truncated, info = asyncio.run(run())

    # Agent step: done is returned directly, no deferral
    assert done is True
    assert reward == 15.0  # win_reward for P1
    assert env._deferred_done is False
    assert env._deferred_p1_terminal_reward == 0.0
    assert info.get("deferred_terminal") is None


def test_p1_terminal_reward_swaps_when_perspective_is_p2(env):
    """P2 trainee loses when its own army is wiped (perspective flip)."""
    r_conf = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    result = env._compute_p1_terminal_reward(
        p1_lost=False, p2_lost=False, p1_army_lost=False, p2_army_lost=True,
        all_units_gone=False, timeout=False, r_conf=r_conf,
        perspective_player_id=2,
    )
    assert result == -5.0, "P2 trainee loses when its army is wiped"


def test_no_deferral_when_p2_trainee_step_terminates(env):
    """Trainee on SC2 player 2: a trainee-acting terminal step is not deferred."""
    env.training_conf["rewards"] = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    env._alternate_trainee_race = True
    env.episode_counter = 2
    env.unit_pools_config = [
        {"race": "terran", "high_level_command": "attack"},
        {"race": "zerg", "high_level_command": "defend"},
    ]
    env.squad_unit_tags = [(2, [200])]
    env.active_player_id = 2
    # P2 army wiped → P2 trainee loss.
    env._has_combat_units = MagicMock(side_effect=lambda owner_id: owner_id == 1)

    alive_group = _make_alive_pylon_group()
    env.bot.structures.return_value = alive_group
    env.bot.enemy_structures.return_value = alive_group

    async def run():
        return await env._finalize_squad_physics_step(
            squad_tags=[200], acting_player_id=2
        )

    obs, reward, done, truncated, info = asyncio.run(run())

    assert done is True
    assert reward == -5.0  # P2 trainee army wiped
    assert env._deferred_done is False
    assert info.get("deferred_terminal") is None


def test_deferred_when_opponent_p1_terminates_p2_trainee_episode(env):
    """Opponent (P1) step ends the episode while trainee is P2 → defer trainee reward."""
    env.training_conf["rewards"] = {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}
    env._alternate_trainee_race = True
    env.episode_counter = 2
    env.unit_pools_config = [
        {"race": "terran", "high_level_command": "attack"},
        {"race": "zerg", "high_level_command": "defend"},
    ]
    env.squad_unit_tags = [(1, [100])]
    env.active_player_id = 1
    # P2 army wiped while P1 acts; zone clear terminates so P2 trainee loss defers.
    env._has_combat_units = MagicMock(side_effect=lambda owner_id: owner_id == 1)
    env._enemy_objective_zone_cleared = MagicMock(return_value=True)

    alive_group = _make_alive_pylon_group()
    env.bot.structures.return_value = alive_group
    env.bot.enemy_structures.return_value = alive_group

    async def run():
        return await env._finalize_squad_physics_step(
            squad_tags=[100], acting_player_id=1
        )

    obs, reward, done, truncated, info = asyncio.run(run())

    assert done is False
    assert truncated is False
    assert reward == 0.0
    assert info.get("deferred_terminal") is True
    assert env._deferred_done is True
    assert env._deferred_p1_terminal_reward == -5.0  # P2 trainee army wiped


def test_reset_clears_deferred_state(env):
    """reset() must clear any leftover deferred terminal state from prev episode."""
    env._deferred_done = True
    env._deferred_truncated = True
    env._deferred_p1_terminal_reward = 15.0
    env._deferred_terminal_obs = np.ones((23, 128, 128))

    # Minimal reset mocking
    env._manager.clients = []  # trigger hard-reset path, but we'll just check state clearing
    env.episode_counter = 49   # make hard reset fire immediately

    # Patch hard reset to avoid SC2 calls
    with patch.object(env, '_loop') as mock_loop:
        mock_loop.run_until_complete.return_value = (np.zeros((23, 128, 128)), {})
        try:
            env.reset()
        except Exception:
            pass  # ignore SC2 connection errors in unit test context

    assert env._deferred_done is False
    assert env._deferred_truncated is False
    assert env._deferred_p1_terminal_reward == 0.0
    assert env._deferred_terminal_obs is None
