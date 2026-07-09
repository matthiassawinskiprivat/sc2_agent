"""
Regression test: multi-squad observation perspective handoff.

Bug: in _finalize_squad_physics_step the squad index was advanced AFTER
_get_joint_observation was called.  When P2's squad finished its step, the
observation returned to the collector was from P2's (spatially-flipped)
perspective.  The collector then fed that flipped obs to the policy for P1's
next squad, causing P1's defender to navigate in the completely wrong direction.

Fix: advance the squad index BEFORE calling _get_joint_observation so the
returned obs always reflects the NEXT acting squad.

Coverage:
  - After P2's squad step (non-terminal), obs is regenerated from the
    _next_ squad's perspective, not the just-acted squad's perspective.
  - After P1's squad step (non-terminal), the advance still happens before
    obs generation (advance → P2 squad; obs from P2 squad).
  - Terminal steps do NOT advance and return the current squad's obs.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock


from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(marker: float) -> np.ndarray:
    """Tiny obs array whose first element identifies which 'squad view' it is."""
    a = np.zeros((3, 4, 4), dtype=np.float32)
    a[0, 0, 0] = marker
    return a


@pytest.fixture
def env():
    env = CaptureTheFlagSingleEnv(visualize=False, max_steps=200)

    env.bot = MagicMock()
    env.bot.actions = []
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.apply_action = AsyncMock()
    env.bot.record_pre_action_retreat_state = MagicMock()
    env.bot.do = MagicMock()
    env.bot._unit_last_order = {}
    env.bot.calculate_squad_step_reward = MagicMock(return_value=0.0)
    env.bot.get_observation = MagicMock(return_value=MagicMock())

    # Pylons / army: all alive by default (no terminal conditions)
    pylon_group = MagicMock()
    pylon_group.amount = 1
    env.bot.structures = MagicMock(return_value=pylon_group)
    env.bot.enemy_structures = MagicMock(return_value=pylon_group)

    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._maybe_promote_defender_to_attack = MagicMock()
    env._flush_bot_actions = AsyncMock()
    env._reissue_non_squad_orders = AsyncMock()
    env._agent_attack_units_alive = MagicMock(return_value=1)
    env._has_combat_units = MagicMock(return_value=True)

    env.training_conf = {}
    env.steps_count = 0
    env.max_steps = 200
    env._deferred_done = False
    env._deferred_truncated = False
    env._deferred_p1_terminal_reward = 0.0
    env._deferred_terminal_obs = None

    env.active_player_id = 2  # will be set per test
    env.player_units = {1: [10], 2: [20]}

    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_obs_after_opponent_step_reflects_next_p1_squad():
    """
    After P2's squad acts (non-terminal), the returned observation must come
    from the NEXT squad's perspective (P1 squad), not P2's flipped view.

    Mechanism: _advance_squad_index is called BEFORE _get_joint_observation
    inside _finalize_squad_physics_step.  We verify the order by recording
    the call sequence.
    """
    env = CaptureTheFlagSingleEnv(visualize=False, max_steps=200)

    env.bot = MagicMock()
    env.bot.actions = []
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.apply_action = AsyncMock()
    env.bot.record_pre_action_retreat_state = MagicMock()
    env.bot.do = MagicMock()
    env.bot._unit_last_order = {}
    env.bot.calculate_squad_step_reward = MagicMock(return_value=0.0)
    env.bot.get_observation = MagicMock(return_value=MagicMock())

    pylon_group = MagicMock()
    pylon_group.amount = 1
    env.bot.structures = MagicMock(return_value=pylon_group)
    env.bot.enemy_structures = MagicMock(return_value=pylon_group)

    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._maybe_promote_defender_to_attack = MagicMock()
    env._flush_bot_actions = AsyncMock()
    env._reissue_non_squad_orders = AsyncMock()
    env._agent_attack_units_alive = MagicMock(return_value=1)
    env._has_combat_units = MagicMock(return_value=True)
    env.training_conf = {}
    env.steps_count = 0
    env.max_steps = 200
    env._deferred_done = False
    env._deferred_truncated = False
    env._deferred_p1_terminal_reward = 0.0
    env._deferred_terminal_obs = None
    env.player_units = {1: [10], 2: [20]}

    # Track the call order of advance vs get_observation
    call_log = []

    async def fake_advance():
        call_log.append("advance")

    async def fake_get_obs():
        call_log.append("get_obs")
        return _make_obs(99.0)

    env._advance_squad_index = fake_advance
    env._get_joint_observation = fake_get_obs

    # Run _finalize_squad_physics_step for P2 (acting_player_id=2), non-terminal
    obs, reward, done, truncated, info = asyncio.run(
        env._finalize_squad_physics_step(squad_tags=[20], acting_player_id=2)
    )

    assert not done, "Should not be terminal"
    assert not truncated, "Should not be truncated"

    # THE KEY ASSERTION: advance must come before get_obs
    assert call_log == ["advance", "get_obs"], (
        f"Expected advance before get_obs, got: {call_log}. "
        "The observation perspective bug is back: obs was generated before advancing "
        "the squad, so the collector sees the wrong (flipped) squad's view."
    )


def test_obs_after_p1_step_still_advances_before_get_obs():
    """Same ordering guarantee holds when P1's squad acts (non-terminal)."""
    env = CaptureTheFlagSingleEnv(visualize=False, max_steps=200)

    env.bot = MagicMock()
    env.bot.actions = []
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.apply_action = AsyncMock()
    env.bot.record_pre_action_retreat_state = MagicMock()
    env.bot.do = MagicMock()
    env.bot._unit_last_order = {}
    env.bot.calculate_squad_step_reward = MagicMock(return_value=0.0)
    env.bot.get_observation = MagicMock(return_value=MagicMock())

    pylon_group = MagicMock()
    pylon_group.amount = 1
    env.bot.structures = MagicMock(return_value=pylon_group)
    env.bot.enemy_structures = MagicMock(return_value=pylon_group)

    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._maybe_promote_defender_to_attack = MagicMock()
    env._flush_bot_actions = AsyncMock()
    env._reissue_non_squad_orders = AsyncMock()
    env._agent_attack_units_alive = MagicMock(return_value=1)
    env._has_combat_units = MagicMock(return_value=True)
    env.training_conf = {}
    env.steps_count = 0
    env.max_steps = 200
    env._deferred_done = False
    env._deferred_truncated = False
    env._deferred_p1_terminal_reward = 0.0
    env._deferred_terminal_obs = None
    env.player_units = {1: [10], 2: [20]}

    call_log = []

    async def fake_advance():
        call_log.append("advance")

    async def fake_get_obs():
        call_log.append("get_obs")
        return _make_obs(42.0)

    env._advance_squad_index = fake_advance
    env._get_joint_observation = fake_get_obs

    obs, reward, done, truncated, info = asyncio.run(
        env._finalize_squad_physics_step(squad_tags=[10], acting_player_id=1)
    )

    assert not done
    assert call_log == ["advance", "get_obs"], (
        f"advance must precede get_obs for P1 step too, got: {call_log}"
    )


def test_terminal_step_does_not_advance_before_get_obs():
    """
    When the episode ends (done=True), squad should NOT advance.
    obs is generated from the current (terminating) squad's view.
    """
    env = CaptureTheFlagSingleEnv(visualize=False, max_steps=200)

    env.bot = MagicMock()
    env.bot.actions = []
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.apply_action = AsyncMock()
    env.bot.record_pre_action_retreat_state = MagicMock()
    env.bot.do = MagicMock()
    env.bot._unit_last_order = {}
    env.bot.calculate_squad_step_reward = MagicMock(return_value=0.0)
    env.bot.get_observation = MagicMock(return_value=MagicMock())

    # Trigger done: P2's pylon gone → P1 wins
    dead_pylon_group = MagicMock()
    dead_pylon_group.amount = 0  # P2 pylon dead
    alive_pylon_group = MagicMock()
    alive_pylon_group.amount = 1
    env.bot.structures = MagicMock(return_value=alive_pylon_group)
    env.bot.enemy_structures = MagicMock(return_value=dead_pylon_group)

    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._maybe_promote_defender_to_attack = MagicMock()
    env._flush_bot_actions = AsyncMock()
    env._reissue_non_squad_orders = AsyncMock()
    env._agent_attack_units_alive = MagicMock(return_value=1)
    env._has_combat_units = MagicMock(return_value=True)
    env.training_conf = {"rewards": {"win_reward": 15.0, "loss_penalty": -5.0, "draw_penalty": -2.0}}
    env.steps_count = 0
    env.max_steps = 200
    env._deferred_done = False
    env._deferred_truncated = False
    env._deferred_p1_terminal_reward = 0.0
    env._deferred_terminal_obs = None
    env.player_units = {1: [10], 2: [20]}

    call_log = []

    async def fake_advance():
        call_log.append("advance")

    async def fake_get_obs():
        call_log.append("get_obs")
        return _make_obs(7.0)

    env._advance_squad_index = fake_advance
    env._get_joint_observation = fake_get_obs

    obs, reward, done, truncated, info = asyncio.run(
        env._finalize_squad_physics_step(squad_tags=[10], acting_player_id=1)
    )

    assert done, "P2 pylon gone → should be done"
    assert "advance" not in call_log, (
        "Squad must NOT advance when episode ends (terminal obs stays on current squad)"
    )
    assert "get_obs" in call_log, "Obs must still be generated for terminal return"
