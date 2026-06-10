"""
Tests for pool-ordered squad cycling in CaptureTheFlagSingleEnv.

The desired cycle is: pool-0 (P1) → pool-1 (P2) → pool-2 (P1) → pool-3 (P2) → …
Each squad exhausts all its units before moving to the next squad.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UNIT_POOLS_CONFIG = [
    {"high_level_command": "attack", "agent": True,  "region": "A"},  # pool 0 → P1
    {"high_level_command": "defend", "agent": False, "region": "A"},  # pool 1 → P2
    {"high_level_command": "defend", "agent": True,  "region": "B"},  # pool 2 → P1
    {"high_level_command": "attack", "agent": False, "region": "B"},  # pool 3 → P2
]


def _make_env():
    """Return a CaptureTheFlagSingleEnv with squad state hand-initialised."""
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    # Patch away SC2 infrastructure
    with patch("train.envs.capture_the_flag_single_env.SingleClientEnv.__init__",
               lambda self, *a, **kw: None):
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)

    # Minimal state
    env.unit_pools_config = UNIT_POOLS_CONFIG
    env.squad_unit_tags = []
    env.current_squad_index = 0
    env.unit_pool_index = {}
    env.player_units = {1: [], 2: []}
    env.active_player_id = 1
    env.current_unit_index = 0
    return env


def _set_squads(env, squads):
    """squads: list of (player_id, [tag, …])"""
    env.squad_unit_tags = list(squads)
    env.active_player_id = squads[0][0]
    env.current_squad_index = 0
    env.current_unit_index = 0


# ---------------------------------------------------------------------------
# _rebuild_squad_unit_tags
# ---------------------------------------------------------------------------

def test_rebuild_squad_unit_tags_order():
    """squad_unit_tags preserves pool config order and filters dead units."""
    env = _make_env()
    # Pool 0 (P1): tags 10, 11  — tag 11 is alive
    # Pool 1 (P2): tags 20, 21  — both alive
    # Pool 2 (P1): tags 30       — alive
    # Pool 3 (P2): tags 40, 41   — tag 40 died
    env.unit_pool_index = {10: 0, 11: 0, 20: 1, 21: 1, 30: 2, 40: 3, 41: 3}
    env.player_units = {1: [11, 30], 2: [20, 21, 41]}  # 10 and 40 are dead

    env._rebuild_squad_unit_tags()

    assert len(env.squad_unit_tags) == 4

    p0, t0 = env.squad_unit_tags[0]  # pool 0 → P1
    assert p0 == 1
    assert set(t0) == {11}           # tag 10 filtered out

    p1, t1 = env.squad_unit_tags[1]  # pool 1 → P2
    assert p1 == 2
    assert set(t1) == {20, 21}

    p2, t2 = env.squad_unit_tags[2]  # pool 2 → P1
    assert p2 == 1
    assert set(t2) == {30}

    p3, t3 = env.squad_unit_tags[3]  # pool 3 → P2
    assert p3 == 2
    assert set(t3) == {41}           # tag 40 filtered out


# ---------------------------------------------------------------------------
# _current_squad
# ---------------------------------------------------------------------------

def test_current_squad_returns_correct_squad():
    env = _make_env()
    _set_squads(env, [(1, [10, 11]), (2, [20]), (1, [30]), (2, [40, 41])])

    env.current_squad_index = 0
    pid, tags = env._current_squad()
    assert pid == 1 and tags == [10, 11]

    env.current_squad_index = 1
    pid, tags = env._current_squad()
    assert pid == 2 and tags == [20]

    env.current_squad_index = 3
    pid, tags = env._current_squad()
    assert pid == 2 and tags == [40, 41]


def test_current_squad_fallback_when_no_squads():
    """Without squad_unit_tags, fall back to player_units."""
    env = _make_env()
    env.squad_unit_tags = []
    env.player_units = {1: [10, 11], 2: [20]}
    env.active_player_id = 1

    pid, tags = env._current_squad()
    assert pid == 1 and tags == [10, 11]


# ---------------------------------------------------------------------------
# Squad-advance logic (simulating _async_step cycling)
# ---------------------------------------------------------------------------

def _advance_squad(env):
    """Simulate squad advance after all units in the current squad have acted."""
    env.current_unit_index = 0
    if env.squad_unit_tags:
        n = len(env.squad_unit_tags)
        for _ in range(n):
            env.current_squad_index = (env.current_squad_index + 1) % n
            sq_player, sq_tags = env.squad_unit_tags[env.current_squad_index]
            if sq_tags:
                env.active_player_id = sq_player
                break
    else:
        env.active_player_id = 2 if env.active_player_id == 1 else 1


def _simulate_agent_partial_step(env):
    """One collector iteration on an agent squad (increment unit index; advance squad when exhausted)."""
    _, cur_tags = env._current_squad()
    env.current_unit_index += 1
    if env.current_unit_index >= len(cur_tags):
        _advance_squad(env)


def test_squad_cycling_full_rotation():
    """
    Verify the full cycle: P1-squad0 → P2-squad1 → P1-squad2 → P2-squad3 → P1-squad0.
    Each squad has 2 units.
    """
    env = _make_env()
    _set_squads(env, [(1, [10, 11]), (2, [20, 21]), (1, [30, 31]), (2, [40, 41])])

    expected_cycle = [
        # (squad_idx, player_id, unit_index)
        (0, 1, 0), (0, 1, 1),   # squad 0 exhausted after unit_index reaches 2
        (1, 2, 0), (1, 2, 1),
        (2, 1, 0), (2, 1, 1),
        (3, 2, 0), (3, 2, 1),
        (0, 1, 0),               # wraps back to squad 0
    ]

    for expected_sq, expected_pid, expected_ui in expected_cycle:
        assert env.current_squad_index == expected_sq, (
            f"squad {expected_sq}: got {env.current_squad_index}")
        assert env.active_player_id == expected_pid
        assert env.current_unit_index == expected_ui

        # Simulate one partial agent-squad step, then advance when squad is exhausted
        _simulate_agent_partial_step(env)


def test_squad_cycling_skips_empty_squads():
    """
    If a squad loses all its units, the cycler must skip it and go to the next
    non-empty squad.
    """
    env = _make_env()
    # Pool 1 (P2) is empty — all units died
    _set_squads(env, [(1, [10]), (2, []), (1, [30]), (2, [40])])

    # Exhaust squad 0 (single unit)
    _simulate_agent_partial_step(env)

    # Should skip the empty squad 1 and land on squad 2
    assert env.current_squad_index == 2
    assert env.active_player_id == 1


def test_squad_index_preserved_within_squad():
    """unit_index advances within a squad without changing the squad index."""
    env = _make_env()
    _set_squads(env, [(1, [10, 11, 12]), (2, [20])])

    env.current_unit_index = 0
    _simulate_agent_partial_step(env)
    assert env.current_squad_index == 0
    assert env.active_player_id == 1

    _simulate_agent_partial_step(env)
    assert env.current_squad_index == 0

    _simulate_agent_partial_step(env)
    assert env.current_squad_index == 1
    assert env.active_player_id == 2


def test_requires_policy_action_agent_vs_opponent_squad():
    env = _make_env()
    _set_squads(env, [(1, [10, 11]), (2, [20]), (1, [30]), (2, [40])])
    assert env.requires_policy_action() is True

    env.current_squad_index = 1
    env.active_player_id = 2
    env.current_unit_index = 0
    assert env.requires_policy_action() is False

    env.current_squad_index = 2
    env.active_player_id = 1
    assert env.requires_policy_action() is True


def test_requires_policy_false_for_empty_agent_squad():
    env = _make_env()
    _set_squads(env, [(1, []), (2, [20])])
    assert env.requires_policy_action() is False
