"""Regression tests for CTF squad pointer env stepping."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from gymnasium import spaces

from src.agent.squad_constants import (
    SQUAD_FLAT_ACTION_DIM,
    SQUAD_UNIT_FEAT_DIM,
    squad_action_space_nvec,
)


@pytest.fixture
def squad_env():
    with patch("train.envs.capture_the_flag_single_env.ConfigManager") as mock_cm, patch(
        "train.envs.capture_the_flag_single_env.SingleClientEnv.__init__", return_value=None
    ):
        mock_cm.return_value.get_agent_architecture.return_value = "squad"
        mock_cm.return_value.get_cumulative_active_layers.return_value = ["layer_a"]
        mock_cm.return_value.get_cumulative_active_tactical_layers.return_value = ["layer_a"]
        mock_cm.return_value.get_cumulative_action_heads.return_value = ["move_global", "attack_global"]
        mock_cm.return_value.get_cumulative_categorical_heads.return_value = []

        from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
        env.training_conf = {
            "agent_architecture": "squad",
            "active_layers": ["layer_a"],
            "active_tactical_layers": ["layer_a"],
            "action_heads": ["move_global", "attack_global"],
            "categorical_heads": [],
            "env": {"squad_selection": {"score_threshold": 0.5, "min_selected_units": 1}},
            "rewards": {},
        }
        env.level_name = "test_ctf"
        env.config_manager = mock_cm.return_value
        env.agent_architecture = "squad"
        env.unit_pools_config = [
            {"high_level_command": "attack", "region": "A", "race": "terran"},
            {"high_level_command": "defend", "region": "A", "race": "zerg"},
            {"high_level_command": "defend", "region": "B", "race": "terran"},
            {"high_level_command": "attack", "region": "B", "race": "zerg"},
        ]
        env._alternate_trainee_race = False
        env.squad_unit_tags = [
            (1, [101, 102]),
            (2, [201]),
            (1, [301]),
            (2, [401]),
        ]
        env.current_trainee_pool_index = 0
        env.current_squad_index = 0
        env.current_unit_index = 0
        env.steps_count = 0
        env.max_steps = 200
        env.player_units = {1: [101, 102, 301], 2: [201, 401]}
        env._deferred_done = False
        env._deferred_truncated = False
        env._preview_computer_opponent = MagicMock(return_value=True)
        env._physics_step_size = MagicMock(return_value=1)
        env._maybe_promote_defender_to_attack = MagicMock()
        env._has_combat_units = MagicMock(return_value=True)
        env._agent_attack_units_alive = MagicMock(return_value=1)
        env._refresh_bot_state = AsyncMock()
        env._refresh_unit_lists = AsyncMock()
        env._flush_bot_actions = AsyncMock()
        env._reissue_non_squad_orders = AsyncMock()
        env._advance_trainee_pool_index = AsyncMock()
        env._get_squad_observation_for_pool = MagicMock(
            return_value={
                "map": np.zeros((1, 128, 128), dtype=np.float32),
                "command": np.zeros(10, dtype=np.float32),
                "anchor_tactical": np.zeros((1, 32, 32), dtype=np.float32),
                "unit_scalars": np.zeros((24, SQUAD_UNIT_FEAT_DIM), dtype=np.float32),
                "unit_mask": np.array([1.0, 1.0] + [0.0] * 22, dtype=np.float32),
            }
        )
        env._apply_squad_action_for_pool = AsyncMock()
        env._get_joint_observation = AsyncMock(
            return_value={
                "map": np.zeros((1, 128, 128), dtype=np.float32),
                "command": np.zeros(10, dtype=np.float32),
                "anchor_tactical": np.zeros((1, 32, 32), dtype=np.float32),
                "unit_scalars": np.zeros((24, SQUAD_UNIT_FEAT_DIM), dtype=np.float32),
                "unit_mask": np.array([1.0, 1.0] + [0.0] * 22, dtype=np.float32),
            }
        )
        env._pool_player_id = lambda pool: 1 if pool["race"] == "terran" else 2
        env._is_pool_trainee = lambda pool: pool["race"] == "terran"
        env._paired_opponent_pool_index = MagicMock(return_value=1)
        env._pool_alive_tags = lambda idx: env.squad_unit_tags[idx][1]
        env.bot = MagicMock()
        env.bot.client = MagicMock()
        env.bot.client.step = AsyncMock()
        env.bot.structures = MagicMock(return_value=MagicMock(amount=1))
        env.bot.enemy_structures = MagicMock(return_value=MagicMock(amount=1))
        env.bot.calculate_squad_step_reward = MagicMock(return_value=0.5)
        # Unconfigured MagicMock() coerces via __float__ to 1.0, not 0.0 — pin the
        # terminal army-value bonus explicitly so unrelated tests are not silently
        # off by one full value unit (see src/agent/ctf_economic_rewards.py).
        env.bot.terminal_retained_value_bonus = MagicMock(return_value=0.0)
        env.bot.squad_action_masks = MagicMock(
            return_value=np.ones(SQUAD_FLAT_ACTION_DIM, dtype=bool)
        )
        yield env


def test_requires_policy_action_true_when_trainee_pool_alive(squad_env):
    assert squad_env.requires_policy_action() is True


def test_requires_policy_action_false_when_trainee_pool_empty(squad_env):
    squad_env.squad_unit_tags[0] = (1, [])
    assert squad_env.requires_policy_action() is False


def test_get_obs_focus_unit_tag_returns_minus_one(squad_env):
    assert squad_env.get_obs_focus_unit_tag() == -1


def test_get_current_trainee_pool_index_matches_current_pool(squad_env):
    squad_env.current_trainee_pool_index = 2
    assert squad_env.get_current_trainee_pool_index() == 2


def test_attach_action_masks_includes_trainee_pool_index(squad_env):
    squad_env.current_trainee_pool_index = 2
    squad_env._cached_squad_action_masks = MagicMock(
        return_value=np.ones(SQUAD_FLAT_ACTION_DIM, dtype=bool)
    )
    info = squad_env._attach_action_masks({"squad_pointer": True})
    assert info["trainee_pool_index"] == 2
    assert info["requires_policy_action"] is True


def test_squad_pointer_step_applies_trainee_and_advances(squad_env):
    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    action[0] = 1
    obs, reward, done, truncated, info = asyncio.run(
        squad_env._async_step_squad_pointer(action)
    )
    squad_env._apply_squad_action_for_pool.assert_awaited_once()
    squad_env.bot.client.step.assert_awaited()
    squad_env._advance_trainee_pool_index.assert_awaited()
    assert squad_env._advance_trainee_pool_index.await_args.kwargs.get(
        "refresh_units"
    ) is False
    assert reward == 0.5
    assert info.get("squad_pointer") is True
    assert "action_masks" in info
    assert info["action_masks"].dtype == bool
    assert info.get("requires_policy_action") is True
    assert info.get("trainee_pool_index") == 0


def test_squad_pointer_step_applies_paired_opponent_action(squad_env):
    """A due step (step 0) still gives the policy opponent a fresh squad order."""
    squad_env._preview_computer_opponent = MagicMock(return_value=False)
    squad_env._predict_opponent_squad_pointer = AsyncMock(
        return_value=np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    )
    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    action[0] = 1
    asyncio.run(squad_env._async_step_squad_pointer(action))
    assert squad_env._apply_squad_action_for_pool.await_count == 2
    squad_env._predict_opponent_squad_pointer.assert_awaited_once_with(1)
    applied_pools = [
        call.args[0] for call in squad_env._apply_squad_action_for_pool.await_args_list
    ]
    assert applied_pools == [0, 1]


def test_paired_opponent_pool_index_integration(squad_env):
    from train.envs.ctf.squad_pool_pairing import paired_opponent_pool_index

    squad_env._is_pool_trainee = squad_env._is_pool_trainee
    idx = paired_opponent_pool_index(
        0, squad_env.unit_pools_config, is_pool_trainee=squad_env._is_pool_trainee
    )
    assert idx == 1


def test_squad_action_masks_reuses_cache_without_rebuilding_obs(squad_env):
    obs = {
        "unit_mask": np.array([1.0, 1.0] + [0.0] * 22, dtype=np.float32),
    }
    mask = np.ones(26, dtype=bool)
    squad_env.steps_count = 3
    squad_env.current_trainee_pool_index = 0
    squad_env.bot.squad_action_masks = MagicMock(return_value=mask)
    squad_env._cache_squad_action_mask(obs, 0)

    with patch.object(
        squad_env, "_get_squad_observation_for_pool", side_effect=AssertionError("should not rebuild obs")
    ), patch.object(
        squad_env, "_squad_unit_mask_for_pool", side_effect=AssertionError("should not rebuild unit mask")
    ):
        assert np.array_equal(squad_env.action_masks(), mask)
        assert np.array_equal(squad_env.action_masks(), mask)
    assert squad_env.bot.squad_action_masks.call_count == 1


def test_squad_action_masks_cache_miss_uses_pool_observation(squad_env):
    """Cache miss must rebuild from the pool observation.

    Command legality (``heal_unit`` needs a ready healer and a wounded patient,
    ``attack_unit`` a visible hostile) is observation-derived, so the bare unit
    mask is not enough to produce a mask the learner can reproduce.
    """
    flat_mask = np.ones(SQUAD_FLAT_ACTION_DIM, dtype=bool)
    squad_env.steps_count = 2
    squad_env.current_trainee_pool_index = 0
    squad_env._squad_action_mask_cache = None
    squad_env.bot.squad_action_masks = MagicMock(return_value=flat_mask)

    out = squad_env.action_masks()

    squad_env._get_squad_observation_for_pool.assert_called_with(0)
    obs = squad_env._get_squad_observation_for_pool.return_value
    args = squad_env.bot.squad_action_masks.call_args.args
    assert args[0] is obs["unit_mask"]
    assert args[1] is obs
    assert np.array_equal(out, flat_mask)


def test_hard_anchor_tag_remembered_and_resolved(squad_env):
    """Option A: previous-step hard selection drives the next obs crop tag."""
    squad_env._squad_anchor_tag_by_pool = {}
    squad_env._remember_squad_anchor_tag(0, 102)
    assert squad_env._resolve_squad_anchor_tag(0, [101, 102]) == 102
    # Dead preferred tag falls back to first sorted alive.
    assert squad_env._resolve_squad_anchor_tag(0, [101]) == 101
    # No memory → provisional first sorted alive.
    squad_env._squad_anchor_tag_by_pool = {}
    assert squad_env._resolve_squad_anchor_tag(0, [102, 101]) == 101


def test_squad_pointer_p2_trainee_does_not_defer_terminal(squad_env):
    """Pointer steps always record a trainee action; P2 trainee terminals are not deferred."""
    squad_env.training_conf["rewards"] = {
        "win_reward": 15.0,
        "loss_penalty": -5.0,
        "draw_penalty": -2.0,
    }
    squad_env._alternate_trainee_race = True
    squad_env.episode_counter = 2
    squad_env._is_pool_trainee = lambda pool: pool["race"] == "zerg"
    squad_env._pool_player_id = lambda pool: 1 if pool["race"] == "terran" else 2
    squad_env.current_trainee_pool_index = 1
    squad_env.bot.structures.return_value = MagicMock(amount=1)
    squad_env.bot.enemy_structures.return_value = MagicMock(amount=0)

    obs, reward, done, truncated, info = asyncio.run(
        squad_env._finalize_squad_pointer_physics_step(
            trainee_tags=[201],
            opponent_tags=[101],
            trainee_player_id=2,
        )
    )
    assert done is True
    assert truncated is False
    assert reward == 0.5 - 5.0
    assert info.get("deferred_terminal") is None
    assert squad_env._deferred_done is False


def test_empty_trainee_army_forces_physics_terminal(squad_env):
    """Skip-loop must not freeze: no trainee tags → one physics step that can end."""
    squad_env.squad_unit_tags = [
        (1, []),
        (2, [201]),
        (1, []),
        (2, [401]),
    ]
    squad_env.current_trainee_pool_index = 0
    squad_env._has_combat_units = MagicMock(side_effect=lambda owner_id: owner_id != 1)
    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    obs, reward, done, truncated, info = asyncio.run(
        squad_env._async_step_squad_pointer(action)
    )
    squad_env.bot.client.step.assert_awaited()
    squad_env._apply_squad_action_for_pool.assert_not_awaited()
    assert info.get("squad_physics_step") is True
    assert done is True
    assert squad_env.steps_count == 1


def test_begin_squad_obs_cache_seeds_last_obs(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    last = {"unit_mask": np.array([1.0], dtype=np.float32)}
    squad_env._last_obs = last
    squad_env.current_trainee_pool_index = 0
    squad_env._squad_obs_step_cache = None
    squad_env._squad_obs_builder = MagicMock()
    CaptureTheFlagSingleEnv._begin_squad_obs_cache(squad_env)
    assert squad_env._squad_obs_step_cache[0] is last
    squad_env._squad_obs_builder.clear_global_cache.assert_not_called()


def test_get_squad_observation_hits_step_cache(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    cached = {"unit_mask": np.ones(24, dtype=np.float32)}
    squad_env._squad_obs_step_cache = {0: cached}
    squad_env._ensure_squad_obs_builder = MagicMock(
        side_effect=AssertionError("must not rebuild")
    )
    out = CaptureTheFlagSingleEnv._get_squad_observation_for_pool(squad_env, 0)
    assert out is cached


def test_squad_pointer_step_size_is_fixed_not_army_size(squad_env):
    """A 12-vs-12 decision advances 8 frames, not one frame per living unit."""
    from train.envs.capture_the_flag_single_env import DEFAULT_SQUAD_POINTER_STEP_SIZE

    assert squad_env._squad_pointer_step_size() == DEFAULT_SQUAD_POINTER_STEP_SIZE
    asyncio.run(
        squad_env._finalize_squad_pointer_physics_step(
            trainee_tags=list(range(12)),
            opponent_tags=list(range(100, 112)),
            trainee_player_id=1,
        )
    )
    assert (
        squad_env.bot.client.step.await_args.kwargs.get("step_size")
        == DEFAULT_SQUAD_POINTER_STEP_SIZE
    )
    squad_env._physics_step_size.assert_not_called()


def test_squad_pointer_step_size_yaml_override(squad_env):
    squad_env.training_conf["env"]["squad_pointer_step_size"] = 4
    assert squad_env._squad_pointer_step_size() == 4
    squad_env.training_conf["env"]["squad_pointer_step_size"] = 0
    assert squad_env._squad_pointer_step_size() == 1


def test_finalize_invalidates_obs_cache_after_physics(squad_env):
    squad_env._invalidate_pre_physics_obs_caches = MagicMock()
    asyncio.run(
        squad_env._finalize_squad_pointer_physics_step(
            trainee_tags=[101],
            opponent_tags=[201],
            trainee_player_id=1,
        )
    )
    squad_env.bot.client.step.assert_awaited()
    squad_env._invalidate_pre_physics_obs_caches.assert_called()


def test_opponent_decision_holds_between_samples(squad_env):
    """Non-due paired turns reuse the last order: no obs build, no forward, no new apply."""
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env._preview_computer_opponent = MagicMock(return_value=False)
    # Opponent pool 1 already took its first (due) turn.
    squad_env._opponent_decision_turn_counts = {1: 1}
    squad_env._predict_opponent_squad_pointer = AsyncMock(
        side_effect=AssertionError("hold step must not run the opponent network")
    )
    squad_env._apply_squad_action_for_pool = AsyncMock(return_value=[101])
    assert CaptureTheFlagSingleEnv.peek_opponent_squad_pointer(squad_env) is None
    squad_env._get_squad_observation_for_pool.assert_not_called()

    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    action[0] = 1
    asyncio.run(squad_env._async_step_squad_pointer(action))
    squad_env._predict_opponent_squad_pointer.assert_not_awaited()
    assert squad_env._apply_squad_action_for_pool.await_count == 1
    assert squad_env._apply_squad_action_for_pool.await_args.args[0] == 0
    reissued = squad_env._reissue_non_squad_orders.await_args.args[0]
    assert 201 not in reissued
    assert squad_env._opponent_decision_turn_counts[1] == 2


def test_opponent_decision_due_every_fourth_paired_turn(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env._preview_computer_opponent = MagicMock(return_value=False)
    squad_env._squad_obs_step_cache = {}
    squad_env._opponent_lstm_states = {}
    squad_env.bot.squad_action_masks = MagicMock(return_value=np.ones(8, dtype=bool))
    for turn, due in ((0, True), (1, False), (3, False), (4, True)):
        squad_env._opponent_decision_turn_counts = {1: turn}
        squad_env._get_squad_observation_for_pool.reset_mock()
        payload = CaptureTheFlagSingleEnv.peek_opponent_squad_pointer(squad_env)
        if due:
            assert payload is not None
            squad_env._get_squad_observation_for_pool.assert_called_once()
        else:
            assert payload is None
            squad_env._get_squad_observation_for_pool.assert_not_called()


def test_opponent_decision_interval_not_phase_locked_across_arenas(squad_env):
    """Dual trainee pools alternate each step; both opponents must get due turns.

    Regression: ``steps_count % 4 == 0`` was always true on the attack-trainee
    step and always false on the defend-trainee step, so the attack-role
    opponent never received a fresh order.
    """
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = squad_env
    env._preview_computer_opponent = MagicMock(return_value=False)
    env._opponent_decision_turn_counts = {}
    # Mirror layout: trainee 0↔opp 1, trainee 2↔opp 3
    assert env._opponent_decision_due(1) is True
    assert env._opponent_decision_due(3) is True
    CaptureTheFlagSingleEnv._bump_opponent_decision_turn(env, 1)
    CaptureTheFlagSingleEnv._bump_opponent_decision_turn(env, 3)
    assert env._opponent_decision_due(1) is False
    assert env._opponent_decision_due(3) is False
    for _ in range(3):
        CaptureTheFlagSingleEnv._bump_opponent_decision_turn(env, 1)
        CaptureTheFlagSingleEnv._bump_opponent_decision_turn(env, 3)
    assert env._opponent_decision_due(1) is True
    assert env._opponent_decision_due(3) is True


def test_alternating_arenas_both_opponents_get_fresh_orders(squad_env):
    """Defend opp (pool 1) and attack opp (pool 3) both sample on their due turns.

    With the old global ``steps_count % 4`` clock, only the even trainee steps
    (pool 0 ↔ defend opp 1) were ever due — the attack-role opponent stayed idle
    forever even after the null-target reissue fix.
    """
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
    from train.envs.ctf.squad_pool_pairing import paired_opponent_pool_index

    env = squad_env
    env.training_conf["opponent_decision_interval"] = 4
    env._preview_computer_opponent = MagicMock(return_value=False)
    env._opponent_decision_turn_counts = {}
    env._pending_opponent_squad_action = None
    env._pending_opponent_squad_pool_idx = None
    env._paired_opponent_pool_index = lambda idx: paired_opponent_pool_index(
        idx, env.unit_pools_config, is_pool_trainee=env._is_pool_trainee
    )
    env._trainee_pool_indices = lambda: [0, 2]
    env._refresh_unit_lists = AsyncMock()

    # After each physics step, advance like the real finalize path.
    async def _finalize_and_advance(*_a, **_k):
        await CaptureTheFlagSingleEnv._advance_trainee_pool_index(
            env, refresh_units=False
        )
        return {"map": np.zeros(1)}, 0.0, False, False, {"squad_physics_step": True}

    env._finalize_squad_pointer_physics_step = _finalize_and_advance
    env._predict_opponent_squad_pointer = AsyncMock(
        side_effect=lambda pool_idx: np.full(
            len(squad_action_space_nvec()), int(pool_idx), dtype=np.int64
        )
    )
    env._apply_squad_action_for_pool = AsyncMock(
        side_effect=lambda pool_idx, _action: (
            [101] if int(pool_idx) in (0, 2) else [201 if int(pool_idx) == 1 else 401]
        )
    )

    applied_opp_pools: list[int] = []
    peeked_opp_pools: list[int] = []

    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    action[0] = 1
    # Each arena pairs every other env step. Interval 4 → dues at turns 0 and 4
    # need five pairings per opponent. Starting on trainee 0, that is 10 env steps
    # so pool 3 also reaches its second due turn.
    for _ in range(10):
        env._squad_obs_step_cache = {}
        peek = CaptureTheFlagSingleEnv.peek_opponent_squad_pointer(env)
        if peek is not None:
            peeked_opp_pools.append(int(peek["pool_idx"]))
            env.set_pending_opponent_squad_action(
                np.full(len(squad_action_space_nvec()), peek["pool_idx"], dtype=np.int64),
                pool_idx=peek["pool_idx"],
            )
        before = env._apply_squad_action_for_pool.await_count
        asyncio.run(env._async_step_squad_pointer(action))
        for call in env._apply_squad_action_for_pool.await_args_list[before:]:
            pool = int(call.args[0])
            if pool in (1, 3):
                applied_opp_pools.append(pool)

    assert peeked_opp_pools.count(1) == 2, peeked_opp_pools
    assert peeked_opp_pools.count(3) == 2, peeked_opp_pools
    assert applied_opp_pools.count(1) == 2, applied_opp_pools
    assert applied_opp_pools.count(3) == 2, applied_opp_pools
    # Old steps_count % 4 clock would have applied pool 3 zero times.


def test_stale_pending_opponent_action_ignored_for_other_pool(squad_env):
    """Pending peeked for pool 1 must not be applied after handoff to pool 3."""
    from train.envs.ctf.squad_pool_pairing import paired_opponent_pool_index

    squad_env._preview_computer_opponent = MagicMock(return_value=False)
    squad_env.current_trainee_pool_index = 2
    squad_env._opponent_decision_turn_counts = {3: 0}
    squad_env._paired_opponent_pool_index = lambda idx: paired_opponent_pool_index(
        idx, squad_env.unit_pools_config, is_pool_trainee=squad_env._is_pool_trainee
    )
    squad_env._predict_opponent_squad_pointer = AsyncMock(
        return_value=np.arange(len(squad_action_space_nvec()), dtype=np.int64)
    )
    stale = np.full(len(squad_action_space_nvec()), 7, dtype=np.int64)
    squad_env._pending_opponent_squad_action = stale
    squad_env._pending_opponent_squad_pool_idx = 1
    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    action[0] = 1
    asyncio.run(squad_env._async_step_squad_pointer(action))
    applied = [
        call.args[1] for call in squad_env._apply_squad_action_for_pool.await_args_list
    ]
    assert len(applied) == 2
    assert not np.array_equal(applied[1], stale)
    squad_env._predict_opponent_squad_pointer.assert_awaited_once_with(3)


def test_opponent_decision_interval_one_samples_every_step(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env.training_conf["opponent_decision_interval"] = 1
    squad_env._preview_computer_opponent = MagicMock(return_value=False)
    squad_env._opponent_decision_turn_counts = {1: 2}
    squad_env._squad_obs_step_cache = {}
    squad_env._opponent_lstm_states = {}
    squad_env.bot.squad_action_masks = MagicMock(return_value=np.ones(8, dtype=bool))
    payload = CaptureTheFlagSingleEnv.peek_opponent_squad_pointer(squad_env)
    assert payload is not None
    squad_env._get_squad_observation_for_pool.assert_called_once()


def test_peek_opponent_none_for_computer_preview(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env._preview_computer_opponent = MagicMock(return_value=True)
    assert CaptureTheFlagSingleEnv.peek_opponent_squad_pointer(squad_env) is None


def test_peek_opponent_reuses_step_cache(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env._preview_computer_opponent = MagicMock(return_value=False)
    obs = {"unit_mask": np.ones(24, dtype=np.float32)}
    mask = np.ones(8, dtype=bool)
    squad_env.bot.squad_action_masks = MagicMock(return_value=mask)
    squad_env._get_squad_observation_for_pool = MagicMock(return_value=obs)
    squad_env._squad_obs_step_cache = {}
    squad_env._opponent_lstm_states = {}
    payload = CaptureTheFlagSingleEnv.peek_opponent_squad_pointer(squad_env)
    assert payload["pool_idx"] == 1
    assert payload["obs"] is obs
    assert payload["mask"] is mask
    assert payload["episode_start"] is True
    squad_env._get_squad_observation_for_pool.assert_called_once_with(1)


def test_pending_opponent_action_skips_in_worker_predict(squad_env):
    squad_env._preview_computer_opponent = MagicMock(return_value=False)
    squad_env._predict_opponent_squad_pointer = AsyncMock(
        side_effect=AssertionError("collector already batched opponent")
    )
    pending = np.arange(len(squad_action_space_nvec()), dtype=np.int64)
    squad_env._pending_opponent_squad_action = pending
    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    action[0] = 1
    asyncio.run(squad_env._async_step_squad_pointer(action))
    applied_actions = [
        call.args[1] for call in squad_env._apply_squad_action_for_pool.await_args_list
    ]
    assert np.array_equal(applied_actions[1], pending)


def test_needs_ability_refresh_false_without_spell_heads(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env._cached_squad_action_heads = None
    assert CaptureTheFlagSingleEnv._needs_ability_refresh(squad_env) is False
    squad_env.training_conf["action_heads"] = ["move_global", "snipe"]
    squad_env._cached_squad_action_heads = None
    assert CaptureTheFlagSingleEnv._needs_ability_refresh(squad_env) is True


def test_query_spell_abilities_off_by_default(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    assert CaptureTheFlagSingleEnv._query_spell_abilities_enabled(squad_env) is False
    squad_env.training_conf["env"]["query_spell_abilities"] = True
    assert CaptureTheFlagSingleEnv._query_spell_abilities_enabled(squad_env) is True


def test_refresh_bot_state_skips_ability_rpc_by_default(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env.training_conf["action_heads"] = ["snipe", "heal_unit", "aoe_local"]
    squad_env._cached_squad_action_heads = None
    squad_env._collector_profile_flag = False
    obs = MagicMock()
    obs.observation = MagicMock()
    squad_env.bot.client.observation = AsyncMock(return_value=obs)
    squad_env.bot._prepare_step = MagicMock()
    squad_env.bot._proto_game_info = None
    squad_env.bot._available_abilities_by_tag = {1: frozenset()}
    squad_env._note_visible_heal_tumors = MagicMock()

    with patch(
        "src.utils.sc2_observation_compat.game_state_from_response_observation",
        return_value=MagicMock(),
    ), patch(
        "src.utils.unit_feature_helpers.refresh_healer_ability_cache",
        new_callable=AsyncMock,
    ) as mock_query:
        asyncio.run(CaptureTheFlagSingleEnv._refresh_bot_state(squad_env))
    mock_query.assert_not_called()
    assert squad_env.bot._available_abilities_by_tag == {}


def test_ability_query_units_uses_pool_tags(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    u101 = MagicMock()
    u201 = MagicMock()
    squad_env.bot.all_units.find_by_tag = MagicMock(
        side_effect=lambda tag: {101: u101, 102: u101, 201: u201, 301: u101, 401: u201}.get(tag)
    )
    units = CaptureTheFlagSingleEnv._ability_query_units(squad_env)
    tags = {id(u) for u in units}
    assert id(u101) in tags
    assert id(u201) in tags


def test_refresh_bot_state_profiles_gamestate(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env._collector_profile_flag = True
    obs = MagicMock()
    obs.observation = MagicMock()
    squad_env.bot.client.observation = AsyncMock(return_value=obs)
    squad_env.bot._prepare_step = MagicMock()
    squad_env.bot._proto_game_info = None
    squad_env.bot._available_abilities_by_tag = {}
    squad_env._note_visible_heal_tumors = MagicMock()

    with patch(
        "src.utils.sc2_observation_compat.game_state_from_response_observation",
        return_value=MagicMock(),
    ):
        asyncio.run(CaptureTheFlagSingleEnv._refresh_bot_state(squad_env))
    means = CaptureTheFlagSingleEnv.pop_collector_profile_means(squad_env)
    assert "collector/profile_observation_ms" in means
    assert "collector/profile_gamestate_ms" in means
    assert means["collector/profile_gamestate_ms"] >= 0.0


def test_squad_pointer_step_profiles_reward_bookkeeping_and_unit_lists(squad_env):
    """Unwrapped step work shows up under SC2_COLLECTOR_PROFILE, and stays quiet when off."""
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    action[0] = 1
    squad_env._collector_profile_flag = False
    asyncio.run(squad_env._async_step_squad_pointer(action))
    assert CaptureTheFlagSingleEnv.pop_collector_profile_means(squad_env) == {}

    squad_env._collector_profile_flag = True
    asyncio.run(squad_env._async_step_squad_pointer(action))
    means = CaptureTheFlagSingleEnv.pop_collector_profile_means(squad_env)
    expected = (
        "collector/profile_reward_ms",
        "collector/profile_econ_ms",
        "collector/profile_heal_rec_ms",
        "collector/profile_cmd_rec_ms",
        "collector/profile_cd_retreat_ms",
        "collector/profile_unit_lists_ms",
    )
    for key in expected:
        assert len(key) <= 40
        assert key in means
        assert means[key] >= 0.0


def test_step_profiles_worker_and_shm_export(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
    from train.envs.single_client_env import SingleClientEnv

    squad_env._collector_profile_flag = True
    obs = {"map": np.zeros((1,), dtype=np.float32)}
    info = {}
    with patch.object(
        SingleClientEnv, "step", return_value=(obs, 0.0, False, False, info)
    ):
        returned = CaptureTheFlagSingleEnv.step(squad_env, np.zeros(1))
    assert returned[0] is obs
    means = CaptureTheFlagSingleEnv.pop_collector_profile_means(squad_env)
    for key in (
        "collector/profile_worker_step_ms",
        "collector/profile_shm_export_ms",
    ):
        assert len(key) <= 40
        assert key in means
        assert means[key] >= 0.0


def test_flush_and_reissue_profile_order_path(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env._collector_profile_flag = True
    squad_env.bot.actions = [MagicMock()]
    squad_env.bot.client.actions = AsyncMock()
    squad_env.player_units = {1: [], 2: []}
    squad_env.bot._unit_last_order = {}

    async def _run():
        await CaptureTheFlagSingleEnv._reissue_non_squad_orders(squad_env, set())
        await CaptureTheFlagSingleEnv._flush_bot_actions(squad_env)

    asyncio.run(_run())
    means = CaptureTheFlagSingleEnv.pop_collector_profile_means(squad_env)
    assert "collector/profile_reissue_ms" in means
    assert "collector/profile_action_flush_ms" in means
    assert squad_env.bot.actions == []
    squad_env.bot.client.actions.assert_awaited()


def test_pop_collector_profile_means(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    squad_env._collector_profile_flag = True
    CaptureTheFlagSingleEnv._profile_add(squad_env, "obs_build", 0.02)
    CaptureTheFlagSingleEnv._profile_add(squad_env, "obs_build", 0.04)
    means = CaptureTheFlagSingleEnv.pop_collector_profile_means(squad_env)
    assert means["collector/profile_obs_build_ms"] == pytest.approx(30.0)
    assert CaptureTheFlagSingleEnv.pop_collector_profile_means(squad_env) == {}


def test_pop_collector_profile_cache_hit_rates(squad_env):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    CaptureTheFlagSingleEnv._profile_add_count(squad_env, "cache_tg_hit", 9)
    CaptureTheFlagSingleEnv._profile_add_count(squad_env, "cache_tg_miss", 1)
    CaptureTheFlagSingleEnv._profile_add_count(squad_env, "cache_tt_hit", 4)
    CaptureTheFlagSingleEnv._profile_add_count(squad_env, "cache_tt_miss", 1)
    CaptureTheFlagSingleEnv._profile_add_count(squad_env, "cache_wpn_hit", 20)
    CaptureTheFlagSingleEnv._profile_add_count(squad_env, "cache_wpn_miss", 0)
    CaptureTheFlagSingleEnv._profile_add_count(squad_env, "cache_gmap_hit", 1)
    CaptureTheFlagSingleEnv._profile_add_count(squad_env, "cache_gmap_swap", 1)
    CaptureTheFlagSingleEnv._profile_add_count(squad_env, "cache_gmap_full", 1)
    means = CaptureTheFlagSingleEnv.pop_collector_profile_means(squad_env)
    assert means["collector/profile_cache_tg_hit_n"] == 9.0
    assert means["collector/profile_cache_tg_rate"] == pytest.approx(0.9)
    assert means["collector/profile_cache_tt_rate"] == pytest.approx(0.8)
    assert means["collector/profile_cache_wpn_rate"] == pytest.approx(1.0)
    assert means["collector/profile_cache_gmap_rate"] == pytest.approx(2.0 / 3.0)
    assert CaptureTheFlagSingleEnv.pop_collector_profile_means(squad_env) == {}


def test_controller_pointer_appends_stolen_spellcaster_after_native_tags():
    from sc2.ids.buff_id import BuffId
    from sc2.ids.unit_typeid import UnitTypeId

    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.unit_pools_config = [{"race": "zerg"}]
    env._pool_alive_tags = lambda idx: [30, 10]
    env._pool_player_id = lambda pool: 1
    templar = MagicMock()
    templar.tag = 20
    templar.type_id = UnitTypeId.HIGHTEMPLAR
    templar.owner_id = 1
    templar.is_structure = False
    templar.buffs = {BuffId.NEURALPARASITE}
    zealot = MagicMock()
    zealot.tag = 21
    zealot.type_id = UnitTypeId.ZEALOT
    zealot.owner_id = 1
    zealot.is_structure = False
    zealot.buffs = {BuffId.NEURALPARASITE}
    env._all_units_by_tag = lambda: {20: templar, 21: zealot}
    assert CaptureTheFlagSingleEnv._squad_pointer_tags(env, 0) == [10, 30, 20]


def test_stolen_caster_snipe_is_issued_only_for_the_controller():
    from sc2.ids.buff_id import BuffId
    from sc2.ids.unit_typeid import UnitTypeId

    from src.agent.ctf_bot import CaptureTheFlagBot

    stolen = MagicMock()
    stolen.tag = 7
    stolen.type_id = UnitTypeId.HIGHTEMPLAR
    stolen.owner_id = 1
    stolen.buffs = {BuffId.NEURALPARASITE}
    stolen.position = MagicMock()
    stolen.is_idle = False
    enemy = MagicMock()
    enemy.tag = 8
    enemy.owner_id = 2
    enemy.energy_max = 200.0
    enemy.is_structure = False
    enemy.is_biological = False
    enemy.position = MagicMock()
    bot = MagicMock()
    bot.all_units.find_by_tag.side_effect = lambda tag: {7: stolen, 8: enemy}[int(tag)]
    bot._unit_last_order = {}
    bot._snipe_lock = {}
    bot._snipe_ability_ready_after_loop = {}
    bot._command_attempt_tags = {"snipe": set()}
    bot.state = MagicMock(game_loop=0)
    bot.frame_count = 0
    bot._mark_command_attempt = (
        lambda u, kind: CaptureTheFlagBot._mark_command_attempt(bot, u, kind)
    )
    bot._set_last_order_flags = (
        lambda tag, *, moved=False, attacked=False: CaptureTheFlagBot._set_last_order_flags(
            bot, tag, moved=moved, attacked=attacked
        )
    )
    bot._refresh_snipe_lock = lambda u: CaptureTheFlagBot._refresh_snipe_lock(bot, u)
    bot._refresh_aoe_lock = lambda u: False
    bot._issue_snipe_with_lock = (
        lambda u, t: CaptureTheFlagBot._issue_snipe_with_lock(bot, u, t)
    )
    bot.last_action_was_move_by_unit = {}
    bot.last_action_was_attack_by_unit = {}

    CaptureTheFlagBot._apply_unit_target_command(
        bot, [7], head_name="snipe", target_unit_tag=8, perspective_player_id=1
    )
    bot.do.assert_called()
    bot.do.reset_mock()
    CaptureTheFlagBot._apply_unit_target_command(
        bot, [7], head_name="snipe", target_unit_tag=8, perspective_player_id=2
    )
    bot.do.assert_not_called()
