"""Tests for fixed-size units_per_physics_step chunking across squads."""
import asyncio

import numpy as np
import pytest
from gymnasium import spaces
from unittest.mock import MagicMock, AsyncMock, patch

UNIT_POOLS_CONFIG = [
    {"high_level_command": "attack", "agent": True, "region": "A"},
    {"high_level_command": "defend", "agent": False, "region": "A"},
]


def _make_env(agent_tags=None, opponent_tags=None, chunk_size=8):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    with patch(
        "train.envs.capture_the_flag_single_env.SingleClientEnv.__init__",
        lambda self, *a, **kw: None,
    ):
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)

    agent_tags = agent_tags or [10, 11, 12]
    opponent_tags = opponent_tags or [20, 21]

    env.unit_pools_config = UNIT_POOLS_CONFIG
    env.squad_unit_tags = [(1, agent_tags), (2, opponent_tags)]
    env.current_squad_index = 0
    env.current_unit_index = 0
    env.active_player_id = 1
    env.player_units = {1: list(agent_tags), 2: list(opponent_tags)}
    env.unit_pool_index = {}
    env.steps_count = 0
    env.max_steps = 100
    env.training_conf = {
        "rewards": {},
        "env": {
            "step_mul_per_unit": 2,
            "units_per_physics_step": chunk_size,
        },
    }
    env.action_space = spaces.MultiDiscrete([4, 16])
    env._prepared_chunk = None

    env.bot = MagicMock()
    env.bot.actions = []
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.client.actions = AsyncMock()
    env.bot.apply_action = AsyncMock()
    env.bot.record_pre_action_retreat_state = MagicMock()
    env.bot.calculate_squad_step_reward = MagicMock(return_value=1.0)
    env.bot.do = MagicMock()
    env.bot._unit_last_order = {}
    env.bot.all_units = MagicMock()
    env.bot.all_units.find_by_tag = MagicMock(return_value=None)
    env.bot.structures = MagicMock()
    env.bot.enemy_structures = MagicMock()
    env.bot.structures.return_value.amount = 1
    env.bot.enemy_structures.return_value.amount = 1
    env.bot.get_observation = MagicMock(
        side_effect=lambda **kw: {"map": np.zeros((1, 4, 4), dtype=np.float32)}
    )
    env.bot.action_masks = MagicMock(return_value=np.ones(20, dtype=bool))

    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._get_joint_observation = AsyncMock(
        return_value={"map": np.zeros((1, 4, 4), dtype=np.float32)}
    )
    env._maybe_promote_defender_to_attack = MagicMock()
    env._has_combat_units = MagicMock(return_value=True)
    env._agent_attack_units_alive = MagicMock(return_value=1)
    env.opponent_model = MagicMock()
    env._opponent_lstm_state = None
    env._opponent_last_focus_tag = None

    return env


def test_peek_chunk_spans_squads():
    env = _make_env(chunk_size=4)
    chunk = env._peek_acting_chunk()
    assert len(chunk) == 4
    assert [e.tag for e in chunk] == [10, 11, 12, 20]
    assert [e.is_agent for e in chunk] == [True, True, True, False]


def test_peek_chunk_caps_at_available_units():
    env = _make_env(chunk_size=8)
    chunk = env._peek_acting_chunk()
    assert len(chunk) == 5
    assert [e.tag for e in chunk] == [10, 11, 12, 20, 21]


def test_prepare_acting_chunk_agent_obs_only():
    env = _make_env(chunk_size=4)
    batch = env.prepare_acting_chunk()
    assert batch is not None
    assert len(batch["obs"]) == 3
    assert batch["chunk_size"] == 4
    assert env._prepared_chunk is not None
    assert len(env._prepared_chunk) == 4


def test_requires_policy_mixed_vs_opponent_only_chunk():
    from train.envs.capture_the_flag_single_env import _ChunkEntry

    env = _make_env(chunk_size=4)
    assert env.requires_policy_action() is True

    env._prepared_chunk = [
        _ChunkEntry(1, 2, 20, False),
        _ChunkEntry(1, 2, 21, False),
    ]
    assert env.requires_policy_action() is False


def test_chunk_physics_step_size_uses_chunk_len():
    env = _make_env(chunk_size=4)
    env.set_squad_batch_actions(
        np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    )
    with patch(
        "train.squad_batch_inference.opponent_squad_batch_predict",
        return_value=(np.array([[0, 0], [1, 1]]), None, 21),
    ):
        obs, reward, done, truncated, info = asyncio.run(
            env._async_step(np.array([0, 0]))
        )

    assert info.get("squad_physics_step") is True
    assert env.bot.client.step.await_args.kwargs.get("step_size") == 8
    assert env.bot.apply_action.await_count == 5
    assert env.current_squad_index == 1
    assert env.current_unit_index == 1


def test_opponent_only_chunk_no_policy():
    env = _make_env(chunk_size=2)
    env.current_squad_index = 1
    env.current_unit_index = 0
    env.active_player_id = 2

    with patch(
        "train.squad_batch_inference.opponent_squad_batch_predict",
        return_value=(np.array([[0, 0], [1, 1]]), None, 21),
    ):
        assert env.requires_policy_action() is False
        obs, reward, done, truncated, info = asyncio.run(
            env._async_step(np.array([0, 0]))
        )

    assert info.get("squad_physics_step") is True
    assert env.bot.apply_action.await_count == 2
    env.bot.client.step.assert_called_once()
    assert env.bot.client.step.await_args.kwargs.get("step_size") == 4


def test_chunk_physics_refreshes_all_opponents():
    env = _make_env(chunk_size=4)
    env.set_squad_batch_actions(
        np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    )
    with patch.object(
        env, "_apply_fresh_opponent_actions", new=AsyncMock()
    ) as mock_all_opp:
        asyncio.run(env._async_step(np.array([0, 0])))
    mock_all_opp.assert_awaited_once()


def test_reissue_skips_opponent_units_in_chunk_mode():
    env = _make_env(chunk_size=4)
    env.bot._unit_last_order = {20: ("move", MagicMock())}
    unit = MagicMock(owner_id=2)
    env.bot.all_units.find_by_tag = MagicMock(return_value=unit)

    asyncio.run(env._reissue_non_squad_orders({10}, skip_player_ids={2}))
    env.bot.do.assert_not_called()

    asyncio.run(env._reissue_non_squad_orders({10}, skip_player_ids=set()))
    env.bot.do.assert_called_once()


def test_opponent_predict_falls_back_to_collector_model():
    import asyncio
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    with patch(
        "train.envs.capture_the_flag_single_env.SingleClientEnv.__init__",
        lambda self, *a, **kw: None,
    ):
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)

    env.active_player_id = 2
    env.action_space = spaces.MultiDiscrete([4, 16])
    env.opponent_model = None
    env.model = MagicMock()
    env.model.policy = MagicMock()
    env.model.policy.set_training_mode = MagicMock()
    env.bot = MagicMock()
    env.bot.get_observation = MagicMock(
        return_value={"map": np.zeros((1, 4, 4), dtype=np.float32)}
    )
    env.bot.action_masks = MagicMock(return_value=np.ones(20, dtype=bool))

    with patch(
        "train.squad_batch_inference.opponent_squad_batch_predict",
        return_value=(np.array([[2, 3], [4, 5]]), None, 21),
    ) as mock_predict:
        actions = asyncio.run(env._predict_opponent_squad_batch([20, 21]))

    assert mock_predict.call_args[0][0] is env.model
    assert len(actions) == 2


def test_ctf_defaults_squad_micro_physics():
    from train.envs.capture_the_flag_single_env import (
        CaptureTheFlagSingleEnv,
        DEFAULT_STEP_MUL_PER_UNIT,
        DEFAULT_UNITS_PER_PHYSICS_STEP,
    )

    with patch(
        "train.envs.capture_the_flag_single_env.SingleClientEnv.__init__",
        lambda self, *a, **kw: None,
    ):
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.training_conf = {"rewards": {}, "env": {}}

    assert env._units_per_physics_step() == DEFAULT_UNITS_PER_PHYSICS_STEP
    assert env._uses_chunk_physics() is False
    assert env._physics_step_size(acting_unit_count=3) == DEFAULT_STEP_MUL_PER_UNIT * 3


def test_squad_mode_when_units_per_physics_step_zero():
    env = _make_env(chunk_size=0)
    assert env._uses_chunk_physics() is False

    env.squad_unit_tags = [(1, [10, 11, 12])]
    env.unit_pools_config = [{"agent": True}]
    env.current_squad_index = 0
    env.current_unit_index = 0
    batch = env.prepare_squad_policy_batch()
    assert batch is not None
    assert len(batch["obs"]) == 3
