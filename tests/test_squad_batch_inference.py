"""Tests for batched squad policy inference (agent collector + opponent env)."""
import numpy as np
import pytest
import torch
from gymnasium import spaces
from unittest.mock import MagicMock, AsyncMock, patch

from train.squad_batch_inference import (
    SquadBatchRolloutHelper,
    agent_squad_batch_forward,
    expand_rnn_state_for_batch,
    normalize_single_env_rnn_states,
    opponent_squad_batch_predict,
    stack_env_dict_observations,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates


def _dict_obs(tag: int) -> dict:
    return {
        "map": np.full((2, 4, 4), tag, dtype=np.float32),
        "command": np.array([tag, 0.0, 0.0], dtype=np.float32),
    }


def test_stack_env_dict_observations():
    stacked = stack_env_dict_observations([_dict_obs(1), _dict_obs(2)])
    assert stacked["map"].shape == (2, 2, 4, 4)
    assert stacked["command"].shape == (2, 3)
    assert stacked["map"][1, 0, 0, 0] == 2


def test_prepare_squad_policy_batch_agent_squad_start():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    with patch(
        "train.envs.capture_the_flag_single_env.SingleClientEnv.__init__",
        lambda self, *a, **kw: None,
    ):
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)

    env.unit_pools_config = [{"agent": True, "high_level_command": "attack"}]
    env.squad_unit_tags = [(1, [10, 11, 12])]
    env.current_squad_index = 0
    env.current_unit_index = 0
    env.active_player_id = 1
    env.bot = MagicMock()
    env.bot.get_observation = MagicMock(side_effect=lambda **kw: _dict_obs(kw.get("perspective_player_id", 1)))
    env.bot.action_masks = MagicMock(return_value=np.ones(10, dtype=bool))

    batch = env.prepare_squad_policy_batch()
    assert batch is not None
    assert len(batch["obs"]) == 3
    assert len(batch["focus_tags"]) == 3


def test_prepare_squad_policy_batch_skips_partial_and_single_unit():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    with patch(
        "train.envs.capture_the_flag_single_env.SingleClientEnv.__init__",
        lambda self, *a, **kw: None,
    ):
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)

    env.unit_pools_config = [{"agent": True}]
    env.squad_unit_tags = [(1, [10])]
    env.current_squad_index = 0
    env.current_unit_index = 0
    env.bot = MagicMock()
    assert env.prepare_squad_policy_batch() is None

    env.squad_unit_tags = [(1, [10, 11])]
    env.current_unit_index = 1
    assert env.prepare_squad_policy_batch() is None


def test_agent_squad_batch_forward_non_recurrent():
    policy = MagicMock()
    policy.obs_to_tensor = MagicMock(side_effect=lambda batch: (batch, True))
    actions = torch.tensor([[1, 2], [3, 4], [5, 6]])
    values = torch.tensor([[0.1], [0.2], [0.3]])
    log_probs = torch.tensor([-0.1, -0.2, -0.3])
    policy.forward = MagicMock(return_value=(actions, values, log_probs))

    obs_list = [_dict_obs(i) for i in range(3)]
    masks = [np.ones(10, dtype=bool) for _ in range(3)]
    act_t, val_t, logp_t, lstm, starts = agent_squad_batch_forward(
        policy,
        obs_list,
        masks,
        first_episode_start=1.0,
        lstm_states=None,
        is_recurrent=False,
        device=torch.device("cpu"),
    )
    assert policy.forward.call_count == 1
    assert act_t.shape[0] == 3
    assert starts == [1.0, 1.0, 1.0]
    assert lstm is None


def test_opponent_squad_batch_predict_feedforward():
    model = MagicMock()
    model.device = torch.device("cpu")
    policy = MagicMock(spec=["obs_to_tensor", "set_training_mode", "forward"])
    model.policy = policy

    obs_list = [_dict_obs(1), _dict_obs(2)]
    masks = [np.ones(8, dtype=bool) for _ in range(2)]

    stacked = stack_env_dict_observations(obs_list)
    policy.obs_to_tensor = MagicMock(return_value=(stacked, True))
    policy.set_training_mode = MagicMock()
    policy.forward = MagicMock(
        return_value=(
            torch.tensor([[1, 0], [0, 1]]),
            torch.zeros(2, 1),
            torch.zeros(2),
        )
    )

    actions, state, last_tag = opponent_squad_batch_predict(
        model, obs_list, masks, lstm_state=None, deterministic=True
    )
    assert policy.forward.call_count == 1
    assert actions.shape == (2, 2)
    assert state is None
    assert last_tag is None


def test_opponent_squad_env_uses_batch_predict():
    import asyncio
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    with patch(
        "train.envs.capture_the_flag_single_env.SingleClientEnv.__init__",
        lambda self, *a, **kw: None,
    ):
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)

    env.active_player_id = 2
    env.action_space = spaces.MultiDiscrete([4, 16])
    env.bot = MagicMock()
    env.bot.get_observation = MagicMock(side_effect=lambda **kw: _dict_obs(1))
    env.bot.action_masks = MagicMock(return_value=np.ones(20, dtype=bool))
    env.bot.apply_action = AsyncMock()
    env.bot.record_pre_action_retreat_state = MagicMock()
    env._opponent_lstm_state = None
    env.opponent_model = MagicMock()
    env._opponent_last_focus_tag = None

    with patch(
        "train.squad_batch_inference.opponent_squad_batch_predict",
        return_value=(np.array([[1, 2], [3, 4]]), "new_state", 101),
    ) as mock_batch:
        actions = asyncio.run(env._predict_opponent_squad_batch([100, 101]))

    assert mock_batch.call_count == 1
    assert len(actions) == 2
    assert env._opponent_lstm_state == "new_state"


def test_opponent_recurrent_predict_uses_batched_forward():
    """Recurrent opponent inference uses one batched forward (same as agent squads)."""
    model = MagicMock()
    model.device = torch.device("cpu")
    policy = MagicMock()
    policy.lstm_actor = torch.nn.GRU(4, 4)
    policy.rnn_type = "gru"
    model.policy = policy

    obs_list = [_dict_obs(1), _dict_obs(2)]
    masks = [np.ones(8, dtype=bool) for _ in range(2)]

    with patch(
        "train.squad_batch_inference.agent_squad_batch_forward",
        return_value=(
            torch.tensor([[1, 2], [3, 4]]),
            None,
            None,
            None,
            [1.0, 1.0],
        ),
    ) as mock_batch:
        actions, state, last_tag = opponent_squad_batch_predict(
            model,
            obs_list,
            masks,
            lstm_state=None,
            deterministic=True,
            focus_tags=[10, 11],
            last_focus_tag=None,
        )

    assert mock_batch.call_count == 1
    assert actions.shape == (2, 2)
    assert last_tag == 11
    from sb3_contrib.common.recurrent.type_aliases import RNNStates

    policy = MagicMock()
    policy.lstm_actor = MagicMock(num_layers=1, hidden_size=4)
    policy.rnn_type = "gru"
    bad = RNNStates((torch.zeros(1, 0, 4),), (torch.zeros(1, 0, 4),))
    fixed = normalize_single_env_rnn_states(bad, policy, torch.device("cpu"))
    expanded = expand_rnn_state_for_batch(fixed, 3, torch.device("cpu"))
    assert expanded.pi[0].shape == (1, 3, 4)

    # model.predict / opponent_squad_batch_predict store (pi_tuple, vf_tuple), not RNNStates
    predict_style = (
        (np.zeros((1, 1, 4), dtype=np.float32),),
        (np.zeros((1, 1, 4), dtype=np.float32),),
    )
    fixed_predict = normalize_single_env_rnn_states(
        predict_style, policy, torch.device("cpu")
    )
    assert fixed_predict.pi[0].shape == (1, 1, 4)


def test_resolve_rollout_full_squad_produces_one_env_step_batch():
    vec = MagicMock()
    vec.action_space = spaces.MultiDiscrete([4, 16])
    vec.env_method = MagicMock(
        side_effect=lambda name, *a, **kw: {
            "prepare_policy_batch": [
                {
                    "obs": [_dict_obs(0), _dict_obs(1), _dict_obs(2)],
                    "masks": [np.ones(20, dtype=bool)] * 3,
                    "focus_tags": [1, 2, 3],
                }
            ],
        }.get(name, [None])
    )

    model = MagicMock()
    model.device = torch.device("cpu")
    policy = MagicMock()
    model.policy = policy
    stacked = stack_env_dict_observations([_dict_obs(0), _dict_obs(1), _dict_obs(2)])
    policy.obs_to_tensor = MagicMock(return_value=(stacked, True))
    policy.forward = MagicMock(
        return_value=(
            torch.tensor([[0, 1], [1, 2], [2, 3]]),
            torch.tensor([[0.5], [0.4], [0.3]]),
            torch.tensor([-0.2, -0.3, -0.4]),
        )
    )
    model._last_lstm_states = None

    helper = SquadBatchRolloutHelper(1)
    obs = {k: v[np.newaxis, ...] for k, v in _dict_obs(0).items()}

    resolution = helper.resolve_rollout(
        vec_env=vec,
        num_envs=1,
        obs=obs,
        requires_policy=np.array([True]),
        model=model,
        is_recurrent=False,
        last_dones=np.array([False]),
        prev_focus_tags=np.array([-1], dtype=np.int64),
        lstm_states=None,
        device=torch.device("cpu"),
    )

    assert resolution.step_actions.shape == (1, 2)
    assert 0 in resolution.squad_batch_actions
    assert resolution.squad_batch_actions[0].shape == (3, 2)
    assert len(resolution.buffer_rows) == 3
    assert all(row.apply_reward for row in resolution.buffer_rows)


def test_write_rollout_rows_applies_squad_reward_to_every_unit_row():
    from train.squad_batch_inference import (
        CachedSquadStep,
        RolloutBufferRow,
        RolloutResolution,
        write_rollout_rows_to_buffer,
    )

    buffer = MagicMock()
    buffer.buffer_size = 8
    buffer.pos = 0
    recorded_rewards = []

    def _capture_add(_obs, _action, reward, *_args, **_kwargs):
        recorded_rewards.append(reward.copy())
        buffer.pos += 1

    buffer.add = MagicMock(side_effect=_capture_add)

    rows = [
        RolloutBufferRow(
            env_idx=0,
            cached=CachedSquadStep(
                obs=_dict_obs(i),
                action=np.array([i, i], dtype=np.int64),
                value=torch.tensor([0.1]),
                log_prob=torch.tensor(-0.1),
                mask=np.ones(8, dtype=bool),
                lstm_start=1.0 if i else 0.0,
                lstm_states_before=None,
            ),
            apply_reward=True,
        )
        for i in range(3)
    ]
    resolution = RolloutResolution(
        step_actions=np.zeros((1, 2), dtype=np.int64),
        squad_batch_actions={0: np.zeros((3, 2), dtype=np.int64)},
        buffer_rows=rows,
        vec_rows=[],
        updated_lstm=None,
    )
    pre_obs = {k: v[np.newaxis, ...] for k, v in _dict_obs(0).items()}

    written = write_rollout_rows_to_buffer(
        buffer=buffer,
        resolution=resolution,
        pre_step_obs=pre_obs,
        rewards=np.array([2.5]),
        num_envs=1,
        is_recurrent=False,
        fallback_lstm=None,
        last_dones=np.array([False]),
    )

    assert written == 3
    assert len(recorded_rewards) == 3
    assert all(r[0] == pytest.approx(2.5) for r in recorded_rewards)


def test_write_rollout_rows_respects_max_rows():
    from train.squad_batch_inference import (
        CachedSquadStep,
        RolloutBufferRow,
        RolloutResolution,
        write_rollout_rows_to_buffer,
    )

    buffer = MagicMock()
    buffer.buffer_size = 2
    buffer.pos = 1

    def _advance_pos(*args, **kwargs):
        buffer.pos += 1

    buffer.add = MagicMock(side_effect=_advance_pos)

    rows = [
        RolloutBufferRow(
            env_idx=0,
            cached=CachedSquadStep(
                obs=_dict_obs(i),
                action=np.array([i, i], dtype=np.int64),
                value=torch.tensor([0.1]),
                log_prob=torch.tensor(-0.1),
                mask=np.ones(8, dtype=bool),
                lstm_start=1.0,
                lstm_states_before=None,
            ),
            apply_reward=True,
        )
        for i in range(3)
    ]
    resolution = RolloutResolution(
        step_actions=np.zeros((1, 2), dtype=np.int64),
        squad_batch_actions={0: np.zeros((3, 2), dtype=np.int64)},
        buffer_rows=rows,
        vec_rows=[],
        updated_lstm=None,
    )
    pre_obs = {k: v[np.newaxis, ...] for k, v in _dict_obs(0).items()}

    written = write_rollout_rows_to_buffer(
        buffer=buffer,
        resolution=resolution,
        pre_step_obs=pre_obs,
        rewards=np.array([1.0]),
        num_envs=1,
        is_recurrent=False,
        fallback_lstm=None,
        last_dones=np.array([False]),
        max_rows=1,
    )

    assert written == 1
    assert buffer.add.call_count == 1
    assert buffer.pos == 2
