"""Regression: need_forward rows must store pre-forward RNN state (episode_start=0)."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import get_schedule_fn

from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from src.agent.utils.disk_buffer import DiskDictRolloutBuffer
from train.squad_batch_inference import SquadBatchRolloutHelper, write_rollout_rows_to_buffer


def _obs_space():
    return spaces.Dict(
        {
            "map": spaces.Box(0, 1, (4, 128, 128), dtype=np.float32),
            "tactical": spaces.Box(0, 1, (2, 32, 32), dtype=np.float32),
            "command": spaces.Box(0, 1, (10,), dtype=np.float32),
        }
    )


def _dict_obs(seed: float):
    return {
        "map": np.full((4, 128, 128), seed, dtype=np.float32),
        "tactical": np.full((2, 32, 32), seed, dtype=np.float32),
        "command": np.array([seed, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def test_need_forward_stores_pre_forward_lstm_for_replay(tmp_path):
    policy = UniversalRecurrentPolicy(
        _obs_space(),
        spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32]),
        get_schedule_fn(3e-4),
        rnn_type="gru",
        lstm_hidden_size=32,
        features_extractor_class=UnitControlNetwork,
        features_extractor_kwargs={
            "features_dim": 64,
            "thinking_steps": 1,
            "agent_architecture": "universal",
        },
    )
    policy.action_dist = MaskableMultiCategoricalDistribution(
        policy.action_space.nvec
    )
    policy.set_training_mode(False)

    vec = MagicMock()
    vec.action_space = spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32])
    vec.env_method = MagicMock(
        side_effect=lambda name, *a, **kw: {
            "prepare_policy_batch": [None],
            "action_masks": [np.ones(FIXED_UNIVERSAL_COMMAND_DIM + 32 * 32, dtype=bool)],
            "get_obs_focus_unit_tag": [42],
        }.get(name, [None])
    )

    model = MagicMock()
    model.device = torch.device("cpu")
    model.policy = policy

    h = torch.randn(1, 1, 32)
    lstm_vec = RNNStates((h,), (h,))
    model._last_lstm_states = lstm_vec

    helper = SquadBatchRolloutHelper(1)
    obs = {k: v[np.newaxis, ...] for k, v in _dict_obs(0.5).items()}

    with torch.no_grad():
        resolution = helper.resolve_rollout(
            vec_env=vec,
            num_envs=1,
            obs=obs,
            requires_policy=np.array([True]),
            model=model,
            is_recurrent=True,
            last_dones=np.array([False]),
            prev_focus_tags=np.array([42], dtype=np.int64),
            lstm_states=lstm_vec,
            device=torch.device("cpu"),
        )

    assert len(resolution.vec_rows) == 1
    vec_row = resolution.vec_rows[0]
    assert vec_row.lstm_starts[0] == 0.0
    assert vec_row.pre_lstm_by_env[0] is not None
    torch.testing.assert_close(
        vec_row.pre_lstm_by_env[0].pi[0], h, atol=1e-6, rtol=0
    )

    buf = DiskDictRolloutBuffer(
        4,
        _obs_space(),
        spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32]),
        device="cpu",
        n_envs=1,
        temp_dir=str(tmp_path),
        storage_backend="ram",
        rnn_type="gru",
        hidden_size=32,
    )
    write_rollout_rows_to_buffer(
        buffer=buf,
        resolution=resolution,
        pre_step_obs=obs,
        rewards=np.array([0.0]),
        num_envs=1,
        is_recurrent=True,
        fallback_lstm=lstm_vec,
        last_dones=np.array([False]),
    )
    buf.full = True
    buf.compute_returns_and_advantage(torch.zeros((1, 1)), np.array([False]))

    policy.set_training_mode(True)
    batch = next(iter(buf.get(batch_size=8)))
    acts = batch.actions.long()
    _, log_p, _ = policy.evaluate_actions(
        batch.observations,
        acts,
        batch.lstm_states,
        batch.episode_starts,
        batch.action_masks,
    )
    mask = batch.mask > 1e-8
    diff = (log_p - batch.old_log_prob)[mask].abs().max().item()
    assert diff < 0.05, f"|log_p-old|={diff:.4f}"
