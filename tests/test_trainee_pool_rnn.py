"""Per-pool trainee RNN carry: swap on arena change, do not zero episode_starts."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from train.squad_pointer_rollout import (
    SquadPointerRolloutHelper,
    stack_info_trainee_pool_indices,
)


class _FakePolicy:
    rnn_type = "gru"

    def __init__(self, hidden=8):
        self.lstm_actor = torch.nn.GRU(hidden, hidden)
        self.seen_h = []

    def obs_to_tensor(self, obs):
        return ({k: torch.as_tensor(v) for k, v in obs.items()}, True)

    def set_training_mode(self, _mode):
        return None

    def forward(self, obs, lstm_states, episode_starts, action_masks=None, deterministic=False):
        h = lstm_states.pi[0]
        self.seen_h.append(h.detach().clone())
        starts = episode_starts.view(1, -1, 1)
        h_in = h * (1.0 - starts)
        new_h = h_in + 1.0
        n = h.shape[1]
        actions = torch.zeros((n, 2), dtype=torch.int64)
        values = torch.zeros((n, 1))
        logp = torch.zeros((n,))
        return actions, values, logp, RNNStates((new_h,), (new_h,))


def _obs(num_envs=1):
    return {
        "map": np.zeros((num_envs, 2, 4, 4), dtype=np.float32),
        "command": np.zeros((num_envs, 3), dtype=np.float32),
    }


def _vec_env(num_envs=1):
    vec = MagicMock()
    vec.action_space = spaces.MultiDiscrete([2, 4])
    vec.has_attr = MagicMock(return_value=False)
    vec.env_method = MagicMock(return_value=[np.ones(8, dtype=bool)] * num_envs)
    return vec


def test_stack_info_trainee_pool_indices():
    stacked = stack_info_trainee_pool_indices(
        [{"trainee_pool_index": 0}, {"trainee_pool_index": 2}],
        2,
    )
    assert stacked.tolist() == [0, 2]
    assert stack_info_trainee_pool_indices([{"action_masks": True}, {}], 2) is None


def test_pointer_helper_restores_pool_state_after_arena_swap():
    helper = SquadPointerRolloutHelper(1)
    vec = _vec_env(1)
    policy = _FakePolicy()
    model = MagicMock()
    model.device = torch.device("cpu")
    model.policy = policy
    model.policy.features_extractor = None

    h0 = torch.zeros(1, 1, 8)
    lstm = RNNStates((h0,), (h0,))
    masks = np.ones((1, 8), dtype=bool)

    def _run(pool, last_done=False):
        return helper.resolve_rollout(
            vec_env=vec,
            num_envs=1,
            obs=_obs(1),
            requires_policy=np.array([True]),
            model=model,
            is_recurrent=True,
            last_dones=np.array([last_done]),
            prev_focus_tags=np.array([-1], dtype=np.int64),
            lstm_states=lstm,
            device=torch.device("cpu"),
            action_masks=masks,
            trainee_pool_indices=np.array([pool], dtype=np.int64),
        )

    res_a = _run(0)
    assert res_a.vec_rows[0].lstm_starts[0] == 0.0
    assert res_a.vec_rows[0].rnn_context_ids[0] == 0
    h_after_a = policy.seen_h[0] + 1.0

    res_b = _run(2)
    assert res_b.vec_rows[0].lstm_starts[0] == 0.0
    assert res_b.vec_rows[0].rnn_context_ids[0] == 2
    torch.testing.assert_close(policy.seen_h[1], torch.zeros(1, 1, 8), atol=1e-6, rtol=0)

    res_a2 = _run(0)
    assert res_a2.vec_rows[0].lstm_starts[0] == 0.0
    torch.testing.assert_close(policy.seen_h[2], h_after_a, atol=1e-6, rtol=0)


def test_pointer_helper_episode_done_zeros_even_if_pool_cached():
    helper = SquadPointerRolloutHelper(1)
    vec = _vec_env(1)
    policy = _FakePolicy()
    model = MagicMock()
    model.device = torch.device("cpu")
    model.policy = policy
    model.policy.features_extractor = None
    h0 = torch.zeros(1, 1, 8)
    lstm = RNNStates((h0,), (h0,))
    masks = np.ones((1, 8), dtype=bool)

    helper.resolve_rollout(
        vec_env=vec,
        num_envs=1,
        obs=_obs(1),
        requires_policy=np.array([True]),
        model=model,
        is_recurrent=True,
        last_dones=np.array([False]),
        prev_focus_tags=np.array([-1], dtype=np.int64),
        lstm_states=lstm,
        device=torch.device("cpu"),
        action_masks=masks,
        trainee_pool_indices=np.array([0], dtype=np.int64),
    )
    helper.clear_env(0)
    helper.resolve_rollout(
        vec_env=vec,
        num_envs=1,
        obs=_obs(1),
        requires_policy=np.array([True]),
        model=model,
        is_recurrent=True,
        last_dones=np.array([True]),
        prev_focus_tags=np.array([-1], dtype=np.int64),
        lstm_states=lstm,
        device=torch.device("cpu"),
        action_masks=masks,
        trainee_pool_indices=np.array([0], dtype=np.int64),
    )
    assert helper.resolve_rollout
    torch.testing.assert_close(policy.seen_h[1], torch.zeros(1, 1, 8), atol=1e-6, rtol=0)


def test_gather_vec_lstm_for_pools_uses_next_arena_carry():
    helper = SquadPointerRolloutHelper(1)
    policy = _FakePolicy()
    h = torch.full((1, 1, 8), 3.0)
    helper._trainee_pool_lstm[(0, 2)] = RNNStates((h,), (h,))
    vec, starts = helper.gather_vec_lstm_for_pools(
        pool_indices=np.array([2], dtype=np.int64),
        last_dones=np.array([False]),
        policy=policy,
        device=torch.device("cpu"),
        num_envs=1,
    )
    assert starts[0] == 0.0
    torch.testing.assert_close(vec.pi[0], h, atol=1e-6, rtol=0)
    _, starts_done = helper.gather_vec_lstm_for_pools(
        pool_indices=np.array([2], dtype=np.int64),
        last_dones=np.array([True]),
        policy=policy,
        device=torch.device("cpu"),
        num_envs=1,
    )
    assert starts_done[0] == 1.0


def test_aba_logprob_matches_stored_pre_state(tmp_path):
    """Learner must score the third A-step with the stored A carry, not B's output."""
    from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution
    from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
    from src.agent.models.unit_control_network import UnitControlNetwork
    from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
    from src.agent.utils.disk_buffer import DiskDictRolloutBuffer
    from stable_baselines3.common.utils import get_schedule_fn
    from train.squad_batch_inference import write_rollout_rows_to_buffer

    obs_space = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (4, 128, 128), dtype=np.float32),
            "tactical": spaces.Box(0, 1, (2, 32, 32), dtype=np.float32),
            "command": spaces.Box(0, 1, (10,), dtype=np.float32),
        }
    )
    action_space = spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32])
    policy = UniversalRecurrentPolicy(
        obs_space,
        action_space,
        get_schedule_fn(3e-4),
        rnn_type="gru",
        lstm_hidden_size=16,
        features_extractor_class=UnitControlNetwork,
        features_extractor_kwargs={
            "features_dim": 32,
            "thinking_steps": 1,
            "agent_architecture": "universal",
        },
    )
    policy.action_dist = MaskableMultiCategoricalDistribution(policy.action_space.nvec)
    policy.set_training_mode(False)
    helper = SquadPointerRolloutHelper(1)
    vec = MagicMock()
    vec.action_space = action_space
    vec.has_attr = MagicMock(return_value=False)
    mask = np.ones(FIXED_UNIVERSAL_COMMAND_DIM + 32 * 32, dtype=bool)
    vec.env_method = MagicMock(return_value=[mask])
    model = MagicMock()
    model.device = torch.device("cpu")
    model.policy = policy

    def _row(seed, pool):
        obs = {
            "map": np.full((1, 4, 128, 128), seed, dtype=np.float32),
            "tactical": np.full((1, 2, 32, 32), seed, dtype=np.float32),
            "command": np.array([[seed] + [0.0] * 9], dtype=np.float32),
        }
        h = torch.zeros(1, 1, 16)
        with torch.no_grad():
            resolution = helper.resolve_rollout(
                vec_env=vec,
                num_envs=1,
                obs=obs,
                requires_policy=np.array([True]),
                model=model,
                is_recurrent=True,
                last_dones=np.array([False]),
                prev_focus_tags=np.array([-1], dtype=np.int64),
                lstm_states=RNNStates((h,), (h,)),
                device=torch.device("cpu"),
                action_masks=mask[None],
                trainee_pool_indices=np.array([pool], dtype=np.int64),
            )
        return obs, resolution

    buf = DiskDictRolloutBuffer(
        4,
        obs_space,
        action_space,
        device="cpu",
        n_envs=1,
        temp_dir=str(tmp_path),
        storage_backend="ram",
        rnn_type="gru",
        hidden_size=16,
    )
    for seed, pool in ((0.1, 0), (0.4, 2), (0.7, 0)):
        obs, resolution = _row(seed, pool)
        write_rollout_rows_to_buffer(
            buffer=buf,
            resolution=resolution,
            pre_step_obs=obs,
            rewards=np.array([0.0]),
            num_envs=1,
            is_recurrent=True,
            fallback_lstm=RNNStates((torch.zeros(1, 1, 16),), (torch.zeros(1, 1, 16),)),
            last_dones=np.array([False]),
        )
    assert buf.pos == 3
    assert buf.rnn_context_ids[:3, 0].tolist() == [0, 2, 0]
    assert buf.episode_starts[:3, 0].tolist() == [0.0, 0.0, 0.0]

    policy.set_training_mode(True)
    t, e = 2, 0
    obs_t = {
        k: torch.as_tensor(np.asarray(arr[t, e])[None].copy())
        for k, arr in buf.observations.items()
    }
    acts = torch.as_tensor(np.asarray(buf.actions[t, e])[None].copy()).long()
    starts = torch.as_tensor([float(buf.episode_starts[t, e])])
    masks = torch.as_tensor(np.asarray(buf.action_masks[t, e])[None].copy())
    h_pi = torch.as_tensor(np.asarray(buf.hidden_states_pi[t, :, e, :])[:, None, :].copy())
    h_vf = torch.as_tensor(np.asarray(buf.hidden_states_vf[t, :, e, :])[:, None, :].copy())
    _, new_lp, _ = policy.evaluate_actions(
        obs_t, acts, RNNStates((h_pi,), (h_vf,)), starts, masks
    )
    old = float(buf.log_probs[t, e])
    assert abs(float(new_lp.reshape(-1)[0]) - old) < 0.05
