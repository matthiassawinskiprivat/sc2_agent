"""Squad multi-env buffer rows: old_log_prob must match evaluate_actions (on-policy)."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.utils import get_schedule_fn

from src.agent.maskable_recurrent_ppo import MaskableRecurrentActorCriticPolicy
from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from src.agent.utils.disk_buffer import DiskDictRolloutBuffer
from train.squad_batch_inference import (
    agent_squad_batch_forward,
    build_buffer_lstm_states,
    merge_obs_row,
    normalize_single_env_rnn_states,
    select_rnn_state_column,
    zero_rnn_state_like,
)


def _obs_space():
    return spaces.Dict(
        {
            "map": spaces.Box(0, 1, (4, 128, 128), dtype=np.float32),
            "tactical": spaces.Box(0, 1, (2, 32, 32), dtype=np.float32),
            "command": spaces.Box(0, 1, (10,), dtype=np.float32),
        }
    )


def _make_policy(universal: bool = False):
    obs_space = _obs_space()
    act_space = spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32])
    if universal:
        policy = UniversalRecurrentPolicy(
            obs_space,
            act_space,
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
    else:
        policy = MaskableRecurrentActorCriticPolicy(
            obs_space,
            act_space,
            get_schedule_fn(3e-4),
            rnn_type="gru",
            lstm_hidden_size=32,
            features_extractor_class=CombinedExtractor,
        )
    policy.action_dist = MaskableMultiCategoricalDistribution(act_space.nvec)
    return policy


def _vec_obs(n_envs=2):
    return {
        "map": np.random.rand(n_envs, 4, 128, 128).astype(np.float32),
        "tactical": np.random.rand(n_envs, 2, 32, 32).astype(np.float32),
        "command": np.random.rand(n_envs, 10).astype(np.float32),
    }


def _unit_obs(tag: int):
    return {
        "map": np.full((4, 128, 128), tag * 0.01, dtype=np.float32),
        "tactical": np.full((2, 32, 32), tag * 0.02, dtype=np.float32),
        "command": np.array([float(tag)] + [0.0] * 9, dtype=np.float32),
    }


def _buffer_add_squad_row(
    buf,
    *,
    env_i: int,
    pre_step_obs,
    lstm_vec: RNNStates,
    lstm_before: RNNStates,
    unit_obs,
    action,
    value,
    log_prob,
    lstm_start: float,
    mask: np.ndarray,
):
    num_envs = pre_step_obs["map"].shape[0]
    policy_active = np.zeros((num_envs,), dtype=np.float32)
    policy_active[env_i] = 1.0
    lstm_starts = np.zeros((num_envs,), dtype=np.float32)
    lstm_starts[env_i] = lstm_start
    lstm_starts = np.maximum(lstm_starts, 1.0 - policy_active)

    row_actions = np.zeros((num_envs, 2), dtype=np.int64)
    row_actions[env_i] = action
    values = torch.zeros((num_envs, 1))
    values[env_i] = value.reshape(1)
    log_probs = torch.zeros((num_envs,))
    log_probs[env_i] = log_prob.reshape(())

    masks = np.zeros((num_envs, mask.shape[0]), dtype=bool)
    masks[env_i] = mask

    per_env = [None] * num_envs
    per_env[env_i] = lstm_before
    buffer_lstm = build_buffer_lstm_states(lstm_vec, per_env)

    buf.add(
        merge_obs_row(pre_step_obs, env_i, unit_obs),
        row_actions,
        np.zeros((num_envs,), dtype=np.float32),
        lstm_starts,
        values,
        log_probs,
        buffer_lstm,
        masks,
        policy_active=policy_active,
    )


@pytest.mark.parametrize("universal", [False, True])
def test_squad_pattern_log_prob_matches_evaluate(tmp_path, universal):
    policy = _make_policy(universal=universal)
    policy.set_training_mode(False)
    buf = DiskDictRolloutBuffer(
        12,
        _obs_space(),
        spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32]),
        device="cpu",
        n_envs=2,
        temp_dir=str(tmp_path),
        storage_backend="ram",
        rnn_type="gru",
        hidden_size=32,
    )

    h = torch.zeros(1, 2, 32)
    lstm_vec = RNNStates((h,), (h,))
    pre = _vec_obs(2)

    squad_units = [_unit_obs(10), _unit_obs(11), _unit_obs(12)]
    mask_list = [np.ones(FIXED_UNIVERSAL_COMMAND_DIM + 32 * 32, dtype=bool) for _ in squad_units]
    env_lstm = normalize_single_env_rnn_states(
        select_rnn_state_column(lstm_vec, 0), policy, policy.device
    )
    lstm_before_first = RNNStates(
        tuple(t.clone() for t in env_lstm.pi),
        tuple(t.clone() for t in env_lstm.vf),
    )
    zero_before = zero_rnn_state_like(env_lstm)

    act_t, val_t, logp_t, _, row_starts = agent_squad_batch_forward(
        policy,
        squad_units,
        mask_list,
        first_episode_start=1.0,
        lstm_states=env_lstm,
        is_recurrent=True,
        device=policy.device,
    )
    act_np = act_t.cpu().numpy()
    for j, uobs in enumerate(squad_units):
        _buffer_add_squad_row(
            buf,
            env_i=0,
            pre_step_obs=pre,
            lstm_vec=lstm_vec,
            lstm_before=deepcopy(lstm_before_first) if j == 0 else deepcopy(zero_before),
            unit_obs=uobs,
            action=act_np[j],
            value=val_t[j].detach(),
            log_prob=logp_t[j].detach(),
            lstm_start=row_starts[j],
            mask=mask_list[j],
        )

    # Env1 single row (env0 inactive) — interleaved layout
    pre2 = _vec_obs(2)
    env1_lstm = normalize_single_env_rnn_states(
        select_rnn_state_column(lstm_vec, 1), policy, policy.device
    )
    uobs1 = _unit_obs(20)
    mask1 = np.ones(FIXED_UNIVERSAL_COMMAND_DIM + 32 * 32, dtype=bool)
    with torch.no_grad():
        a1, v1, lp1, _ = policy.forward(
            policy.obs_to_tensor({k: v[np.newaxis] for k, v in uobs1.items()})[0],
            env1_lstm,
            torch.tensor([1.0]),
            action_masks=torch.as_tensor(mask1[np.newaxis]),
        )
    _buffer_add_squad_row(
        buf,
        env_i=1,
        pre_step_obs=pre2,
        lstm_vec=lstm_vec,
        lstm_before=env1_lstm,
        unit_obs=uobs1,
        action=a1.cpu().numpy()[0],
        value=v1[0].detach(),
        log_prob=lp1[0].detach(),
        lstm_start=1.0,
        mask=mask1,
    )

    buf.full = True
    buf.compute_returns_and_advantage(torch.zeros((2, 1)), np.array([False, False]))

    policy.set_training_mode(True)
    buf.to_device(torch.device("cpu"))
    batch = next(iter(buf.get(batch_size=64)))

    acts = batch.actions.long()
    _, log_p, _ = policy.evaluate_actions(
        batch.observations,
        acts,
        batch.lstm_states,
        batch.episode_starts,
        batch.action_masks,
    )
    mask = batch.mask > 1e-8
    diff = (log_p - batch.old_log_prob)[mask].abs()
    assert diff.mean().item() < 0.1, f"mean |log_p-old|={diff.mean().item():.4f}"
    approx_kl = (((torch.exp(log_p - batch.old_log_prob) - 1) - (log_p - batch.old_log_prob))[mask]).mean()
    assert approx_kl.item() < 0.05, f"approx_kl={approx_kl.item():.4f}"
