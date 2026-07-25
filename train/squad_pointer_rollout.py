"""
Rollout helper for CTF squad pointer architecture.

One policy forward and one buffer row per env step (no per-unit squad batching).
Recurrent state resets on episode done only — not on focal-unit change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from gymnasium import spaces

from train.squad_batch_inference import (
    RolloutResolution,
    VecRolloutBufferRow,
    _gather_rnn_columns,
    _scatter_rnn_columns,
    clone_rnn_states,
    normalize_vec_rnn_states,
    select_rnn_state_column,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class SquadPointerRolloutHelper:
    """Single forward per env step for squad MultiDiscrete pointer actions."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    def reset(self) -> None:
        pass

    def clear_env(self, env_idx: int) -> None:
        pass

    def resolve_rollout(
        self,
        *,
        vec_env,
        num_envs: int,
        obs: Dict[str, np.ndarray],
        requires_policy: np.ndarray,
        model,
        is_recurrent: bool,
        last_dones: np.ndarray,
        prev_focus_tags: np.ndarray,
        lstm_states: Optional[RNNStates],
        device: torch.device,
    ) -> RolloutResolution:
        del prev_focus_tags  # squad pointer: no focal-unit LSTM resets

        if isinstance(vec_env.action_space, spaces.MultiDiscrete):
            step_actions = np.zeros(
                (num_envs, len(vec_env.action_space.nvec)), dtype=np.int64
            )
        else:
            step_actions = np.zeros((num_envs,), dtype=np.int64)

        updated_lstm = lstm_states
        if is_recurrent and updated_lstm is not None:
            updated_lstm = normalize_vec_rnn_states(
                updated_lstm, model.policy, num_envs, device
            )

        need_forward = [i for i in range(num_envs) if requires_policy[i]]
        vec_rows: List[VecRolloutBufferRow] = []

        if need_forward:
            sub_obs = {k: v[need_forward] for k, v in obs.items()}
            sub_masks = np.array(vec_env.env_method("action_masks"))[need_forward]
            obs_t = model.policy.obs_to_tensor(sub_obs)[0]
            m_t = torch.as_tensor(sub_masks).to(device)

            lstm_starts = np.zeros((len(need_forward),), dtype=np.float32)
            for j, env_idx in enumerate(need_forward):
                lstm_starts[j] = float(last_dones[env_idx])

            pre_lstm_by_env: Dict[int, Optional[RNNStates]] = {}
            if is_recurrent:
                sub_starts = torch.as_tensor(lstm_starts, dtype=torch.float32, device=device)
                sub_lstm = (
                    _gather_rnn_columns(updated_lstm, need_forward)
                    if updated_lstm is not None
                    else None
                )
                pre_lstm_by_env = {
                    env_idx: select_rnn_state_column(sub_lstm, j)
                    for j, env_idx in enumerate(need_forward)
                }
                act_t, val_t, logp_t, new_sub_lstm = model.policy.forward(
                    obs_t, sub_lstm, sub_starts, action_masks=m_t
                )
                if updated_lstm is not None and new_sub_lstm is not None:
                    updated_lstm = _scatter_rnn_columns(
                        updated_lstm, need_forward, new_sub_lstm
                    )
            else:
                act_t, val_t, logp_t = model.policy.forward(
                    obs_t, deterministic=False, action_masks=m_t
                )

            act_np = act_t.cpu().numpy()
            for j, env_idx in enumerate(need_forward):
                step_actions[env_idx] = act_np[j]

            vec_rows.append(
                VecRolloutBufferRow(
                    env_indices=list(need_forward),
                    actions=act_np,
                    values=val_t.detach(),
                    log_probs=logp_t.detach(),
                    masks=sub_masks,
                    lstm_starts=lstm_starts,
                    pre_lstm_by_env=pre_lstm_by_env,
                )
            )
            fe = getattr(model.policy, "features_extractor", None)
            if fe is not None and hasattr(fe, "clear_forward_cache"):
                fe.clear_forward_cache()

        return RolloutResolution(
            step_actions=step_actions,
            squad_batch_actions={},
            buffer_rows=[],
            vec_rows=vec_rows,
            updated_lstm=updated_lstm,
        )
