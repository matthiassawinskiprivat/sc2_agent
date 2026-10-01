"""
Rollout helper for CTF squad pointer architecture.

One policy forward and one buffer row per env step (no per-unit squad batching).
Recurrent state resets on episode done only — not on focal-unit change.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from gymnasium import spaces

from train.opponent_peek_ipc import (
    OpponentPeekSharedMemory,
    attach_opponent_peek_to_vec,
    pack_pending_opponent_actions,
    reconstruct_opponent_peek_obs,
)
from train.squad_batch_inference import (
    RolloutResolution,
    VecRolloutBufferRow,
    _gather_rnn_columns,
    _obs_tensor_to_finite,
    _scatter_rnn_columns,
    clone_rnn_states,
    normalize_vec_rnn_states,
    opponent_pointer_vec_predict,
    select_rnn_state_column,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates


def stack_info_action_masks(
    infos: Any, num_envs: int
) -> Optional[np.ndarray]:
    """Pull per-env flat masks from ``env.step`` / ``reset`` info dicts."""
    if infos is None:
        return None
    if isinstance(infos, dict):
        infos = [infos]
    if not isinstance(infos, (list, tuple)) or len(infos) < num_envs:
        return None
    rows: List[np.ndarray] = []
    for i in range(num_envs):
        info = infos[i]
        if not isinstance(info, dict):
            return None
        mask = info.get("action_masks")
        if mask is None:
            return None
        rows.append(np.asarray(mask, dtype=bool).reshape(-1))
    return np.stack(rows, axis=0)


def stack_info_requires_policy(
    infos: Any, num_envs: int
) -> Optional[np.ndarray]:
    """Pull per-env ``requires_policy_action`` flags from step/reset infos."""
    if infos is None:
        return None
    if isinstance(infos, dict):
        infos = [infos]
    if not isinstance(infos, (list, tuple)) or len(infos) < num_envs:
        return None
    flags: List[bool] = []
    for i in range(num_envs):
        info = infos[i]
        if not isinstance(info, dict) or "requires_policy_action" not in info:
            return None
        flags.append(bool(info["requires_policy_action"]))
    return np.asarray(flags, dtype=bool)


class SquadPointerRolloutHelper:
    """Single forward per env step for squad MultiDiscrete pointer actions."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._peek_shm: Optional[OpponentPeekSharedMemory] = None
        self._vec_has_attr: Dict[str, bool] = {}

    def bind_peek_shm(
        self,
        shm: Optional[OpponentPeekSharedMemory],
        vec_env=None,
        *,
        attach: bool = True,
    ) -> None:
        self._peek_shm = shm
        self._vec_has_attr.clear()
        if attach and shm is not None and vec_env is not None:
            attach_opponent_peek_to_vec(vec_env, shm)

    def _cached_has_attr(self, vec_env, name: str) -> bool:
        """One Subproc ``has_attr`` pipe per method name (workers do not grow methods)."""
        cached = self._vec_has_attr.get(name)
        if cached is not None:
            return cached
        if not hasattr(vec_env, "has_attr"):
            self._vec_has_attr[name] = True
            return True
        found = bool(vec_env.has_attr(name))
        self._vec_has_attr[name] = found
        return found

    def close_peek_shm(self, *, unlink: bool = True) -> None:
        shm = self._peek_shm
        self._peek_shm = None
        if shm is not None:
            shm.close(unlink=unlink)

    def reset(self) -> None:
        pass

    def clear_env(self, env_idx: int) -> None:
        pass

    def collector_profile_enabled(self) -> bool:
        """Opt-in wall timers for the collector process (``SC2_COLLECTOR_PROFILE``)."""
        flag = getattr(self, "_profile_flag", None)
        if flag is None:
            raw = os.environ.get("SC2_COLLECTOR_PROFILE", "").strip().lower()
            flag = raw in ("1", "true", "yes")
            self._profile_flag = flag
        return bool(flag)

    def profile_add(self, key: str, seconds: float) -> None:
        if not self.collector_profile_enabled():
            return
        sums = getattr(self, "_profile_sums", None)
        if sums is None:
            sums = {}
            self._profile_sums = sums
            self._profile_counts = {}
        sums[key] = sums.get(key, 0.0) + float(seconds)
        counts = self._profile_counts
        counts[key] = counts.get(key, 0) + 1

    def pop_collector_profile_means(self) -> Dict[str, float]:
        """Mean milliseconds per collector-side slice since the last pop."""
        sums = getattr(self, "_profile_sums", None) or {}
        counts = getattr(self, "_profile_counts", None) or {}
        out = {
            f"collector/profile_{key}_ms": 1000.0 * float(sums[key]) / max(counts.get(key, 1), 1)
            for key in sums
        }
        self._profile_sums = {}
        self._profile_counts = {}
        return out

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
        action_masks: Optional[np.ndarray] = None,
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
            if action_masks is not None and np.asarray(action_masks).shape[0] >= num_envs:
                sub_masks = np.asarray(action_masks)[need_forward]
            else:
                sub_masks = np.array(vec_env.env_method("action_masks"))[need_forward]
            # Match learner ``sanitize_obs_tensors_inplace`` so collected logπ
            # is scored on the same finite tensors PPO will train on.
            obs_t = _obs_tensor_to_finite(model.policy.obs_to_tensor(sub_obs)[0])
            m_t = torch.as_tensor(sub_masks).to(device)

            lstm_starts = np.zeros((len(need_forward),), dtype=np.float32)
            for j, env_idx in enumerate(need_forward):
                lstm_starts[j] = float(last_dones[env_idx])

            pre_lstm_by_env: Dict[int, Optional[RNNStates]] = {}
            # Include ``.cpu()`` so the sample waits for the GPU forward to finish.
            t_fwd = time.perf_counter() if self.collector_profile_enabled() else None
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
            if t_fwd is not None:
                self.profile_add("policy_fwd", time.perf_counter() - t_fwd)
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

        self._batch_opponent_pointer_actions(vec_env, model, num_envs)

        return RolloutResolution(
            step_actions=step_actions,
            squad_batch_actions={},
            buffer_rows=[],
            vec_rows=vec_rows,
            updated_lstm=updated_lstm,
        )

    def _batch_opponent_pointer_actions(self, vec_env, model, num_envs: int) -> None:
        """Gather opponent obs from workers, one CUDA forward, scatter actions."""
        if not hasattr(vec_env, "env_method"):
            return
        t_peek = time.perf_counter() if self.collector_profile_enabled() else None
        try:
            if not self._cached_has_attr(vec_env, "peek_opponent_squad_pointer"):
                return
            payloads = vec_env.env_method("peek_opponent_squad_pointer")
        except Exception:
            if t_peek is not None:
                self.profile_add("peek", time.perf_counter() - t_peek)
            return
        if t_peek is not None:
            self.profile_add("peek", time.perf_counter() - t_peek)
        if not payloads:
            return
        ready: List[tuple] = []
        obs_list: List[Dict[str, np.ndarray]] = []
        for env_idx, payload in enumerate(payloads):
            if not isinstance(payload, dict) or payload.get("obs") is None:
                continue
            obs = reconstruct_opponent_peek_obs(payload, self._peek_shm, env_idx)
            if obs is None:
                continue
            ready.append((env_idx, payload))
            obs_list.append(obs)
        if not ready:
            return
        mask_list = [p["mask"] for _, p in ready]
        lstm_list = [p.get("lstm") for _, p in ready]
        starts = [bool(p.get("episode_start", True)) for _, p in ready]
        t_opp = time.perf_counter() if self.collector_profile_enabled() else None
        try:
            actions, new_lstms = opponent_pointer_vec_predict(
                model,
                obs_list,
                mask_list,
                lstm_states_list=lstm_list,
                episode_starts=starts,
                deterministic=False,
            )
        except Exception:
            if t_opp is not None:
                self.profile_add("opp_fwd", time.perf_counter() - t_opp)
            return
        if t_opp is not None:
            self.profile_add("opp_fwd", time.perf_counter() - t_opp)
        packed = pack_pending_opponent_actions(num_envs, ready, actions, new_lstms)
        t_send = time.perf_counter() if self.collector_profile_enabled() else None
        try:
            try:
                if self._cached_has_attr(vec_env, "set_pending_opponent_squad_actions"):
                    vec_env.env_method("set_pending_opponent_squad_actions", packed)
                    return
            except Exception:
                self._vec_has_attr["set_pending_opponent_squad_actions"] = False
            for env_idx, item in enumerate(packed):
                if item is None:
                    continue
                action, lstm, pool_idx = item
                try:
                    vec_env.env_method(
                        "set_pending_opponent_squad_action",
                        action,
                        lstm,
                        pool_idx,
                        indices=[env_idx],
                    )
                except Exception:
                    return
        finally:
            if t_send is not None:
                self.profile_add("peek_send", time.perf_counter() - t_send)
