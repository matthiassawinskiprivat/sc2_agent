"""
Batched policy inference for CTF squad round-robin.

Agent squads: one parallel forward for all units at squad start (RNN resets on focal-unit
change match the sequential collector). Remaining unit actions are queued and applied on
subsequent env steps without extra policy calls.

Opponent squads: one sequential RNN pass (or single batch forward for feed-forward policies)
inside the env step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from sb3_contrib.common.recurrent.type_aliases import RNNStates


def stack_env_dict_observations(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not obs_list:
        raise ValueError("obs_list must not be empty")
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}


def stack_action_masks(masks_list: List[np.ndarray]) -> np.ndarray:
    return np.stack(masks_list, axis=0)


def sanitize_flat_action_mask(mask: np.ndarray, command_dim: int = 64) -> np.ndarray:
    """
    Ensure at least one command head stays valid.

    All-false command slices yield zero-sum CUDA multinomial probs (device-side assert).
    """
    out = np.asarray(mask, dtype=bool).copy()
    n = min(int(command_dim), out.shape[0])
    if n > 0 and not out[:n].any():
        out[0] = True
    return out


def _obs_tensor_to_finite(obs_t):
    if isinstance(obs_t, dict):
        return {k: torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0) for k, v in obs_t.items()}
    return torch.nan_to_num(obs_t, nan=0.0, posinf=1.0, neginf=0.0)


def _rnn_layers_hidden(policy) -> Tuple[int, int]:
    lstm = policy.lstm_actor
    return lstm.num_layers, lstm.hidden_size


def _rnn_type_name(policy) -> str:
    return getattr(policy, "rnn_type", "gru").lower()


def _unpack_single_env_rnn_states(
    lstm_states: Optional[Any],
) -> Tuple[Optional[Tuple[Any, ...]], Optional[Tuple[Any, ...]]]:
    """Accept RNNStates or (pi, vf) tuples from ``model.predict`` / opponent batch predict."""
    if lstm_states is None:
        return None, None
    if isinstance(lstm_states, RNNStates):
        return lstm_states.pi, lstm_states.vf
    if (
        isinstance(lstm_states, tuple)
        and len(lstm_states) == 2
        and isinstance(lstm_states[0], tuple)
    ):
        return lstm_states[0], lstm_states[1]
    raise TypeError(
        f"Unsupported lstm_states type {type(lstm_states)!r}; "
        "expected RNNStates or (pi_tuple, vf_tuple)."
    )


def normalize_single_env_rnn_states(
    lstm_states: Optional[Any],
    policy,
    device: torch.device,
) -> RNNStates:
    """Coerce RNN states to shape (num_layers, 1, hidden_size) for squad-batch expand."""
    n_layers, hidden = _rnn_layers_hidden(policy)
    rnn_type = _rnn_type_name(policy)

    def _fresh() -> Tuple[torch.Tensor, ...]:
        h = torch.zeros(n_layers, 1, hidden, device=device)
        return (h, torch.zeros_like(h)) if rnn_type == "lstm" else (h,)

    def _normalize_tuple(state_tuple: Optional[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        if not state_tuple:
            return _fresh()
        out: List[torch.Tensor] = []
        for t in state_tuple:
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t, device=device)
            else:
                t = t.to(device)
            if t.ndim == 1:
                t = t.reshape(n_layers, 1, hidden)
            elif t.ndim == 2:
                t = t.reshape(n_layers, 1, hidden)
            elif t.ndim == 3:
                if t.shape[0] != n_layers:
                    t = t[:n_layers]
                if t.shape[1] == 0 or t.shape[1] > 1:
                    # Use first env column when vec-shaped; rebuild when empty.
                    t = (
                        t[:, :1, :]
                        if t.shape[1] > 0
                        else torch.zeros(n_layers, 1, hidden, device=device, dtype=t.dtype)
                    )
                if t.shape[2] != hidden:
                    t = t[..., :hidden].reshape(n_layers, 1, hidden)
            else:
                t = torch.zeros(n_layers, 1, hidden, device=device)
            out.append(t)
        if rnn_type == "lstm" and len(out) == 1:
            return (out[0], torch.zeros_like(out[0]))
        return tuple(out)

    pi_raw, vf_raw = _unpack_single_env_rnn_states(lstm_states)
    if pi_raw is None:
        fresh = _fresh()
        return RNNStates(fresh, fresh)
    return RNNStates(
        _normalize_tuple(pi_raw),
        _normalize_tuple(vf_raw),
    )


def normalize_vec_rnn_states(
    lstm_states: RNNStates,
    policy,
    num_envs: int,
    device: torch.device,
) -> RNNStates:
    """Ensure vec-env RNN states are (num_layers, num_envs, hidden_size)."""
    n_layers, hidden = _rnn_layers_hidden(policy)
    rnn_type = _rnn_type_name(policy)

    def _fresh() -> Tuple[torch.Tensor, ...]:
        h = torch.zeros(n_layers, num_envs, hidden, device=device)
        return (h, torch.zeros_like(h)) if rnn_type == "lstm" else (h,)

    def _normalize_tuple(state_tuple: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        if not state_tuple:
            return _fresh()
        out: List[torch.Tensor] = []
        for t in state_tuple:
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t, device=device)
            else:
                t = t.to(device)
            if t.ndim == 1:
                t = t.reshape(n_layers, 1, hidden).repeat(1, num_envs, 1)
            elif t.ndim == 2:
                t = t.reshape(n_layers, 1, hidden).repeat(1, num_envs, 1)
            elif t.ndim == 3:
                if t.shape[0] != n_layers:
                    t = t[:n_layers]
                if t.shape[1] == 0:
                    t = torch.zeros(n_layers, num_envs, hidden, device=device, dtype=t.dtype)
                elif t.shape[1] == 1 and num_envs > 1:
                    t = t.repeat(1, num_envs, 1)
                elif t.shape[1] > num_envs:
                    t = t[:, :num_envs, :]
                elif t.shape[1] < num_envs:
                    pad = torch.zeros(
                        n_layers, num_envs - t.shape[1], hidden, device=device, dtype=t.dtype
                    )
                    t = torch.cat([t, pad], dim=1)
                if t.shape[2] != hidden:
                    t = t[..., :hidden]
            else:
                t = torch.zeros(n_layers, num_envs, hidden, device=device)
            out.append(t)
        if rnn_type == "lstm" and len(out) == 1:
            return (out[0], torch.zeros_like(out[0]))
        return tuple(out)

    return RNNStates(
        _normalize_tuple(lstm_states.pi),
        _normalize_tuple(lstm_states.vf),
    )


def ensure_collector_lstm_states(model, num_envs: int) -> None:
    """Reset collector LSTM carry if missing or wrong shape after checkpoint load."""
    if not hasattr(model.policy, "lstm_actor"):
        return
    device = model.device
    policy = model.policy
    n_layers, hidden = _rnn_layers_hidden(policy)
    rnn_type = _rnn_type_name(model)

    def _ok(state_tuple: Tuple[torch.Tensor, ...]) -> bool:
        return (
            bool(state_tuple)
            and isinstance(state_tuple[0], torch.Tensor)
            and state_tuple[0].ndim == 3
            and state_tuple[0].shape[1] == num_envs
        )

    if (
        not hasattr(model, "_last_lstm_states")
        or model._last_lstm_states is None
        or not _ok(model._last_lstm_states.pi)
        or not _ok(model._last_lstm_states.vf)
    ):
        h = torch.zeros(n_layers, num_envs, hidden, device=device)
        fresh = (h, torch.zeros_like(h)) if rnn_type == "lstm" else (h,)
        model._last_lstm_states = RNNStates(fresh, fresh)


def expand_rnn_state_for_batch(
    lstm_states: RNNStates, batch_size: int, device: torch.device
) -> RNNStates:
    """Replicate a single-env RNN state across ``batch_size`` parallel columns."""

    def _expand_tuple(state_tuple: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        out = []
        for t in state_tuple:
            t = t.to(device)
            env_dim = t.shape[1] if t.ndim == 3 else 0
            if env_dim == batch_size:
                out.append(t)
            elif env_dim == 1:
                out.append(t.repeat(1, batch_size, 1))
            elif env_dim == 0:
                # Degenerate slice from a bad carry; treat as zeros then expand.
                layers, _, hidden = t.shape if t.ndim == 3 else (t.shape[0], 1, t.shape[-1])
                out.append(
                    torch.zeros(layers, batch_size, hidden, device=device, dtype=t.dtype)
                )
            else:
                raise ValueError(
                    f"Unexpected RNN state env dim {env_dim} for batch_size={batch_size}"
                )
        return tuple(out)

    return RNNStates(_expand_tuple(lstm_states.pi), _expand_tuple(lstm_states.vf))


def select_rnn_state_column(lstm_states: RNNStates, col: int) -> RNNStates:
    """Extract column ``col`` from batched RNN states (single-env slice)."""

    def _select(state_tuple: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple(s[:, col : col + 1, :].contiguous() for s in state_tuple)

    return RNNStates(_select(lstm_states.pi), _select(lstm_states.vf))


def zero_rnn_state_like(lstm_states: RNNStates) -> RNNStates:
    """Return a same-shape zero RNN state (for buffer entries after focal-unit reset)."""

    def _zero(state_tuple: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.zeros_like(s) for s in state_tuple)

    return RNNStates(_zero(lstm_states.pi), _zero(lstm_states.vf))


def clone_rnn_states(lstm_states: Optional[RNNStates]) -> Optional[RNNStates]:
    """Detached clone for buffer storage (safe after autograd forwards)."""
    if lstm_states is None:
        return None
    return RNNStates(
        tuple(t.clone().detach() for t in lstm_states.pi),
        tuple(t.clone().detach() for t in lstm_states.vf),
    )


def merge_obs_row(
    vec_obs: Dict[str, np.ndarray],
    env_idx: int,
    single_obs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Replace one row of a vectorized Dict observation."""
    merged = {}
    for key, arr in vec_obs.items():
        row = arr.copy()
        row[env_idx] = single_obs[key]
        merged[key] = row
    return merged


def agent_squad_batch_forward(
    policy,
    obs_list: List[Dict[str, np.ndarray]],
    masks_list: List[np.ndarray],
    *,
    first_episode_start: float,
    lstm_states: Optional[RNNStates],
    is_recurrent: bool,
    device: torch.device,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[RNNStates], List[float]]:
    """
    Forward pass for all units in an agent squad at once.

    Returns (actions, values, log_probs, updated_lstm_states, lstm_starts_per_row).
    """
    n = len(obs_list)
    batch_obs = stack_env_dict_observations(obs_list)
    obs_t = _obs_tensor_to_finite(policy.obs_to_tensor(batch_obs)[0])
    cmd_dim = int(policy.action_space.nvec[0]) if hasattr(policy, "action_space") else 64
    sanitized_masks = [sanitize_flat_action_mask(m, cmd_dim) for m in masks_list]
    masks_t = torch.as_tensor(stack_action_masks(sanitized_masks)).to(device)

    lstm_starts = [float(first_episode_start)] + [1.0] * (n - 1)

    if not is_recurrent:
        actions, values, log_probs = policy.forward(
            obs_t, deterministic=deterministic, action_masks=masks_t
        )
        return actions, values, log_probs, lstm_states, lstm_starts

    starts_t = torch.as_tensor(lstm_starts, dtype=torch.float32, device=device)
    single_lstm = normalize_single_env_rnn_states(lstm_states, policy, device)
    expanded = expand_rnn_state_for_batch(single_lstm, n, device)
    actions, values, log_probs, new_lstm = policy.forward(
        obs_t, expanded, starts_t, action_masks=masks_t, deterministic=deterministic
    )
    env_lstm = select_rnn_state_column(new_lstm, n - 1)
    return actions, values, log_probs, env_lstm, lstm_starts


def opponent_squad_batch_predict(
    model,
    obs_list: List[Dict[str, np.ndarray]],
    masks_list: List[np.ndarray],
    *,
    lstm_state: Optional[Any] = None,
    deterministic: bool = True,
    focus_tags: Optional[List[int]] = None,
    last_focus_tag: Optional[int] = None,
) -> Tuple[np.ndarray, Any, Optional[int]]:
    """
    Predict actions for every unit in an opponent squad or chunk.

    Feed-forward policies use one batched forward. Recurrent policies use the same
    batched RNN forward as agent squad collection (via ``agent_squad_batch_forward``).
    """
    policy = model.policy
    policy.set_training_mode(False)
    device = model.device
    n = len(obs_list)
    if n == 0:
        return np.zeros((0,), dtype=np.int64), lstm_state, last_focus_tag

    with torch.no_grad():
        is_recurrent = isinstance(getattr(policy, "lstm_actor", None), torch.nn.Module)

        if not is_recurrent:
            batch_obs = stack_env_dict_observations(obs_list)
            obs_t = policy.obs_to_tensor(batch_obs)[0]
            masks_t = torch.as_tensor(stack_action_masks(masks_list)).to(device)
            actions, _, _ = policy.forward(
                obs_t, deterministic=deterministic, action_masks=masks_t
            )
            new_last = focus_tags[-1] if focus_tags else last_focus_tag
            return actions.cpu().numpy(), lstm_state, new_last

        first_start = 1.0
        if focus_tags and last_focus_tag is not None and focus_tags[0] == last_focus_tag:
            first_start = 0.0

        norm_rnn = normalize_single_env_rnn_states(lstm_state, policy, device)
        actions_t, _, _, new_lstm, _ = agent_squad_batch_forward(
            policy,
            obs_list,
            masks_list,
            first_episode_start=first_start,
            lstm_states=norm_rnn,
            is_recurrent=True,
            device=device,
            deterministic=deterministic,
        )
        new_state = None
        if new_lstm is not None:
            pi_np = tuple(t.cpu().numpy() for t in new_lstm.pi)
            new_state = (pi_np, pi_np)
        new_last = focus_tags[-1] if focus_tags else last_focus_tag
        return actions_t.cpu().numpy(), new_state, new_last


@dataclass
class CachedSquadStep:
    obs: Dict[str, np.ndarray]
    action: np.ndarray
    value: torch.Tensor
    log_prob: torch.Tensor
    mask: np.ndarray
    lstm_start: float
    lstm_states_before: Optional[RNNStates]


@dataclass
class RolloutBufferRow:
    """One learner transition to write after a (possibly multi-unit) env step."""

    env_idx: int
    cached: CachedSquadStep
    apply_reward: bool


@dataclass
class VecRolloutBufferRow:
    """One vec-env timestep: all ``env_indices`` acted in a single batched forward."""

    env_indices: List[int]
    actions: np.ndarray
    values: torch.Tensor
    log_probs: torch.Tensor
    masks: np.ndarray
    lstm_starts: np.ndarray
    pre_lstm_by_env: Dict[int, Optional[RNNStates]]


@dataclass
class RolloutResolution:
    """One vec-env step plus all learner rows produced by that step."""

    step_actions: np.ndarray
    squad_batch_actions: Dict[int, np.ndarray]
    buffer_rows: List[RolloutBufferRow]
    vec_rows: List[VecRolloutBufferRow]
    updated_lstm: Optional[RNNStates]


def write_vec_rollout_row_to_buffer(
    *,
    buffer,
    pre_step_obs: Dict[str, np.ndarray],
    vec_row: VecRolloutBufferRow,
    rewards: np.ndarray,
    num_envs: int,
    is_recurrent: bool,
    fallback_lstm: Optional[RNNStates],
) -> int:
    """Append one buffer row for a batched multi-env forward (standard vec-env layout)."""
    env_indices = vec_row.env_indices
    policy_active = np.zeros((num_envs,), dtype=np.float32)
    policy_active[env_indices] = 1.0

    action_dim = int(vec_row.actions.shape[-1]) if vec_row.actions.ndim > 1 else 1
    if action_dim > 1:
        row_actions = np.zeros((num_envs, action_dim), dtype=np.int64)
    else:
        row_actions = np.zeros((num_envs,), dtype=np.int64)

    values = torch.zeros((num_envs, 1))
    log_probs = torch.zeros((num_envs,))
    mask_len = int(vec_row.masks.shape[-1])
    masks = np.zeros((num_envs, mask_len), dtype=bool)
    lstm_starts = np.zeros((num_envs,), dtype=np.float32)

    for j, env_i in enumerate(env_indices):
        row_actions[env_i] = vec_row.actions[j]
        values[env_i] = vec_row.values[j].reshape(1)
        log_probs[env_i] = vec_row.log_probs[j].reshape(())
        masks[env_i] = vec_row.masks[j]
        lstm_starts[env_i] = float(vec_row.lstm_starts[j])

    lstm_starts = np.maximum(lstm_starts, 1.0 - policy_active)

    lstm_before: List[Optional[RNNStates]] = [None] * num_envs
    for env_i in env_indices:
        lstm_before[env_i] = vec_row.pre_lstm_by_env.get(env_i)
    buffer_lstm = (
        build_buffer_lstm_states(fallback_lstm, lstm_before)
        if is_recurrent and fallback_lstm is not None
        else None
    )

    buffer.add(
        pre_step_obs,
        row_actions,
        np.asarray(rewards, dtype=np.float32).reshape(num_envs),
        lstm_starts,
        values,
        log_probs,
        buffer_lstm,
        masks,
        policy_active=policy_active,
    )
    return 1


def write_rollout_rows_to_buffer(
    *,
    buffer,
    resolution: RolloutResolution,
    pre_step_obs: Dict[str, np.ndarray],
    rewards: np.ndarray,
    num_envs: int,
    is_recurrent: bool,
    fallback_lstm: Optional[RNNStates],
    last_dones: np.ndarray,
    max_rows: Optional[int] = None,
) -> int:
    """Append transition rows for one env step. Returns number of rows written."""
    written = 0
    buffer_size = getattr(buffer, "buffer_size", None)
    for vec_row in resolution.vec_rows:
        if max_rows is not None and written >= max_rows:
            break
        if buffer_size is not None and buffer.pos >= buffer_size:
            break
        written += write_vec_rollout_row_to_buffer(
            buffer=buffer,
            pre_step_obs=pre_step_obs,
            vec_row=vec_row,
            rewards=rewards,
            num_envs=num_envs,
            is_recurrent=is_recurrent,
            fallback_lstm=fallback_lstm,
        )

    for row in resolution.buffer_rows:
        if max_rows is not None and written >= max_rows:
            break
        if buffer_size is not None and buffer.pos >= buffer_size:
            break
        env_i = row.env_idx
        step_rewards = np.zeros((num_envs,), dtype=np.float32)
        if row.apply_reward:
            step_rewards[env_i] = rewards[env_i]

        obs_buf = merge_obs_row(pre_step_obs, env_i, row.cached.obs)

        cached_action = np.asarray(row.cached.action, dtype=np.int64)
        if cached_action.ndim == 0:
            row_actions = np.zeros((num_envs,), dtype=np.int64)
            row_actions[env_i] = cached_action
        else:
            row_actions = np.zeros(
                (num_envs, cached_action.shape[0]), dtype=np.int64
            )
            row_actions[env_i] = cached_action

        values = torch.zeros((num_envs, 1), device=row.cached.value.device)
        values[env_i] = row.cached.value.reshape(1)
        log_probs = torch.zeros((num_envs,), device=row.cached.log_prob.device)
        log_probs[env_i] = row.cached.log_prob.reshape(())

        mask_len = int(row.cached.mask.shape[0])
        masks = np.zeros((num_envs, mask_len), dtype=bool)
        masks[env_i] = row.cached.mask

        lstm_starts = np.zeros((num_envs,), dtype=np.float32)
        lstm_starts[env_i] = row.cached.lstm_start

        policy_active = np.zeros((num_envs,), dtype=np.float32)
        policy_active[env_i] = 1.0
        lstm_starts = np.maximum(lstm_starts, 1.0 - policy_active)

        lstm_before = [None] * num_envs
        lstm_before[env_i] = row.cached.lstm_states_before
        buffer_lstm = (
            build_buffer_lstm_states(fallback_lstm, lstm_before)
            if is_recurrent and fallback_lstm is not None
            else None
        )

        buffer.add(
            obs_buf,
            row_actions,
            step_rewards,
            lstm_starts,
            values,
            log_probs,
            buffer_lstm,
            masks,
            policy_active=policy_active,
        )
        written += 1
    return written


class SquadBatchRolloutHelper:
    """Batched squad inference + one env step per full agent squad."""

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
        """One policy pass + one vec-env step can produce multiple learner rows."""
        from gymnasium import spaces

        if isinstance(vec_env.action_space, spaces.MultiDiscrete):
            step_actions = np.zeros(
                (num_envs, len(vec_env.action_space.nvec)), dtype=np.int64
            )
        else:
            step_actions = np.zeros((num_envs,), dtype=np.int64)

        buffer_rows: List[RolloutBufferRow] = []
        vec_rows: List[VecRolloutBufferRow] = []
        squad_batch_actions: Dict[int, np.ndarray] = {}
        updated_lstm = lstm_states
        if is_recurrent and updated_lstm is not None:
            updated_lstm = normalize_vec_rnn_states(
                updated_lstm, model.policy, num_envs, device
            )

        try:
            batch_prepared = vec_env.env_method("prepare_policy_batch")
        except Exception:
            try:
                batch_prepared = vec_env.env_method("prepare_squad_policy_batch")
            except Exception:
                batch_prepared = [None] * num_envs

        handled = np.zeros((num_envs,), dtype=bool)

        for env_idx in range(num_envs):
            if not requires_policy[env_idx]:
                continue
            prepared = batch_prepared[env_idx]
            if prepared is None:
                continue
            obs_list = prepared.get("obs", [])
            if not obs_list:
                continue
            mask_list = prepared["masks"]
            focus_tags = prepared["focus_tags"]
            first_start = float(
                last_dones[env_idx] or focus_tags[0] != prev_focus_tags[env_idx]
            )

            env_lstm = (
                select_rnn_state_column(updated_lstm, env_idx)
                if is_recurrent and updated_lstm is not None
                else None
            )
            lstm_before_first = clone_rnn_states(env_lstm)

            act_t, val_t, logp_t, new_env_lstm, row_starts = agent_squad_batch_forward(
                model.policy,
                obs_list,
                mask_list,
                first_episode_start=first_start,
                lstm_states=env_lstm,
                is_recurrent=is_recurrent,
                device=device,
            )

            act_np = act_t.cpu().numpy()
            step_actions[env_idx] = act_np[0]
            squad_batch_actions[env_idx] = act_np

            if is_recurrent and updated_lstm is not None and new_env_lstm is not None:
                updated_lstm = _write_env_lstm_column(updated_lstm, env_idx, new_env_lstm)

            zero_before = zero_rnn_state_like(env_lstm) if env_lstm is not None else None
            for j, obs_j in enumerate(obs_list):
                buffer_rows.append(
                    RolloutBufferRow(
                        env_idx=env_idx,
                        cached=CachedSquadStep(
                            obs=obs_j,
                            action=act_np[j],
                            value=val_t[j].detach(),
                            log_prob=logp_t[j].detach(),
                            mask=mask_list[j],
                            lstm_start=row_starts[j],
                            lstm_states_before=(
                                clone_rnn_states(lstm_before_first)
                                if j == 0
                                else clone_rnn_states(zero_before)
                            ),
                        ),
                        # Joint squad physics step: every unit shares the same scalar reward.
                        apply_reward=True,
                    )
                )
            handled[env_idx] = True

        need_forward = [
            i for i in range(num_envs) if requires_policy[i] and not handled[i]
        ]

        if need_forward:
            sub_obs = {k: v[need_forward] for k, v in obs.items()}
            sub_masks = np.array(vec_env.env_method("action_masks"))[need_forward]
            obs_t = model.policy.obs_to_tensor(sub_obs)[0]
            m_t = torch.as_tensor(sub_masks).to(device)

            lstm_starts = np.zeros((len(need_forward),), dtype=np.float32)
            for j, env_idx in enumerate(need_forward):
                focus = vec_env.env_method("get_obs_focus_unit_tag", indices=[env_idx])[0]
                lstm_starts[j] = float(
                    last_dones[env_idx] or focus != prev_focus_tags[env_idx]
                )

            pre_lstm_by_env: Dict[int, Optional[RNNStates]] = {}
            if is_recurrent:
                sub_starts = torch.as_tensor(lstm_starts, dtype=torch.float32, device=device)
                sub_lstm = (
                    _gather_rnn_columns(updated_lstm, need_forward)
                    if updated_lstm is not None
                    else updated_lstm
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

        return RolloutResolution(
            step_actions=step_actions,
            squad_batch_actions=squad_batch_actions,
            buffer_rows=buffer_rows,
            vec_rows=vec_rows,
            updated_lstm=updated_lstm,
        )


def build_buffer_lstm_states(
    fallback: RNNStates,
    per_env: List[Optional[RNNStates]],
) -> RNNStates:
    """Merge per-env pre-forward RNN states into one vec-env ``RNNStates`` for buffer storage."""
    pi_tensors = [t.clone() for t in fallback.pi]
    vf_tensors = [t.clone() for t in fallback.vf]
    for env_idx, state in enumerate(per_env):
        if state is None:
            continue
        for i, src in enumerate(state.pi):
            pi_tensors[i][:, env_idx : env_idx + 1, :] = src.to(pi_tensors[i].device)
        for i, src in enumerate(state.vf):
            vf_tensors[i][:, env_idx : env_idx + 1, :] = src.to(vf_tensors[i].device)
    return RNNStates(tuple(pi_tensors), tuple(vf_tensors))


def _write_env_lstm_column(
    batched_lstm: RNNStates, env_idx: int, env_lstm: RNNStates
) -> RNNStates:
    """Copy single-env RNN state into column ``env_idx`` of a vec-env RNNStates bundle."""

    def _write(
        dst_tuple: Tuple[torch.Tensor, ...],
        src_tuple: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        out = []
        for dst, src in zip(dst_tuple, src_tuple):
            new = dst.clone()
            new[:, env_idx : env_idx + 1, :] = src.to(dst.device)
            out.append(new)
        return tuple(out)

    return RNNStates(
        _write(batched_lstm.pi, env_lstm.pi),
        _write(batched_lstm.vf, env_lstm.vf),
    )


def _gather_rnn_columns(lstm_states: RNNStates, indices: List[int]) -> RNNStates:
    def _gather(state_tuple: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.cat([t[:, i : i + 1, :] for i in indices], dim=1) for t in state_tuple)

    return RNNStates(_gather(lstm_states.pi), _gather(lstm_states.vf))


def _scatter_rnn_columns(
    batched_lstm: RNNStates, indices: List[int], sub_lstm: RNNStates
) -> RNNStates:
    updated = RNNStates(
        tuple(t.clone() for t in batched_lstm.pi),
        tuple(t.clone() for t in batched_lstm.vf),
    )
    for j, env_idx in enumerate(indices):
        col = select_rnn_state_column(sub_lstm, j)
        updated = _write_env_lstm_column(updated, env_idx, col)
    return updated
