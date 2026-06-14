import os
import numpy as np
import torch
import torch as th
import json
from typing import Dict, Generator, Optional, Union, Any, Tuple, NamedTuple
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from loguru import logger

from src.agent.maskable_recurrent_ppo import MaskableRecurrentRolloutBufferSamples, RNNStates


def build_recurrent_sequence_spans(
    n_envs: int,
    buffer_size: int,
    episode_starts: np.ndarray,
    max_seq_len: int = 32,
    policy_active: Optional[np.ndarray] = None,
) -> list[tuple[int, int, int]]:
    """
    RNN minibatch sequence boundaries per env column.

    Splits on focal-unit / episode resets (``episode_starts``), skips timesteps where
    ``policy_active`` is 0 (another env owned that buffer row), and caps length at
    ``max_seq_len``. Inactive rows must not sit inside a sequence or the RNN hidden
    state drifts from what collection used when computing ``old_log_prob``.
    """
    spans: list[tuple[int, int, int]] = []
    for env_idx in range(n_envs):
        t = 0
        while t < buffer_size:
            while t < buffer_size:
                if policy_active is None or policy_active[t, env_idx] > 1e-8:
                    break
                t += 1
            if t >= buffer_size:
                break
            start = t
            length = 0
            while t < buffer_size and length < max_seq_len:
                if policy_active is not None and policy_active[t, env_idx] <= 1e-8:
                    break
                if t > start and episode_starts[t, env_idx]:
                    break
                length += 1
                t += 1
            if length > 0:
                spans.append((env_idx, start, length))
    return spans


class FlatMaskableRolloutBufferSamples(NamedTuple):
    """Minibatch for feed-forward MaskablePPO (no RNN state)."""

    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class DiskDictRolloutBuffer(DictRolloutBuffer):
    """
    Recurrent DictRolloutBuffer with optional memmap offload for map/tactical tensors.

    ``storage_backend="disk"`` (default): large observation channels use memmap files.
    ``storage_backend="ram"``: all observations stay in RAM (same add/get API).
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        temp_dir: str = "C:/temp/sc2_buffer_cache",
        rnn_type: str = "gru",
        hidden_size: int = 512,
        storage_backend: str = "disk",
        enable_rnn_sampling: bool = True,
    ):
        super(DiskDictRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

        self.enable_rnn_sampling = bool(enable_rnn_sampling)

        self.storage_backend = str(storage_backend).strip().lower()
        if self.storage_backend not in ("disk", "ram"):
            raise ValueError(
                f"storage_backend must be 'disk' or 'ram', got {storage_backend!r}"
            )

        self.temp_dir = temp_dir
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.instance_id = f"{os.getpid()}_{id(self)}"
        self.instance_dir = os.path.join(self.temp_dir, self.instance_id)
        if self.storage_backend == "disk":
            os.makedirs(self.instance_dir, exist_ok=True)

        logger.info(
            f"DictRolloutBuffer ({self.storage_backend}) initialized: "
            f"{self.rnn_type.upper()}, Hidden: {self.hidden_size}"
        )

        # Recurrent state storage (buffer_size, n_layers, n_envs, hidden_dim)
        self.hidden_states_pi = np.zeros((self.buffer_size, 1, self.n_envs, self.hidden_size), dtype=np.float32)
        self.hidden_states_vf = np.zeros((self.buffer_size, 1, self.n_envs, self.hidden_size), dtype=np.float32)
        if self.rnn_type == "lstm":
            self.cell_states_pi = np.zeros((self.buffer_size, 1, self.n_envs, self.hidden_size), dtype=np.float32)
            self.cell_states_vf = np.zeros((self.buffer_size, 1, self.n_envs, self.hidden_size), dtype=np.float32)

        # Action masks
        if isinstance(self.action_space, spaces.MultiDiscrete):
            m_dim = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.Discrete):
            m_dim = self.action_space.n
        else:
            m_dim = 1
        self.action_masks = np.ones((self.buffer_size, self.n_envs, m_dim), dtype=bool)
        self.policy_active = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Offload large observations to memmap files (disk backend only).
        self.disk_files = {}
        if self.storage_backend == "disk":
            for key, subspace in observation_space.spaces.items():
                if "map" in key or "tactical" in key:
                    filename = os.path.join(self.instance_dir, f"obs_{key}.bin")
                    shape = (self.buffer_size, self.n_envs, *subspace.shape)
                    self.observations[key] = np.memmap(
                        filename, dtype=np.float32, mode="w+", shape=shape
                    )
                    self.disk_files[key] = filename

    def reset(self) -> None:
        """
        Resets the buffer while preserving the underlying np.memmap objects
        for large observation arrays.
        """
        super().reset()
        # Restore memmaps that super().reset() overwrote with np.zeros
        if hasattr(self, 'disk_files'):
            for key, filename in self.disk_files.items():
                if os.path.exists(filename):
                    subspace = self.observation_space.spaces[key]
                    shape = (self.buffer_size, self.n_envs, *subspace.shape)
                    self.observations[key] = np.memmap(filename, dtype=np.float32, mode='r+', shape=shape)
        if hasattr(self, "policy_active"):
            self.policy_active.fill(0.0)

    def to_device(self, device: torch.device):
        self.device = device
        return self

    def close_memmaps(self):
        """Explicitly close memmaps to release Windows file handles and standby memory."""
        if hasattr(self, 'observations'):
            for key in list(self.observations.keys()):
                obs = self.observations[key]
                if isinstance(obs, np.memmap):
                    try:
                        obs._mmap.close()
                    except Exception as e:
                        logger.warning(f"Failed to close memmap {key}: {e}")

    def clear_vram_cache(self):
        """No-op for disk buffer as we don't cache in VRAM."""
        pass

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        lstm_states: RNNStates = None,
        action_masks: Optional[np.ndarray] = None,
        policy_active: Optional[np.ndarray] = None,
    ) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.copy()
        if policy_active is not None:
            self.policy_active[self.pos] = np.asarray(policy_active, dtype=np.float32).reshape(self.n_envs)
        else:
            self.policy_active[self.pos] = 1.0

        if lstm_states is not None:
            # Robustly normalize incoming RNN states to (1, n_envs, hidden_size).
            # In async collectors, state tensors may occasionally come in as single-env
            # tensors (e.g. (1, 1, hidden) or (hidden,)); broadcast when needed.
            self.hidden_states_pi[self.pos] = self._normalize_rnn_state(lstm_states.pi[0], "pi_h")
            self.hidden_states_vf[self.pos] = self._normalize_rnn_state(lstm_states.vf[0], "vf_h")
            if self.rnn_type == "lstm" and len(lstm_states.pi) > 1:
                self.cell_states_pi[self.pos] = self._normalize_rnn_state(lstm_states.pi[1], "pi_c")
                self.cell_states_vf[self.pos] = self._normalize_rnn_state(lstm_states.vf[1], "vf_c")

        super().add(obs, action, reward, episode_start, value, log_prob)

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        """
        GAE that skips timesteps where ``policy_active[step, env] == 0``.

        Async squad collection writes one buffer row per env per game step for
        multi-unit squads, and may leave inactive env slots zeroed when another
        env owns the row. Standard SB3 GAE would bootstrap through those zeros and
        corrupt advantages / returns (low explained variance, stalled learning).
        """
        last_values = last_values.clone().cpu().numpy().flatten()
        filled = int(self.pos) if self.pos > 0 else self.buffer_size
        pa = getattr(self, "policy_active", None)

        self.advantages[:filled] = 0.0
        self.returns[:filled] = 0.0

        for env_idx in range(self.n_envs):
            active_ts = [
                t
                for t in range(filled)
                if pa is None or float(pa[t, env_idx]) > 1e-8
            ]
            if not active_ts:
                continue

            last_gae_lam = 0.0
            for i in reversed(range(len(active_ts))):
                t = active_ts[i]
                if i == len(active_ts) - 1:
                    next_non_terminal = 1.0 - float(dones[env_idx])
                    next_value = float(last_values[env_idx])
                else:
                    t_next = active_ts[i + 1]
                    next_non_terminal = 1.0 - float(self.episode_starts[t_next, env_idx])
                    next_value = float(self.values[t_next, env_idx])

                delta = (
                    float(self.rewards[t, env_idx])
                    + self.gamma * next_value * next_non_terminal
                    - float(self.values[t, env_idx])
                )
                last_gae_lam = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                )
                self.advantages[t, env_idx] = last_gae_lam

            for t in active_ts:
                self.returns[t, env_idx] = (
                    self.advantages[t, env_idx] + self.values[t, env_idx]
                )

    def _normalize_rnn_state(self, state_tensor: torch.Tensor, name: str) -> np.ndarray:
        """
        Normalize an RNN state tensor to shape (1, n_envs, hidden_size).
        Accepts common variants from recurrent SB3 policies and async workers.
        """
        arr = state_tensor.detach().cpu().numpy()

        # Expected format from SB3 recurrent policies.
        if arr.ndim == 3:
            # (n_layers, n_envs, hidden)
            if arr.shape[0] != 1:
                arr = arr[:1]
            if arr.shape[2] != self.hidden_size:
                arr = arr.reshape(1, -1, self.hidden_size)
            if arr.shape[1] == self.n_envs:
                return arr.astype(np.float32, copy=False)
            if arr.shape[1] == 1 and self.n_envs > 1:
                return np.repeat(arr, self.n_envs, axis=1).astype(np.float32, copy=False)

        # (n_envs, hidden) or (1, hidden)
        if arr.ndim == 2 and arr.shape[1] == self.hidden_size:
            if arr.shape[0] == self.n_envs:
                return arr[None, :, :].astype(np.float32, copy=False)
            if arr.shape[0] == 1:
                return np.repeat(arr[None, :, :], self.n_envs, axis=1).astype(np.float32, copy=False)

        # (hidden,) single state
        if arr.ndim == 1 and arr.shape[0] == self.hidden_size:
            base = arr.reshape(1, 1, self.hidden_size)
            return np.repeat(base, self.n_envs, axis=1).astype(np.float32, copy=False)

        # Last resort: try to coerce by total size.
        try:
            coerced = arr.reshape(1, -1, self.hidden_size)
            if coerced.shape[1] == 1 and self.n_envs > 1:
                coerced = np.repeat(coerced, self.n_envs, axis=1)
            elif coerced.shape[1] != self.n_envs:
                raise ValueError(f"{name}: invalid env dim {coerced.shape[1]}, expected {self.n_envs}")
            return coerced.astype(np.float32, copy=False)
        except Exception as exc:
            raise ValueError(
                f"Failed to normalize RNN state '{name}': got shape {arr.shape}, "
                f"expected compatible with (1, {self.n_envs}, {self.hidden_size})"
            ) from exc

    def get(self, batch_size: Optional[int] = None) -> Generator[MaskableRecurrentRolloutBufferSamples, None, None]:
        """
        Yield minibatches. Recurrent policies use padded RNN sequences; feed-forward
        MaskablePPO uses flat shuffled indices (lower GPU memory, no BPTT padding).
        """
        if not self.enable_rnn_sampling:
            yield from self._get_flat(batch_size)
            return

        assert self.full, "Buffer must be full"

        max_seq_len = 32
        filled_steps = int(self.pos) if self.pos > 0 else self.buffer_size
        sequence_spans = build_recurrent_sequence_spans(
            self.n_envs,
            filled_steps,
            self.episode_starts,
            max_seq_len,
            policy_active=getattr(self, "policy_active", None),
        )
        np.random.shuffle(sequence_spans)

        if not sequence_spans:
            return

        if batch_size:
            batches: list[list[tuple[int, int, int]]] = []
            current: list[tuple[int, int, int]] = []
            current_steps = 0
            for span in sequence_spans:
                span_len = span[2]
                if current and current_steps + span_len > batch_size:
                    batches.append(current)
                    current = []
                    current_steps = 0
                current.append(span)
                current_steps += span_len
            if current:
                batches.append(current)
        else:
            batches = [sequence_spans]

        for batch_spans in batches:
            yield self._get_samples_by_sequences(batch_spans)

    def _active_flat_indices(self, filled: int) -> np.ndarray:
        pa = getattr(self, "policy_active", None)
        if pa is None:
            return np.arange(filled * self.n_envs, dtype=np.int64)
        active = pa[:filled] > 1e-8
        steps, envs = np.nonzero(active)
        return (steps.astype(np.int64) * self.n_envs + envs.astype(np.int64))

    def _get_flat(self, batch_size: Optional[int] = None) -> Generator[FlatMaskableRolloutBufferSamples, None, None]:
        assert self.full, "Buffer must be full"
        filled = int(self.pos) if self.pos > 0 else self.buffer_size
        indices = self._active_flat_indices(filled)
        np.random.shuffle(indices)
        if batch_size is None:
            batch_size = max(len(indices), 1)
        step_idx = indices // self.n_envs
        env_idx = indices % self.n_envs

        for start in range(0, len(indices), batch_size):
            sl = slice(start, start + batch_size)
            b_steps = step_idx[sl]
            b_envs = env_idx[sl]
            obs = {
                key: th.as_tensor(arr[b_steps, b_envs], device=self.device)
                for key, arr in self.observations.items()
            }
            yield FlatMaskableRolloutBufferSamples(
                observations=obs,
                actions=th.as_tensor(self.actions[b_steps, b_envs], device=self.device),
                old_values=th.as_tensor(
                    self.values[b_steps, b_envs].flatten(), device=self.device
                ),
                old_log_prob=th.as_tensor(
                    self.log_probs[b_steps, b_envs].flatten(), device=self.device
                ),
                advantages=th.as_tensor(
                    self.advantages[b_steps, b_envs].flatten(), device=self.device
                ),
                returns=th.as_tensor(
                    self.returns[b_steps, b_envs].flatten(), device=self.device
                ),
                action_masks=th.as_tensor(
                    self.action_masks[b_steps, b_envs], device=self.device
                ),
            )

    def _get_samples_by_sequences(
        self, batch_spans: list[tuple[int, int, int]]
    ) -> MaskableRecurrentRolloutBufferSamples:
        """Collect and pad a batch of variable-length RNN sequences."""
        seq_len = max(length for _, _, length in batch_spans)

        obs_batch = {k: [] for k in self.observation_space.keys()}
        act_batch, val_batch, logp_batch, adv_batch, ret_batch, start_batch, mask_batch = [], [], [], [], [], [], []
        h_pi_starts, h_vf_starts = [], []
        c_pi_starts, c_vf_starts = [], []

        for env_idx, start_step, actual_len in batch_spans:
            end_step = start_step + actual_len

            for k in obs_batch.keys():
                chunk = self.observations[k][start_step:end_step, env_idx]
                if actual_len < seq_len:
                    pad_shape = (seq_len - actual_len, *chunk.shape[1:])
                    chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=np.float32)])
                obs_batch[k].append(chunk)

            h_pi_starts.append(self.hidden_states_pi[start_step, :, env_idx, :])
            h_vf_starts.append(self.hidden_states_vf[start_step, :, env_idx, :])
            if self.rnn_type == "lstm":
                c_pi_starts.append(self.cell_states_pi[start_step, :, env_idx, :])
                c_vf_starts.append(self.cell_states_vf[start_step, :, env_idx, :])

            def get_and_pad(arr, is_2d=False):
                chunk = arr[start_step:end_step, env_idx]
                if actual_len < seq_len:
                    pad_shape = (seq_len - actual_len, *chunk.shape[1:]) if is_2d else (seq_len - actual_len,)
                    chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=arr.dtype)])
                return chunk

            act_batch.append(get_and_pad(self.actions, True))
            val_batch.append(get_and_pad(self.values))
            logp_batch.append(get_and_pad(self.log_probs))
            adv_batch.append(get_and_pad(self.advantages))
            ret_batch.append(get_and_pad(self.returns))
            start_batch.append(get_and_pad(self.episode_starts))

            m = np.ones(seq_len, dtype=np.float32)
            m[actual_len:] = 0.0
            pa = get_and_pad(self.policy_active)
            mask_batch.append(m * pa)

        def _np_stack_to_device(arr_list, flatten=True):
            res = th.as_tensor(np.stack(arr_list))
            res = res.to(self.device, non_blocking=True)
            return res.reshape(-1, *res.shape[2:]) if flatten else res

        def to_t(list_of_arr, flatten=True):
            return _np_stack_to_device(list_of_arr, flatten=flatten)

        h_pi = _np_stack_to_device(h_pi_starts, flatten=False).transpose(0, 1).contiguous()
        h_vf = _np_stack_to_device(h_vf_starts, flatten=False).transpose(0, 1).contiguous()
        if self.rnn_type == "lstm":
            c_pi = _np_stack_to_device(c_pi_starts, flatten=False).transpose(0, 1).contiguous()
            c_vf = _np_stack_to_device(c_vf_starts, flatten=False).transpose(0, 1).contiguous()
            lstm_states = RNNStates((h_pi, c_pi), (h_vf, c_vf))
        else:
            lstm_states = RNNStates((h_pi,), (h_vf,))

        m_list = []
        for env_idx, start_step, actual_len in batch_spans:
            end_step = start_step + actual_len
            chunk = self.action_masks[start_step:end_step, env_idx]
            if actual_len < seq_len:
                pad = np.ones((seq_len - actual_len, chunk.shape[-1]), dtype=bool)
                chunk = np.concatenate([chunk, pad])
            m_list.append(chunk)
        batch_action_masks = to_t(m_list)

        return MaskableRecurrentRolloutBufferSamples(
            observations={k: to_t(v) for k, v in obs_batch.items()},
            actions=to_t(act_batch),
            old_values=to_t(val_batch),
            old_log_prob=to_t(logp_batch),
            advantages=to_t(adv_batch),
            returns=to_t(ret_batch),
            lstm_states=lstm_states,
            episode_starts=to_t(start_batch),
            mask=to_t(mask_batch),
            action_masks=batch_action_masks,
        )

    def __del__(self):
        self.close_memmaps()
        if getattr(self, "storage_backend", "disk") != "disk":
            return
        if hasattr(self, "disk_files"):
            for key, filename in self.disk_files.items():
                if os.path.exists(filename):
                    try:
                        del self.observations[key]
                        os.remove(filename)
                    except Exception:
                        pass
            if hasattr(self, "instance_dir") and os.path.exists(self.instance_dir):
                try:
                    os.rmdir(self.instance_dir)
                except Exception:
                    pass
