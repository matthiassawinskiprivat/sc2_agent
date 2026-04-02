import os
import numpy as np
import torch
import torch as th
import json
from typing import Dict, Generator, Optional, Union, Any, Tuple
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from loguru import logger

from src.agent.maskable_recurrent_ppo import MaskableRecurrentRolloutBufferSamples, RNNStates

class DiskDictRolloutBuffer(DictRolloutBuffer):
    """
    A sequence-aware DictRolloutBuffer that stores large arrays in memory-mapped files.
    Optimized for recurrent policies and low RAM usage.
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
        hidden_size: int = 512
    ):
        super(DiskDictRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        
        self.temp_dir = temp_dir
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.instance_id = f"{os.getpid()}_{id(self)}"
        self.instance_dir = os.path.join(self.temp_dir, self.instance_id)
        os.makedirs(self.instance_dir, exist_ok=True)
        
        logger.info(f"DiskDictRolloutBuffer initialized: {self.rnn_type.upper()}, Hidden: {self.hidden_size}")

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

        # Offload large observations to disk
        self.disk_files = {}
        for key, subspace in observation_space.spaces.items():
            if "map" in key or "tactical" in key:
                filename = os.path.join(self.instance_dir, f"obs_{key}.bin")
                shape = (self.buffer_size, self.n_envs, *subspace.shape)
                self.observations[key] = np.memmap(filename, dtype=np.float32, mode='w+', shape=shape)
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
    ) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.copy()
        
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
        OVERRIDE: Instead of random steps, we yield SEQUENCES.
        This is critical for Recurrent policies.
        """
        assert self.full, "Buffer must be full"
        
        # Determine sequence boundaries
        # We force sequences to be at most 32 steps long
        max_seq_len = 32
        
        # Create a list of all sequence start points (env_idx, start_step)
        sequence_starts = []
        for env_idx in range(self.n_envs):
            for start_step in range(0, self.buffer_size, max_seq_len):
                sequence_starts.append((env_idx, start_step))
        
        # Shuffle sequences
        np.random.shuffle(sequence_starts)
        
        # Group into mini-batches of sequences
        # batch_size here refers to total steps, so num_sequences = batch_size // max_seq_len
        num_seq_per_batch = max(1, batch_size // max_seq_len) if batch_size else len(sequence_starts)
        
        for i in range(0, len(sequence_starts), num_seq_per_batch):
            batch_seqs = sequence_starts[i : i + num_seq_per_batch]
            yield self._get_samples_by_sequences(batch_seqs, max_seq_len)

    def _get_samples_by_sequences(self, batch_seqs, seq_len) -> MaskableRecurrentRolloutBufferSamples:
        """Collects and pads a batch of sequences."""
        actual_batch_size = len(batch_seqs) * seq_len
        
        # Prepare storage
        obs_batch = {k: [] for k in self.observation_space.keys()}
        act_batch, val_batch, logp_batch, adv_batch, ret_batch, start_batch, mask_batch = [], [], [], [], [], [], []
        
        # Hidden states at the START of each sequence
        h_pi_starts, h_vf_starts = [], []
        c_pi_starts, c_vf_starts = [], []

        for env_idx, start_step in batch_seqs:
            end_step = min(start_step + seq_len, self.buffer_size)
            actual_len = end_step - start_step
            
            # Observations
            for k in obs_batch.keys():
                # We pull from memmap directly. This is the only place it hits RAM.
                chunk = self.observations[k][start_step:end_step, env_idx]
                if actual_len < seq_len:
                    # Pad if needed
                    pad_shape = (seq_len - actual_len, *chunk.shape[1:])
                    chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=np.float32)])
                obs_batch[k].append(chunk)
            
            # Hidden states (pick the one at start_step)
            h_pi_starts.append(self.hidden_states_pi[start_step, :, env_idx, :])
            h_vf_starts.append(self.hidden_states_vf[start_step, :, env_idx, :])
            if self.rnn_type == "lstm":
                c_pi_starts.append(self.cell_states_pi[start_step, :, env_idx, :])
                c_vf_starts.append(self.cell_states_vf[start_step, :, env_idx, :])

            # Others (Actions, rewards, etc.)
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
            
            # Mask for valid steps (handle padding)
            m = np.ones(seq_len, dtype=np.float32)
            m[actual_len:] = 0.0
            mask_batch.append(m)

        # Convert to Tensors
        def _np_stack_to_device(arr_list, flatten=True):
            res = th.as_tensor(np.stack(arr_list))
            res = res.to(self.device, non_blocking=True)
            return res.reshape(-1, *res.shape[2:]) if flatten else res

        def to_t(list_of_arr, flatten=True):
            return _np_stack_to_device(list_of_arr, flatten=flatten)

        # RNN States: (n_layers, n_seq, dim)
        h_pi = _np_stack_to_device(h_pi_starts, flatten=False).transpose(0, 1).contiguous()
        h_vf = _np_stack_to_device(h_vf_starts, flatten=False).transpose(0, 1).contiguous()
        if self.rnn_type == "lstm":
            c_pi = _np_stack_to_device(c_pi_starts, flatten=False).transpose(0, 1).contiguous()
            c_vf = _np_stack_to_device(c_vf_starts, flatten=False).transpose(0, 1).contiguous()
            lstm_states = RNNStates((h_pi, c_pi), (h_vf, c_vf))
        else:
            lstm_states = RNNStates(h_pi, h_vf)

        # Action masks (also need sequence handling)
        m_list = []
        for env_idx, start_step in batch_seqs:
            end_step = min(start_step + seq_len, self.buffer_size)
            chunk = self.action_masks[start_step:end_step, env_idx]
            if (end_step - start_step) < seq_len:
                pad = np.ones((seq_len - (end_step - start_step), chunk.shape[-1]), dtype=bool)
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
            action_masks=batch_action_masks
        )

    def __del__(self):
        self.close_memmaps()
        if hasattr(self, 'disk_files'):
            for key, filename in self.disk_files.items():
                if os.path.exists(filename):
                    try:
                        del self.observations[key]
                        os.remove(filename)
                    except: pass
            if hasattr(self, 'instance_dir') and os.path.exists(self.instance_dir):
                try: os.rmdir(self.instance_dir)
                except: pass
