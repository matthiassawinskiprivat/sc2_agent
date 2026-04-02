import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union, NamedTuple, ClassVar
from copy import deepcopy
from loguru import logger

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, safe_mean, polyak_update, zip_strict
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)

from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from sb3_contrib.common.maskable.distributions import MaskableDistribution, MaskableMultiCategoricalDistribution, MaskableBernoulliDistribution, MaskableCategoricalDistribution
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers
from sb3_contrib.common.recurrent.type_aliases import RNNStates

def tensor_safe_pad(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: torch.device,
    tensor: Union[np.ndarray, torch.Tensor],
    padding_value: float = 0.0,
) -> torch.Tensor:
    seq = []
    # Convert to tensor once if needed, then slice
    base_tensor = torch.as_tensor(tensor, device=device) if not isinstance(tensor, torch.Tensor) else tensor.to(device)
    for start, end in zip(seq_start_indices, seq_end_indices):
        seq.append(base_tensor[start : end + 1])
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)

def tensor_safe_pad_and_flatten(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: torch.device,
    tensor: Union[np.ndarray, torch.Tensor],
    padding_value: float = 0.0,
) -> torch.Tensor:
    # Use standard flatten(0, 1) to preserve batch dimension for mask comparison if needed, 
    # but here we follow the existing behavior of a full flat return.
    return tensor_safe_pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).reshape(-1)

def create_tensor_safe_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Any, Any]:
    es = episode_starts.flatten()
    ec = env_change.flatten()
    # Sequence starts at episode boundary or environment change
    seq_start = np.logical_or(es, ec)
    seq_start[0] = True
    
    # FOR MEMORY SAFETY: Also force a split every 32 steps to limit BPTT depth
    # This is the single most important fix for the 10GB memory error
    max_seq_len = 32
    counter = 0
    for i in range(len(seq_start)):
        if seq_start[i]:
            counter = 0
        counter += 1
        if counter >= max_seq_len:
            if i + 1 < len(seq_start):
                seq_start[i+1] = True
            counter = 0

    seq_start_indices = np.where(seq_start == True)[0]
    seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(es)-1])])      

    from functools import partial
    local_pad = partial(tensor_safe_pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(tensor_safe_pad_and_flatten, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, local_pad, local_pad_and_flatten

class MaskableRecurrentRolloutBufferSamples(NamedTuple):
    observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    lstm_states: RNNStates
    episode_starts: torch.Tensor
    mask: torch.Tensor
    action_masks: torch.Tensor

class MaskableRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    def __init__(self, *args, **kwargs):
        self.rnn_type = kwargs.pop("rnn_type", "lstm")
        super().__init__(*args, **kwargs)
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            action_dim = sum(self.action_space.nvec)
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
            action_dim = self.action_space.n
        else:
             raise NotImplementedError("Action space not supported")
        self.action_masks = np.ones((self.buffer_size, self.n_envs, action_dim), dtype=bool)
        self.vram_cache: Dict[str, torch.Tensor] = {}

    def reset(self):
        super().reset()
        if hasattr(self, "action_masks"): self.action_masks.fill(True)
        self.vram_cache = {}

    def clear_vram_cache(self):
        self.vram_cache = {}

    def to_device(self, device: torch.device):
        self.vram_cache = {}
        def swap_and_flatten_gpu(arr):
            # Same as SB3: (buffer_size, n_envs, ...) -> (n_envs, buffer_size, ...) -> (n_envs * buffer_size, ...)
            res = arr.swapaxes(0, 1)
            shape = res.shape
            # Zero-copy conversion where possible
            return torch.as_tensor(res.reshape((shape[0] * shape[1], *shape[2:])), device=device)
            
        self.vram_cache["observations"] = swap_and_flatten_gpu(self.observations)
        self.vram_cache["actions"] = swap_and_flatten_gpu(self.actions)
        self.vram_cache["values"] = swap_and_flatten_gpu(self.values)
        self.vram_cache["log_probs"] = swap_and_flatten_gpu(self.log_probs)
        self.vram_cache["advantages"] = swap_and_flatten_gpu(self.advantages)
        self.vram_cache["returns"] = swap_and_flatten_gpu(self.returns)
        self.vram_cache["episode_starts"] = swap_and_flatten_gpu(self.episode_starts)
        self.vram_cache["action_masks"] = swap_and_flatten_gpu(self.action_masks)
        
        h_pi = self.hidden_states_pi.swapaxes(1, 2)
        h_vf = self.hidden_states_vf.swapaxes(1, 2)
        self.vram_cache["hidden_states_pi"] = swap_and_flatten_gpu(h_pi)
        self.vram_cache["hidden_states_vf"] = swap_and_flatten_gpu(h_vf)
        if hasattr(self, "cell_states_pi"):
            c_pi = self.cell_states_pi.swapaxes(1, 2)
            c_vf = self.cell_states_vf.swapaxes(1, 2)
            self.vram_cache["cell_states_pi"] = swap_and_flatten_gpu(c_pi)
            self.vram_cache["cell_states_vf"] = swap_and_flatten_gpu(c_vf)

    def add(self, obs, action, reward, episode_start, value, log_prob, lstm_states, action_masks=None):
        if action_masks is not None: self.action_masks[self.pos] = action_masks.copy()
        self.hidden_states_pi[self.pos] = lstm_states.pi[0].detach().cpu().numpy()
        self.hidden_states_vf[self.pos] = lstm_states.vf[0].detach().cpu().numpy()
        if len(lstm_states.pi) > 1:
            if not hasattr(self, "cell_states_pi"):
                self.cell_states_pi = np.zeros_like(self.hidden_states_pi)
                self.cell_states_vf = np.zeros_like(self.hidden_states_vf)
            self.cell_states_pi[self.pos] = lstm_states.pi[1].detach().cpu().numpy()
            self.cell_states_vf[self.pos] = lstm_states.vf[1].detach().cpu().numpy()
        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action).reshape((self.n_envs, self.action_dim))
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size: self.full = True

    def get(self, batch_size: Optional[int] = None):
        assert self.full, "Buffer must be full before sampling"
        
        # 1. Prepare all sequences (chunks) across all environments
        env_change = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        
        # We need to know where sequences start to batch them
        # create_tensor_safe_sequencers returns indices relative to a flattened (n_envs * buffer_size) array
        # where the environment dimension is swapped to the front.
        starts = self.episode_starts.swapaxes(0, 1).flatten()
        ec = env_change.swapaxes(0, 1).flatten()
        
        seq_start_indices, _, _ = create_tensor_safe_sequencers(starts, ec, self.device)
        n_sequences = len(seq_start_indices)
        
        # 2. Shuffle sequence indices
        indices = np.random.permutation(n_sequences)
        
        # 3. Yield mini-batches of sequences
        # Here batch_size refers to number of sequences (e.g. 8 or 16)
        actual_batch_size = batch_size if batch_size else n_sequences
        
        for start_ind in range(0, n_sequences, actual_batch_size):
            batch_seq_indices = seq_start_indices[indices[start_ind : start_ind + actual_batch_size]]
            yield self._get_samples_by_seq(batch_seq_indices)

    def _get_samples_by_seq(self, seq_start_indices):
        # seq_start_indices are indices in the flattened (n_envs * buffer_size) array
        # Each sequence has a fixed length (except potentially at the end of an env)
        # We need to collect the full window for each sequence.
        
        max_seq_len = 32 # Must match create_tensor_safe_sequencers
        all_indices = []
        for start in seq_start_indices:
            # We take up to max_seq_len steps, but must not cross environment boundaries
            # Environment i occupies [i*B, (i+1)*B]
            env_idx = start // self.buffer_size
            env_end = (env_idx + 1) * self.buffer_size
            length = min(max_seq_len, env_end - start)
            all_indices.extend(range(start, start + length))
        
        all_indices = np.array(all_indices)
        device = self.device

        # Create local sequencers for this specific mini-batch
        # These indices are now relative to the mini-batch's local flattened array
        local_starts = self.vram_cache["episode_starts"][all_indices]
        
        # Sequence starts are at the beginning of each chunk we just picked
        local_seq_starts = np.zeros(len(all_indices), dtype=bool)
        current = 0
        for start in seq_start_indices:
            env_idx = start // self.buffer_size
            env_end = (env_idx + 1) * self.buffer_size
            length = min(max_seq_len, env_end - start)
            local_seq_starts[current] = True
            current += length

        # Helper for padding the mini-batch (O(n_seq) — do not recompute np.where inside inner loop)
        boundary = np.where(local_seq_starts)[0]

        def local_pad(tensor_key):
            tensor = self.vram_cache[tensor_key][all_indices]
            seqs = []
            for i, start_idx in enumerate(boundary):
                end_idx = boundary[i + 1] if i + 1 < len(boundary) else len(all_indices)
                seqs.append(tensor[start_idx:end_idx])
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)

        padded_obs = local_pad("observations")
        padded_act = local_pad("actions")
        n_seq_batch = len(seq_start_indices)
        max_len_batch = padded_obs.shape[1]
        padded_batch_size = n_seq_batch * max_len_batch

        def get_states_gpu(h_key, c_key=None):
            h = self.vram_cache[h_key][all_indices[local_seq_starts]].swapaxes(0, 1)
            if c_key in self.vram_cache:
                c = self.vram_cache[c_key][all_indices[local_seq_starts]].swapaxes(0, 1)
                return (h.contiguous(), c.contiguous())
            return (h.contiguous(),)

        return MaskableRecurrentRolloutBufferSamples(
            observations=padded_obs.reshape((padded_batch_size, *self.obs_shape)),
            actions=padded_act.reshape((padded_batch_size,) + self.actions.shape[2:]),
            old_values=local_pad("values").reshape(-1),
            old_log_prob=local_pad("log_probs").reshape(-1),
            advantages=local_pad("advantages").reshape(-1),
            returns=local_pad("returns").reshape(-1),
            lstm_states=RNNStates(get_states_gpu("hidden_states_pi", "cell_states_pi"), get_states_gpu("hidden_states_vf", "cell_states_vf")),
            episode_starts=local_pad("episode_starts").reshape(-1),
            mask=torch.ones(padded_batch_size, device=device), # Simplified mask for the mini-batch
            action_masks=local_pad("action_masks").reshape(padded_batch_size, -1))

    def _get_samples(self, batch_inds, env_change, env=None):
        if not self.vram_cache: return self._get_samples_standard(batch_inds, env_change, env)
        
        # In flattened VRAM cache, Env i occupies range [i*B, (i+1)*B]
        all_indices = []
        for env_idx in batch_inds:
            all_indices.extend(range(env_idx * self.buffer_size, (env_idx + 1) * self.buffer_size))
        all_indices = np.array(all_indices)
        
        starts = self.vram_cache["episode_starts"][all_indices]
        ec_batch = torch.as_tensor(env_change[:, batch_inds].swapaxes(0, 1).flatten(), device=self.device)
        
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_tensor_safe_sequencers(
            starts.cpu().numpy(), ec_batch.cpu().numpy(), self.device)
            
        n_seq = len(self.seq_start_indices)
        padded_batch_size = n_seq * self.pad(self.vram_cache["actions"][all_indices]).shape[1]
        
        def get_states_gpu(h_key, c_key=None):
            # hidden: (n_envs * buffer_size, n_layers, dim) -> pick first of each seq
            h = self.vram_cache[h_key][all_indices][self.seq_start_indices].swapaxes(0, 1)
            if c_key in self.vram_cache:
                c = self.vram_cache[c_key][all_indices][self.seq_start_indices].swapaxes(0, 1)
                return (h.contiguous(), c.contiguous())
            return (h.contiguous(),)

        return MaskableRecurrentRolloutBufferSamples(
            observations=self.pad(self.vram_cache["observations"][all_indices]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.vram_cache["actions"][all_indices]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.vram_cache["values"][all_indices]),
            old_log_prob=self.pad_and_flatten(self.vram_cache["log_probs"][all_indices]),
            advantages=self.pad_and_flatten(self.vram_cache["advantages"][all_indices]),
            returns=self.pad_and_flatten(self.vram_cache["returns"][all_indices]),
            lstm_states=RNNStates(get_states_gpu("hidden_states_pi", "cell_states_pi"), get_states_gpu("hidden_states_vf", "cell_states_vf")),
            episode_starts=self.pad_and_flatten(self.vram_cache["episode_starts"][all_indices]),
            mask=self.pad_and_flatten(torch.ones_like(self.vram_cache["returns"][all_indices])),
            action_masks=self.pad_and_flatten(self.vram_cache["action_masks"][all_indices]))

    def _get_samples_standard(self, batch_inds, env_change, env=None):
        # swapaxes(0, 1) to put env dimension first, then flatten to get full env trajectories
        def flat(arr):
            res = arr[:, batch_inds].swapaxes(0, 1)
            return res.reshape(-1, *res.shape[2:])
            
        starts = flat(self.episode_starts)
        ec_batch = env_change[:, batch_inds].swapaxes(0, 1).flatten()
        
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(starts, ec_batch, self.device)
        n_seq = len(self.seq_start_indices)
        
        obs_batch = flat(self.observations)
        act_batch = flat(self.actions)
        padded_batch_size = n_seq * self.pad(torch.as_tensor(act_batch, device=self.device)).shape[1]
        
        def get_states(h_s, c_s):
            # h_s: (buffer_size, n_layers, n_envs, dim) -> (n_envs, buffer_size, n_layers, dim)
            h = h_s.swapaxes(0, 2).swapaxes(1, 2)[batch_inds].reshape(-1, h_s.shape[1], h_s.shape[3])[self.seq_start_indices].swapaxes(0, 1)
            c = c_s.swapaxes(0, 2).swapaxes(1, 2)[batch_inds].reshape(-1, c_s.shape[1], c_s.shape[3])[self.seq_start_indices].swapaxes(0, 1)
            return (torch.as_tensor(h, device=self.device).contiguous(), torch.as_tensor(c, device=self.device).contiguous())

        return MaskableRecurrentRolloutBufferSamples(
            observations=self.pad(torch.as_tensor(obs_batch, device=self.device)).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(torch.as_tensor(act_batch, device=self.device)).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(torch.as_tensor(flat(self.values), device=self.device)),
            old_log_prob=self.pad_and_flatten(torch.as_tensor(flat(self.log_probs), device=self.device)),
            advantages=self.pad_and_flatten(torch.as_tensor(flat(self.advantages), device=self.device)),
            returns=self.pad_and_flatten(torch.as_tensor(flat(self.returns), device=self.device)),
            lstm_states=RNNStates(get_states(self.hidden_states_pi, self.cell_states_pi), get_states(self.hidden_states_vf, self.cell_states_vf)),
            episode_starts=self.pad_and_flatten(torch.as_tensor(starts, device=self.device)),
            mask=self.pad_and_flatten(torch.ones_like(torch.as_tensor(starts, device=self.device))),
            action_masks=self.pad_and_flatten(torch.as_tensor(flat(self.action_masks), device=self.device)))

class MaskableRecurrentDictRolloutBuffer(MaskableRecurrentRolloutBuffer):
    def __init__(self, *args, **kwargs):
        observation_space = args[1] if len(args) > 1 else kwargs.get("observation_space")
        self.obs_shape = {k: v.shape for k, v in observation_space.spaces.items()}
        action_space = args[2] if len(args) > 2 else kwargs.get("action_space")
        if isinstance(action_space, gym.spaces.Discrete): self.action_dim = 1
        elif isinstance(action_space, gym.spaces.MultiDiscrete): self.action_dim = len(action_space.nvec)
        else: self.action_dim = 1
        self.rnn_type = kwargs.pop("rnn_type", "lstm")
        super().__init__(*args, **kwargs)

    def add(self, obs, action, reward, episode_start, value, log_prob, lstm_states, action_masks=None):
        if action_masks is not None: self.action_masks[self.pos] = action_masks.copy()
        self.hidden_states_pi[self.pos] = lstm_states.pi[0].detach().cpu().numpy()
        self.hidden_states_vf[self.pos] = lstm_states.vf[0].detach().cpu().numpy()
        if len(lstm_states.pi) > 1:
            if not hasattr(self, "cell_states_pi"):
                self.cell_states_pi = np.zeros_like(self.hidden_states_pi)
                self.cell_states_vf = np.zeros_like(self.hidden_states_vf)
            self.cell_states_pi[self.pos] = lstm_states.pi[1].detach().cpu().numpy()
            self.cell_states_vf[self.pos] = lstm_states.vf[1].detach().cpu().numpy()
            
        for key in self.observations.keys():
            self.observations[key][self.pos] = np.array(obs[key])

        self.actions[self.pos] = np.array(action).reshape((self.n_envs, self.action_dim))
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size: self.full = True

    def reset(self):
        self.pos = 0
        self.full = False
        if hasattr(self, "hidden_state_shape"):
            self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
            self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
            if self.rnn_type == "lstm":
                self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
                self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.observations = {k: np.zeros((self.buffer_size, self.n_envs, *shape), dtype=np.float32) for k, shape in self.obs_shape.items()}
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        if isinstance(self.action_space, gym.spaces.Discrete): m_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete): m_dim = sum(self.action_space.nvec)
        else: m_dim = 1
        self.action_masks = np.ones((self.buffer_size, self.n_envs, m_dim), dtype=bool)
        self.vram_cache = {}

    def to_device(self, device: torch.device):
        self.vram_cache = {}
        def swap_and_flatten_gpu(arr):
            res = arr.swapaxes(0, 1)
            shape = res.shape
            return torch.as_tensor(res.reshape((shape[0] * shape[1], *shape[2:])), device=device)
            
        self.vram_cache["observations"] = {k: swap_and_flatten_gpu(v) for k, v in self.observations.items()}
        for attr in ["actions", "values", "log_probs", "advantages", "returns", "episode_starts", "action_masks"]:
            self.vram_cache[attr] = swap_and_flatten_gpu(getattr(self, attr))
            
        # Hidden states: (buffer_size, n_layers, n_envs, dim) -> (n_envs, buffer_size, n_layers, dim) -> (n_envs * buffer_size, n_layers, dim)
        def swap_h(h):
            res = h.swapaxes(0, 2).swapaxes(1, 2)
            shape = res.shape
            return torch.as_tensor(res.reshape((shape[0] * shape[1], *shape[2:])), device=device)

        self.vram_cache["hidden_states_pi"] = swap_h(self.hidden_states_pi)
        self.vram_cache["hidden_states_vf"] = swap_h(self.hidden_states_vf)
        if hasattr(self, "cell_states_pi"):
            self.vram_cache["cell_states_pi"] = swap_h(self.cell_states_pi)
            self.vram_cache["cell_states_vf"] = swap_h(self.cell_states_vf)

    def _get_samples_by_seq(self, seq_start_indices):
        max_seq_len = 32
        all_indices = []
        for start in seq_start_indices:
            env_idx = start // self.buffer_size
            env_end = (env_idx + 1) * self.buffer_size
            length = min(max_seq_len, env_end - start)
            all_indices.extend(range(start, start + length))
        
        all_indices = np.array(all_indices)
        device = self.device

        local_seq_starts = np.zeros(len(all_indices), dtype=bool)
        current = 0
        for start in seq_start_indices:
            env_idx = start // self.buffer_size
            env_end = (env_idx + 1) * self.buffer_size
            length = min(max_seq_len, env_end - start)
            local_seq_starts[current] = True
            current += length

        def local_pad(tensor_or_dict):
            if isinstance(tensor_or_dict, dict):
                return {k: local_pad_tensor(v[all_indices]) for k, v in tensor_or_dict.items()}
            return local_pad_tensor(tensor_or_dict[all_indices])

        # Precompute once per batch: the old loop called np.where(local_seq_starts) inside
        # every outer iteration (O(n_seq²) scans) and was a major CPU bottleneck.
        _boundary = np.where(local_seq_starts)[0]

        def local_pad_tensor(tensor):
            seqs = []
            for i, start_idx in enumerate(_boundary):
                end_idx = _boundary[i + 1] if i + 1 < len(_boundary) else len(all_indices)
                seqs.append(tensor[start_idx:end_idx])
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)

        padded_obs = local_pad(self.vram_cache["observations"])
        padded_act = local_pad(self.vram_cache["actions"])
        n_seq_batch = len(seq_start_indices)
        max_len_batch = padded_act.shape[1]
        padded_batch_size = n_seq_batch * max_len_batch

        def get_states_gpu(h_key, c_key=None):
            h = self.vram_cache[h_key][all_indices[local_seq_starts]].swapaxes(0, 1)
            if c_key in self.vram_cache:
                c = self.vram_cache[c_key][all_indices[local_seq_starts]].swapaxes(0, 1)
                return (h.contiguous(), c.contiguous())
            return (h.contiguous(),)

        return MaskableRecurrentRolloutBufferSamples(
            observations={k: v.reshape((padded_batch_size, *self.obs_shape[k])) for k, v in padded_obs.items()},
            actions=padded_act.reshape((padded_batch_size,) + self.actions.shape[2:]),
            old_values=local_pad(self.vram_cache["values"]).reshape(-1),
            old_log_prob=local_pad(self.vram_cache["log_probs"]).reshape(-1),
            advantages=local_pad(self.vram_cache["advantages"]).reshape(-1),
            returns=local_pad(self.vram_cache["returns"]).reshape(-1),
            lstm_states=RNNStates(get_states_gpu("hidden_states_pi", "cell_states_pi"), get_states_gpu("hidden_states_vf", "cell_states_vf")),
            episode_starts=local_pad(self.vram_cache["episode_starts"]).reshape(-1),
            mask=torch.ones(padded_batch_size, device=device),
            action_masks=local_pad(self.vram_cache["action_masks"]).reshape(padded_batch_size, -1))

class MaskableRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, rnn_type: str = "lstm", **kwargs):
        self.rnn_type = rnn_type.lower()
        super().__init__(*args, **kwargs)
        if self.rnn_type == "gru":
            self.lstm_actor = nn.GRU(self.features_dim, self.lstm_output_dim, num_layers=kwargs.get("n_lstm_layers", 1))
            if self.enable_critic_lstm: self.lstm_critic = nn.GRU(self.features_dim, self.lstm_output_dim, num_layers=kwargs.get("n_lstm_layers", 1))

    def _process_sequence(self, features, rnn_states, episode_starts, rnn):
        # FIX: n_seq is always the second dimension of the hidden state tensor(s)
        # For LSTM, rnn_states is ((n_layers, n_seq, dim), (n_layers, n_seq, dim))
        # For GRU, rnn_states is (n_layers, n_seq, dim) or ((n_layers, n_seq, dim),)
        if isinstance(rnn_states, tuple):
            n_seq = rnn_states[0].shape[1]
        else:
            n_seq = rnn_states.shape[1]
            
        features_sequence = features.reshape((n_seq, -1, rnn.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        reset_indices = torch.where(episode_starts.any(dim=1))[0].cpu().numpy()
        
        # Unwrap GRU states (h,) -> h for the internal rnn module
        state_input = rnn_states[0] if self.rnn_type == "gru" and isinstance(rnn_states, tuple) else rnn_states
        
        # DEBUG PRINTS
        # logger.debug(f"features_sequence shape: {features_sequence.shape}")
        # logger.debug(f"state_input shape initially: {state_input.shape}")
        
        if len(state_input.shape) != 3:
            logger.error(f"CRITICAL SHAPE ERROR! Expected 3D state_input, got {state_input.shape}")
            # Attempt auto-correction if it somehow lost the num_layers dimension
            if len(state_input.shape) == 2:
                state_input = state_input.unsqueeze(0)
        
        if len(reset_indices) == 0:
            out, r_new = rnn(features_sequence, state_input)
            if self.rnn_type == "gru" and not isinstance(r_new, tuple): r_new = (r_new,)
            return torch.flatten(out.transpose(0, 1), start_dim=0, end_dim=1), r_new
            
        outputs, current_idx = [], 0
        for next_reset in np.concatenate([reset_indices, [len(episode_starts)]]):
            if next_reset > current_idx:
                if not isinstance(state_input, tuple) and state_input.dim() != 3:
                    logger.error(f"Shape became {state_input.shape} before rnn call!")
                    state_input = state_input.unsqueeze(0)
                out, r_new = rnn(features_sequence[current_idx:next_reset], state_input)
                state_input, current_idx = r_new, next_reset
                outputs.append(out)
            if current_idx < len(episode_starts):
                mask = (1.0 - episode_starts[current_idx]).view(1, n_seq, 1)
                if isinstance(state_input, tuple):
                    state_input = tuple(s * mask for s in state_input)
                else:
                    state_input = state_input * mask
                    
                if not isinstance(state_input, tuple) and state_input.dim() != 3:
                    logger.error(f"Shape became {state_input.shape} after mask!")
                    state_input = state_input.unsqueeze(0)
                out, r_new = rnn(features_sequence[current_idx:current_idx+1], state_input)
                state_input, current_idx = r_new, current_idx + 1
                outputs.append(out)
                
        if self.rnn_type == "gru" and not isinstance(state_input, tuple): 
            state_input = (state_input,)
            
        return torch.flatten(torch.cat(outputs, dim=0).transpose(0, 1), start_dim=0, end_dim=1), state_input

    def forward(self, obs, lstm_states, episode_starts, action_masks=None, deterministic=False):
        features = self.extract_features(obs)
        pi_f, vf_f = (features, features) if self.share_features_extractor else features
        l_pi, s_pi = self._process_sequence(pi_f, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic: l_vf, s_vf = self._process_sequence(vf_f, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm: l_vf, s_vf = l_pi.detach(), tuple(s.detach() for s in s_pi)
        else: l_vf, s_vf = self.critic(vf_f), s_pi
        l_pi, l_vf = self.mlp_extractor.forward_actor(l_pi), self.mlp_extractor.forward_critic(l_vf)
        values, dist = self.value_net(l_vf), self._get_action_dist_from_latent(l_pi)
        if action_masks is not None: dist.apply_masking(action_masks)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, values, dist.log_prob(actions), RNNStates(s_pi, s_vf)

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def get_distribution(self, obs, lstm_states, episode_starts, action_masks=None):
        features = self.extract_features(obs)
        pi_f, vf_f = (features, features) if self.share_features_extractor else features
        l_pi, s_pi = self._process_sequence(pi_f, lstm_states.pi, episode_starts, self.lstm_actor)
        s_vf = self._process_sequence(vf_f, lstm_states.vf, episode_starts, self.lstm_critic)[1] if self.lstm_critic else (tuple(s.detach() for s in s_pi) if self.shared_lstm else s_pi)
        dist = self._get_action_dist_from_latent(self.mlp_extractor.forward_actor(l_pi))
        if action_masks is not None: dist.apply_masking(action_masks)
        return dist, RNNStates(s_pi, s_vf)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts, action_masks=None):
        features = self.extract_features(obs)
        pi_f, vf_f = (features, features) if self.share_features_extractor else features
        l_pi, _ = self._process_sequence(pi_f, lstm_states.pi, episode_starts, self.lstm_actor)
        l_vf = self._process_sequence(vf_f, lstm_states.vf, episode_starts, self.lstm_critic)[0] if self.lstm_critic else (l_pi.detach() if self.shared_lstm else self.critic(vf_f))
        l_pi, l_vf = self.mlp_extractor.forward_actor(l_pi), self.mlp_extractor.forward_critic(l_vf)
        dist = self._get_action_dist_from_latent(l_pi)
        if action_masks is not None: dist.apply_masking(action_masks)
        return self.value_net(l_vf), dist.log_prob(actions), dist.entropy()

    def predict(self, observation, state=None, episode_start=None, deterministic=False, action_masks=None):
        self.set_training_mode(False)
        obs_t, vectorized = self.obs_to_tensor(observation)
        
        # FIX: Handle Dict observation spaces
        if isinstance(obs_t, dict):
            # Take the first available tensor to determine the number of environments
            first_key = next(iter(obs_t))
            n_envs = obs_t[first_key].shape[0]
        else:
            n_envs = obs_t.shape[0]

        if state is None:
            h = np.zeros((self.lstm_actor.num_layers, n_envs, self.lstm_actor.hidden_size))
            state = ((h, h), (h, h)) if self.rnn_type == "lstm" else ((h,), (h,))
        if episode_start is None: episode_start = np.zeros((n_envs,), dtype=bool)
        with torch.no_grad():
            def to_t(s): return tuple(torch.as_tensor(t).to(self.device).float() for t in s)
            l_states = RNNStates(to_t(state[0]), to_t(state[1]))
            starts = torch.as_tensor(episode_start).to(self.device).float()
            m = torch.as_tensor(action_masks).to(self.device) if action_masks is not None else None
            acts, _, _, l_states = self.forward(obs_t, l_states, starts, action_masks=m, deterministic=deterministic)
            state = (tuple(t.cpu().numpy() for t in l_states.pi), tuple(t.cpu().numpy() for t in l_states.vf))
        res = acts.cpu().numpy()
        return (res if vectorized else res[0]), state

class MaskableMlpLstmPolicy(MaskableRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if "features_extractor_class" not in kwargs: kwargs["features_extractor_class"] = FlattenExtractor
        super().__init__(*args, **kwargs)

class MaskableCnnLstmPolicy(MaskableRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if "features_extractor_class" not in kwargs: kwargs["features_extractor_class"] = NatureCNN
        super().__init__(*args, **kwargs)

class MaskableMultiInputLstmPolicy(MaskableRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if "features_extractor_class" not in kwargs: kwargs["features_extractor_class"] = CombinedExtractor
        super().__init__(*args, **kwargs)

from imitation_learning.buffer import ImitationReplayBuffer

class MaskableRecurrentPPO(RecurrentPPO):
    policy_class = MaskableRecurrentActorCriticPolicy
    policy_aliases = {"MlpLstmPolicy": MaskableMlpLstmPolicy, "CnnLstmPolicy": MaskableCnnLstmPolicy, "MultiInputLstmPolicy": MaskableMultiInputLstmPolicy}

    def __init__(self, policy, env, learning_rate=3e-4, n_steps=128, batch_size=128, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 use_sde=False, sde_sample_freq=-1, target_kl=None, stats_window_size=100, tensorboard_log=None,
                 policy_kwargs=None, verbose=0, seed=None, device="auto", _init_setup_model=True, rnn_type="lstm",
                 imitation_data_percentage=0.0, imitation_data_max_steps=0, imitation_coef=0.5, level_name=None, **kwargs):
        self.rnn_type = rnn_type.lower()
        if policy_kwargs is None: policy_kwargs = {}
        policy_kwargs["rnn_type"] = self.rnn_type
        super().__init__(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf,
                         normalize_advantage, ent_coef, vf_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl,
                         stats_window_size, tensorboard_log, policy_kwargs, int(verbose), seed, device, _init_setup_model)
        self.imitation_data_percentage, self.imitation_data_max_steps, self.imitation_coef = imitation_data_percentage, imitation_data_max_steps, imitation_coef
        self.imitation_buffer = None
        self._start_timesteps = self.num_timesteps
        if _init_setup_model: self._setup_imitation()

    def _setup_imitation(self, level_name: Optional[str] = None,
                         imitation_data_percentage: Optional[float] = None,
                         imitation_data_max_steps: Optional[int] = None,
                         imitation_coef: Optional[float] = None):
        """Initializes the imitation buffer if data exists for the current level."""
        if imitation_data_percentage is not None: self.imitation_data_percentage = imitation_data_percentage
        if imitation_data_max_steps is not None: self.imitation_data_max_steps = imitation_data_max_steps
        if imitation_coef is not None: self.imitation_coef = imitation_coef
        if self.imitation_data_percentage <= 0:
            self.imitation_buffer = None
            return
        level_name = level_name or getattr(self, "level_name", None)
        if not level_name and self.env is not None:
            level_name = getattr(self.env, "level_name", None)
            if not level_name and hasattr(self.env, "get_attr"):
                try:
                    attr_vals = self.env.get_attr("level_name", indices=[0])
                    if attr_vals: level_name = attr_vals[0]
                except: pass
        if level_name:
            self.imitation_buffer = ImitationReplayBuffer(level_name)
            if self.imitation_buffer.total_size == 0: self.imitation_buffer = None
            else: logger.info(f"Imitation learning ENABLED for level {level_name}")
        else: self.imitation_buffer = None

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs).to(self.device)
        if isinstance(self.action_space, gym.spaces.Discrete): self.policy.action_dist = MaskableCategoricalDistribution(self.action_space.n)
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete): self.policy.action_dist = MaskableMultiCategoricalDistribution(self.action_space.nvec)
        elif isinstance(self.action_space, gym.spaces.MultiBinary): self.policy.action_dist = MaskableBernoulliDistribution(self.action_space.n)
        lstm = self.policy.lstm_actor
        s_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        def r_tuple():
            h = torch.zeros(s_shape, device=self.device)
            return (h, torch.zeros(s_shape, device=self.device)) if self.rnn_type == "lstm" else (h,)
        self._last_lstm_states = RNNStates(r_tuple(), r_tuple())
        h_buf_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        b_cls = MaskableRecurrentDictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else MaskableRecurrentRolloutBuffer
        self.rollout_buffer = b_cls(self.n_steps, self.observation_space, self.action_space, h_buf_shape, self.device, self.gamma, self.gae_lambda, self.n_envs, rnn_type=self.rnn_type)
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf: self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        lstm_states = deepcopy(self._last_lstm_states)
        with torch.no_grad():
            while n_steps < n_rollout_steps:
                m = get_action_masks(env) if is_masking_supported(env) else None
                obs_t = self.policy.obs_to_tensor(self._last_obs)[0]
                starts_t = torch.as_tensor(self._last_episode_starts).float().to(self.device)
                m_t = torch.as_tensor(m).to(self.device) if m is not None else None
                acts, vals, log_ps, lstm_states = self.policy.forward(obs_t, lstm_states, starts_t, action_masks=m_t)
                res_acts = acts.cpu().numpy()
                new_obs, rewards, dones, infos = env.step(res_acts)
                self.num_timesteps += env.num_envs
                if callback.on_step() is False: return False
                self._update_info_buffer(infos)
                n_steps += 1
                rollout_buffer.add(self._last_obs, acts.cpu().numpy() if not isinstance(self.action_space, gym.spaces.Discrete) else acts.cpu().numpy().reshape(-1, 1),
                                   rewards, self._last_episode_starts, vals, log_ps, self._last_lstm_states, m)
                self._last_obs, self._last_episode_starts, self._last_lstm_states = new_obs, dones, lstm_states
        with torch.no_grad():
            vals = self.policy.predict_values(self.policy.obs_to_tensor(new_obs)[0], lstm_states.vf, torch.tensor(dones, dtype=torch.float32, device=self.device))
        rollout_buffer.compute_returns_and_advantage(vals, dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self.rollout_buffer.to_device(self.device)
        c_r = self.clip_range(self._current_progress_remaining)
        c_r_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None
        
        entropy_losses, pg_losses, value_losses, approx_kls, clip_fractions, imitation_losses, loss_ls = [], [], [], [], [], [], []
        
        continue_training = True
        for epoch in range(self.n_epochs):
            for r_data in self.rollout_buffer.get(self.batch_size):
                acts = r_data.actions.long().flatten() if isinstance(self.action_space, gym.spaces.Discrete) else r_data.actions
                vals, log_p, entropy = self.policy.evaluate_actions(r_data.observations, acts, r_data.lstm_states, r_data.episode_starts, r_data.action_masks)
                
                # ... (rest of standard PPO logic) ...
                mask = r_data.mask > 1e-8
                adv = r_data.advantages
                if self.normalize_advantage: adv = (adv - adv[mask].mean()) / (adv[mask].std() + 1e-8)
                
                ratio = torch.exp(log_p - r_data.old_log_prob)
                
                # Policy loss
                pg_1 = adv * ratio
                pg_2 = adv * torch.clamp(ratio, 1 - c_r, 1 + c_r)
                policy_loss = -torch.mean(torch.min(pg_1, pg_2)[mask])
                pg_losses.append(policy_loss.item())
                
                # Clip fraction
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > c_r).float()[mask]).item())

                # Value loss
                if c_r_vf is None:
                    v_pred = vals.flatten()
                else:
                    v_pred = r_data.old_values + torch.clamp(vals.flatten() - r_data.old_values, -c_r_vf, c_r_vf)
                
                value_loss = torch.mean(((r_data.returns - v_pred)**2)[mask])
                value_losses.append(value_loss.item())
                
                # Entropy loss
                entropy_loss = -torch.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                with torch.no_grad():
                    log_ratio = log_p - r_data.old_log_prob
                    # Single scalar sync (avoid .cpu().numpy() per batch — extra GPU sync)
                    approx_kl_div = torch.mean(((torch.exp(log_ratio) - 1) - log_ratio)[mask]).item()
                    approx_kls.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # --- IMITATION LOSS ---
                imitation_loss = torch.tensor(0.0, device=self.device)
                if self.imitation_buffer and self.num_timesteps < self.imitation_data_max_steps:
                    if np.random.rand() < (self.imitation_data_percentage / 100.0):
                        expert_data = self.imitation_buffer.sample_batch(self.batch_size)
                        if expert_data:
                            ex_obs, ex_acts, ex_masks = expert_data
                            ex_obs_t = self.policy.obs_to_tensor(ex_obs)[0]
                            ex_acts_t = torch.as_tensor(ex_acts, device=self.device).long()
                            ex_masks_t = torch.as_tensor(ex_masks, device=self.device).bool() if ex_masks is not None else None
                            
                            if ex_masks_t is not None and len(ex_acts_t.shape) == 1:
                                ex_masks_t.scatter_(1, ex_acts_t.unsqueeze(1), True)

                            # Evaluate expert actions (treating them as independent frames)
                            # We provide zero-initialized states and starts for the recurrent policy
                            n_ex = ex_obs_t.shape[0]
                            s_shape = (self.policy.lstm_actor.num_layers, n_ex, self.policy.lstm_actor.hidden_size)
                            def r_tuple():
                                h = torch.zeros(s_shape, device=self.device)
                                return (h, torch.zeros(s_shape, device=self.device)) if self.rnn_type == "lstm" else (h,)
                            ex_states = RNNStates(r_tuple(), r_tuple())
                            ex_starts = torch.zeros((n_ex,), device=self.device)

                            _, ex_log_p, _ = self.policy.evaluate_actions(ex_obs_t, ex_acts_t, ex_states, ex_starts, action_masks=ex_masks_t)
                            imitation_loss = -torch.mean(ex_log_p) * self.imitation_coef
                            imitation_losses.append(imitation_loss.item())

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss + imitation_loss
                loss_ls.append(loss.item())
                
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            
            if not continue_training:
                break

        self.rollout_buffer.clear_vram_cache()
        self._n_updates += self.n_epochs
        
        # Calculate explained variance (only for valid steps)
        v_true = self.rollout_buffer.returns.flatten()
        v_pred = self.rollout_buffer.values.flatten()
        expl_var = explained_variance(v_pred, v_true)

        # LOG ALL METRICS
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kls))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", np.mean(loss_ls))
        self.logger.record("train/explained_variance", expl_var)
        if len(imitation_losses) > 0:
            self.logger.record("train/imitation_loss", np.mean(imitation_losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", self.clip_range(self._current_progress_remaining))
        self.logger.record("train/learning_rate", self.learning_rate)
        if c_r_vf is not None:
            self.logger.record("train/clip_range_vf", c_r_vf)

    def predict(self, observation, state=None, episode_start=None, deterministic=False, action_masks=None):
        return self.policy.predict(observation, state, episode_start, deterministic, action_masks=action_masks)

    @classmethod
    def load(cls, path, env=None, device="auto", custom_objects=None, **kwargs):
        if custom_objects is None: custom_objects = {}
        from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
        custom_objects.setdefault("policy_class", UniversalRecurrentPolicy)
        
        # Pop architecture info from kwargs to avoid conflicts with explicit arguments later
        pk_override = kwargs.pop("policy_kwargs", {})
        if pk_override:
            if "policy_kwargs" not in custom_objects: custom_objects["policy_kwargs"] = {}
            custom_objects["policy_kwargs"].update(pk_override)
        
        custom_objects.setdefault("policy_kwargs", {}).setdefault("lstm_hidden_size", 512)
        
        rnn_override = kwargs.pop("rnn_type", None)
        if rnn_override:
            custom_objects["rnn_type"] = rnn_override
            if "policy_kwargs" not in custom_objects: custom_objects["policy_kwargs"] = {}
            custom_objects["policy_kwargs"]["rnn_type"] = rnn_override
            
        custom_objects["optimizer"] = None
        tb_log = kwargs.pop("tensorboard_log", None)
        
        try: 
            # Try normal load first
            model = super().load(path, env, device, custom_objects, **kwargs)
            if tb_log is not None:
                model.tensorboard_log = tb_log
                from stable_baselines3.common.logger import configure
                new_logger = configure(tb_log, ["stdout", "tensorboard"])
                model.set_logger(new_logger)
            return model
        except Exception as e:
            logger.warning(f"Surgical reconstruction required: {e}")
            from stable_baselines3.common.save_util import load_from_zip_file
            data, params, _ = load_from_zip_file(path, device, custom_objects)
            data["policy_class"] = custom_objects["policy_class"]
            
            # Ensure policy_kwargs in data is updated with our overrides
            if "policy_kwargs" not in data: data["policy_kwargs"] = {}
            data["policy_kwargs"].update(custom_objects.get("policy_kwargs", {}))
            
            if tb_log is not None:
                data["tensorboard_log"] = tb_log

            # Reconstruct with the best available information
            rnn_type = custom_objects.get("rnn_type") or data.get("rnn_type", "lstm")
            
            # Ensure rnn_type is also in data for model persistence
            data["rnn_type"] = rnn_type
            data["policy_kwargs"]["rnn_type"] = rnn_type

            # Create model instance - kwargs no longer contains rnn_type or policy_kwargs
            model = cls(policy=data["policy_class"], env=env, device=device, _init_setup_model=False, rnn_type=rnn_type, **kwargs)
            
            if isinstance(model.observation_space, gym.spaces.Dict):
                from src.agent.models.unit_control_network import UnitControlNetwork
                data.get("policy_kwargs", {})["features_extractor_class"] = UnitControlNetwork
            
            model.__dict__.update(data)
            
            if tb_log is not None:
                from stable_baselines3.common.logger import configure
                new_logger = configure(tb_log, ["stdout", "tensorboard"])
                model.set_logger(new_logger)

            model._setup_model()
            p_sd, c_sd = params.get("policy", {}), model.policy.state_dict()
            new_sd = {}
            for k, v in p_sd.items():
                keys = [k]
                _has_fe_cnn = (
                    "features_extractor.cnn.0.weight" in c_sd
                    or "features_extractor.cnn_trunk.0.weight" in c_sd
                )
                _has_pi_cnn = (
                    "pi_features_extractor.cnn.0.weight" in c_sd
                    or "pi_features_extractor.cnn_trunk.0.weight" in c_sd
                )
                if k.startswith("pi_features_extractor.") and _has_fe_cnn:
                    keys = [k.replace("pi_features_extractor.", "features_extractor.")]
                elif k.startswith("features_extractor.") and _has_pi_cnn:
                    keys = [k.replace("features_extractor.", "pi_features_extractor."), k.replace("features_extractor.", "vf_features_extractor.")]
                for tk in keys:
                    if tk in c_sd:
                        if v.shape == c_sd[tk].shape: new_sd[tk] = v
                        elif len(v.shape) == 4 and v.shape[1] < c_sd[tk].shape[1] and v.shape[0] == c_sd[tk].shape[0] and v.shape[2:] == c_sd[tk].shape[2:]:
                            logger.info(f"Expanding CNN channels for {tk}"); exp = c_sd[tk].clone(); exp[:, :v.shape[1], :, :] = v; new_sd[tk] = exp
                        elif len(v.shape) == 2 and v.shape[1] < c_sd[tk].shape[1] and v.shape[0] == c_sd[tk].shape[0] and "linear" in tk:
                            logger.info(f"Expanding Linear input features for {tk}: {v.shape[1]} -> {c_sd[tk].shape[1]}")
                            exp = c_sd[tk].clone(); exp[:, :v.shape[1]] = v; new_sd[tk] = exp
                        elif "action_net" in tk or "command_head" in tk:
                            if v.shape[1] == c_sd[tk].shape[1]:
                                logger.warning(f"Partial action head load for {tk}")
                                exp = c_sd[tk].clone()
                                mi = min(v.shape[0], c_sd[tk].shape[0])
                                exp[:mi, :] = v[:mi, :]
                                new_sd[tk] = exp
                            else:
                                logger.error(f"Cannot surgically load {tk} due to hidden size mismatch: {v.shape[1]} vs {c_sd[tk].shape[1]}")
            model.policy.load_state_dict(new_sd, strict=False)
            model._setup_lr_schedule()
            model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(), lr=model.learning_rate, **model.policy.optimizer_kwargs)
            return model
