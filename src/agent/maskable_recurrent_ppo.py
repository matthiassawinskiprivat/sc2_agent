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
    """
    Custom version of SB3's pad function that handles both numpy and torch tensors
    without triggering 'copy construct' warnings.
    """
    seq = []
    for start, end in zip(seq_start_indices, seq_end_indices):
        chunk = tensor[start : end + 1]
        if isinstance(chunk, torch.Tensor):
            # Safe way to handle existing tensors
            seq.append(chunk.clone().detach().to(device))
        else:
            seq.append(torch.as_tensor(chunk, device=device))
            
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)

def tensor_safe_pad_and_flatten(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: torch.device,
    tensor: Union[np.ndarray, torch.Tensor],
    padding_value: float = 0.0,
) -> torch.Tensor:
    return tensor_safe_pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten()

def create_tensor_safe_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Any, Any]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.
    """
    seq_start = np.logical_or(episode_starts, env_change).flatten()
    seq_start[0] = True
    seq_start_indices = np.where(seq_start == True)[0]
    seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)])])      

    from functools import partial
    local_pad = partial(tensor_safe_pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(tensor_safe_pad_and_flatten, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, local_pad, local_pad_and_flatten

def create_tensor_safe_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Any, Any]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.
    """
    seq_start = np.logical_or(episode_starts, env_change).flatten()
    seq_start[0] = True
    seq_start_indices = np.where(seq_start == True)[0]
    seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)])])      

    from functools import partial
    local_pad = partial(tensor_safe_pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(tensor_safe_pad_and_flatten, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, local_pad, local_pad_and_flatten

class MaskableRecurrentRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
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
    """
    Rollout buffer that also stores action masks.
    Supports both LSTM and GRU.
    Included VRAM Caching optimization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            action_dim = sum(self.action_space.nvec)
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
            action_dim = self.action_space.n
        else:
             raise NotImplementedError("Only Discrete, MultiDiscrete and MultiBinary action spaces are supported for masking")
             
        self.action_masks = np.ones((self.buffer_size, self.n_envs, action_dim), dtype=bool)
        self.vram_cache: Dict[str, torch.Tensor] = {}

    def reset(self):
        super().reset()
        if hasattr(self, "action_masks"):
            self.action_masks.fill(True)
        self.clear_vram_cache()

    def clear_vram_cache(self):
        self.vram_cache = {}

    def to_device(self, device: torch.device):
        """
        Move all buffer data to GPU once to speed up sampling.
        Matches RecurrentRolloutBuffer.get() flattening logic.
        """
        self.clear_vram_cache()
        
        # 1. Handle hidden states swap: (B, L, E, D) -> (B, E, L, D)
        h_pi = self.hidden_states_pi.swapaxes(1, 2)
        h_vf = self.hidden_states_vf.swapaxes(1, 2)
        
        if hasattr(self, "cell_states_pi"):
            c_pi = self.cell_states_pi.swapaxes(1, 2)
            c_vf = self.cell_states_vf.swapaxes(1, 2)

        # 2. Swap and flatten: (B, E, ...) -> (E, B, ...) -> (E * B, ...)
        def swap_and_flatten_gpu(arr):
            # We must use the same logic as RecurrentRolloutBuffer.swap_and_flatten
            # Which is: arr.swapaxes(0, 1).reshape((-1, *arr.shape[2:]))
            res = arr.swapaxes(0, 1)
            shape = res.shape
            return res.reshape((shape[0] * shape[1], *shape[2:]))

        self.vram_cache["observations"] = torch.as_tensor(swap_and_flatten_gpu(self.observations)).to(device)
        self.vram_cache["actions"] = torch.as_tensor(swap_and_flatten_gpu(self.actions)).to(device)
        self.vram_cache["values"] = torch.as_tensor(swap_and_flatten_gpu(self.values)).to(device)
        self.vram_cache["log_probs"] = torch.as_tensor(swap_and_flatten_gpu(self.log_probs)).to(device)
        self.vram_cache["advantages"] = torch.as_tensor(swap_and_flatten_gpu(self.advantages)).to(device)
        self.vram_cache["returns"] = torch.as_tensor(swap_and_flatten_gpu(self.returns)).to(device)
        self.vram_cache["episode_starts"] = torch.as_tensor(swap_and_flatten_gpu(self.episode_starts)).to(device)
        self.vram_cache["action_masks"] = torch.as_tensor(swap_and_flatten_gpu(self.action_masks)).to(device)
        
        self.vram_cache["hidden_states_pi"] = torch.as_tensor(swap_and_flatten_gpu(h_pi)).to(device)
        self.vram_cache["hidden_states_vf"] = torch.as_tensor(swap_and_flatten_gpu(h_vf)).to(device)
        if hasattr(self, "cell_states_pi"):
            self.vram_cache["cell_states_pi"] = torch.as_tensor(swap_and_flatten_gpu(c_pi)).to(device)
            self.vram_cache["cell_states_vf"] = torch.as_tensor(swap_and_flatten_gpu(c_vf)).to(device)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        lstm_states: RNNStates,
        action_masks: Optional[np.ndarray] = None,
    ) -> None:
        if action_masks is not None:
             self.action_masks[self.pos] = action_masks.copy()
             
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        
        if len(lstm_states.pi) > 1:
            self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
            self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> MaskableRecurrentRolloutBufferSamples:
        if not self.vram_cache:
            return self._get_samples_standard(batch_inds, env_change, env)
            
        episode_starts_batch = self.vram_cache["episode_starts"][batch_inds].cpu().numpy()
        env_change_batch = env_change[batch_inds]

        self.seq_start_indices, self.pad, self.pad_and_flatten = create_tensor_safe_sequencers(
            episode_starts_batch.flatten(), env_change_batch.flatten(), self.device
        )
        
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.vram_cache["actions"][batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        
        def get_states_gpu(hidden_key, cell_key=None):
            h = self.vram_cache[hidden_key][batch_inds][self.seq_start_indices].swapaxes(0, 1)
            if cell_key in self.vram_cache:
                c = self.vram_cache[cell_key][batch_inds][self.seq_start_indices].swapaxes(0, 1)
                return (h.contiguous(), c.contiguous())
            return (h.contiguous(),)

        lstm_states_pi = get_states_gpu("hidden_states_pi", "cell_states_pi")
        lstm_states_vf = get_states_gpu("hidden_states_vf", "cell_states_vf")

        return MaskableRecurrentRolloutBufferSamples(
            observations=self.pad(self.vram_cache["observations"][batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.vram_cache["actions"][batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.vram_cache["values"][batch_inds]),
            old_log_prob=self.pad_and_flatten(self.vram_cache["log_probs"][batch_inds]),
            advantages=self.pad_and_flatten(self.vram_cache["advantages"][batch_inds]),
            returns=self.pad_and_flatten(self.vram_cache["returns"][batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.vram_cache["episode_starts"][batch_inds]),
            mask=self.pad_and_flatten(torch.ones_like(self.vram_cache["returns"][batch_inds])),
            action_masks=self.pad_and_flatten(self.vram_cache["action_masks"][batch_inds])
        )

    def _get_samples_standard(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> MaskableRecurrentRolloutBufferSamples:
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        
        def get_states(hidden_states, cell_states):
            h = hidden_states[batch_inds][self.seq_start_indices].swapaxes(0, 1)
            h = self.to_torch(h).contiguous()
            c = cell_states[batch_inds][self.seq_start_indices].swapaxes(0, 1)
            c = self.to_torch(c).contiguous()
            return (h, c)

        lstm_states_pi = get_states(self.hidden_states_pi, self.cell_states_pi)
        lstm_states_vf = get_states(self.hidden_states_vf, self.cell_states_vf)

        return MaskableRecurrentRolloutBufferSamples(
            observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
            action_masks=self.pad_and_flatten(self.action_masks[batch_inds])
        )

class MaskableRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction)
    with action masking support.
    
    Supports both LSTM and GRU.
    """

    def __init__(self, *args, rnn_type: str = "lstm", **kwargs):
        self.rnn_type = rnn_type.lower()
        super().__init__(*args, **kwargs)
        
        if self.rnn_type == "gru":
            self.lstm_actor = nn.GRU(
                self.features_dim,
                self.lstm_output_dim,
                num_layers=kwargs.get("n_lstm_layers", 1),
                **kwargs.get("lstm_kwargs", {}),
            )
            if self.enable_critic_lstm:
                self.lstm_critic = nn.GRU(
                    self.features_dim,
                    self.lstm_output_dim,
                    num_layers=kwargs.get("n_lstm_layers", 1),
                    **kwargs.get("lstm_kwargs", {}),
                )

    def _process_sequence(
        self,
        features: torch.Tensor,
        rnn_states: Tuple[torch.Tensor, ...],
        episode_starts: torch.Tensor,
        rnn: nn.Module,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Do a forward pass in the RNN network (LSTM or GRU).
        Optimized by chunking sequences at reset points.
        This provides near-C++ speed while correctly handling episode resets.
        """
        n_seq = rnn_states[0].shape[1]
        # features: (Batch*n_seq, input_dim) -> (n_seq, sequence_length, input_dim) -> (sequence_length, n_seq, input_dim)
        features_sequence = features.reshape((n_seq, -1, rnn.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # Identify reset points across the whole batch
        # If any env resets at time T, we must break the chunk
        reset_indices = torch.where(episode_starts.any(dim=1))[0].cpu().numpy()
        
        if len(reset_indices) == 0:
            # Standard fast path (C++ loop)
            state_input = rnn_states if self.rnn_type == "lstm" else rnn_states[0]
            rnn_output, rnn_states_new = rnn(features_sequence, state_input)
            if self.rnn_type == "gru": 
                rnn_states_new = (rnn_states_new,)
            
            rnn_output = torch.flatten(rnn_output.transpose(0, 1), start_dim=0, end_dim=1)
            return rnn_output, rnn_states_new

        # Chunked fast path
        outputs = []
        current_idx = 0
        # Add the end of sequence as a virtual reset point for the loop logic
        all_reset_points = np.concatenate([reset_indices, [len(episode_starts)]])
        
        for next_reset in all_reset_points:
            # 1. Process chunk before the reset
            if next_reset > current_idx:
                chunk = features_sequence[current_idx:next_reset]
                state_input = rnn_states if self.rnn_type == "lstm" else rnn_states[0]
                out, rnn_states_new = rnn(chunk, state_input)
                if self.rnn_type == "gru": rnn_states_new = (rnn_states_new,)
                rnn_states = rnn_states_new
                outputs.append(out)
                current_idx = next_reset
            
            # 2. Handle the reset and process the reset step
            if current_idx < len(episode_starts):
                # Apply resets to the hidden state before processing the reset step
                mask = (1.0 - episode_starts[current_idx]).view(1, n_seq, 1)
                rnn_states = tuple(s * mask for s in rnn_states)
                
                # Process exactly one step
                chunk = features_sequence[current_idx:current_idx+1]
                state_input = rnn_states if self.rnn_type == "lstm" else rnn_states[0]
                out, rnn_states_new = rnn(chunk, state_input)
                if self.rnn_type == "gru": rnn_states_new = (rnn_states_new,)
                rnn_states = rnn_states_new
                outputs.append(out)
                current_idx += 1
                
        rnn_output = torch.cat(outputs, dim=0)
        rnn_output = torch.flatten(rnn_output.transpose(0, 1), start_dim=0, end_dim=1)
        return rnn_output, rnn_states

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
            
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = tuple(s.detach() for s in lstm_states_pi)
        else:
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if action_masks is not None:
            distribution.apply_masking(action_masks)
            
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> MaskableDistribution:
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, MaskableCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MaskableMultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MaskableBernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution for masking")
            
    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[MaskableDistribution, RNNStates]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
            
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        
        if self.lstm_critic is not None:
             _, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
             lstm_states_vf = tuple(s.detach() for s in lstm_states_pi)
        else:
             lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if action_masks is not None:
            distribution.apply_masking(action_masks)
            
        return distribution, RNNStates(lstm_states_pi, lstm_states_vf)
        
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor, 
        lstm_states: RNNStates, 
        episode_starts: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
            
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
            
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy
        
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[list(observation.keys())[0]].shape[0]
        else:
            n_envs = observation.shape[0]

        if state is None:
            single_hidden = np.zeros(self.lstm_hidden_state_shape)
            h = np.concatenate([single_hidden for _ in range(n_envs)], axis=1)
            if self.rnn_type == "lstm":
                c = np.concatenate([single_hidden for _ in range(n_envs)], axis=1)
                state = ((h, c), (h, c))
            else:
                state = ((h,), (h,))
            
        if episode_start is None:
            episode_start = np.zeros((n_envs,), dtype=bool)

        with torch.no_grad():
            def to_tensor_tuple(s):
                return tuple(torch.as_tensor(t).to(self.device).float() for t in s)
                
            lstm_states = RNNStates(
                pi=to_tensor_tuple(state[0]),
                vf=to_tensor_tuple(state[1])
            )
            
            episode_starts = torch.as_tensor(episode_start).to(self.device).float()
            
            action_masks_tensor = None
            if action_masks is not None:
                action_masks_tensor = torch.as_tensor(action_masks).to(self.device)

            actions, _, _, lstm_states = self.forward(
                observation,
                lstm_states,
                episode_starts,
                action_masks=action_masks_tensor,
                deterministic=deterministic
            )
            
            def to_numpy_tuple(s):
                return tuple(t.cpu().numpy() for t in s)
                
            state = (to_numpy_tuple(lstm_states.pi), to_numpy_tuple(lstm_states.vf))

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions.cpu().numpy())
            else:
                actions = np.clip(actions.cpu().numpy(), self.action_space.low, self.action_space.high)
        else:
            actions = actions.cpu().numpy()

        if vectorized_env:
            return actions, state

        return actions[0], state


class MaskableMlpLstmPolicy(MaskableRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = FlattenExtractor
        super().__init__(*args, **kwargs)

class MaskableCnnLstmPolicy(MaskableRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = NatureCNN
        super().__init__(*args, **kwargs)

class MaskableMultiInputLstmPolicy(MaskableRecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = CombinedExtractor
        super().__init__(*args, **kwargs)


from imitation_learning.buffer import ImitationReplayBuffer

class MaskableRecurrentPPO(RecurrentPPO):
    """
    Custom PPO implementation that supports:
    1. Action Masking (from sb3-contrib MaskablePPO)
    2. Recurrent Policy (from sb3-contrib RecurrentPPO)
    3. Hybrid Action Space (Spatial + Categorical)
    4. Imitation Learning from expert demonstrations
    """
    policy_class = MaskableRecurrentActorCriticPolicy

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpLstmPolicy": MaskableMlpLstmPolicy,
        "CnnLstmPolicy": MaskableCnnLstmPolicy,
        "MultiInputLstmPolicy": MaskableMultiInputLstmPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: int = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        # New parameters for recurrent support
        rnn_type: str = "lstm",
        # New parameters for imitation learning
        imitation_data_percentage: float = 0.0,
        imitation_data_max_steps: int = 0,
        imitation_coef: float = 0.5,
        level_name: Optional[str] = None,
        **kwargs,
    ):
        # Ensure verbose is an int to prevent TypeErrors in SB3 base class
        try:
            verbose = int(verbose)
        except (ValueError, TypeError):
            verbose = 0

        self.rnn_type = rnn_type.lower()
        # Ensure rnn_type is passed to policy_kwargs for the policy to use
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs["rnn_type"] = self.rnn_type

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
        self.imitation_data_percentage = imitation_data_percentage
        self.imitation_data_max_steps = imitation_data_max_steps
        self.imitation_coef = imitation_coef
        self.imitation_buffer = None
        self._start_timesteps = self.num_timesteps
        
        if _init_setup_model:
            self._setup_imitation()

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["imitation_buffer"]

    def _setup_imitation(self, level_name: Optional[str] = None, 
                         imitation_data_percentage: Optional[float] = None,
                         imitation_data_max_steps: Optional[int] = None,
                         imitation_coef: Optional[float] = None):
        """Initializes the imitation buffer if data exists for the current level."""
        # Capture current steps as the 'start' for relative imitation tracking
        self._start_timesteps = self.num_timesteps
        
        # Apply Overrides
        if imitation_data_percentage is not None: self.imitation_data_percentage = imitation_data_percentage
        if imitation_data_max_steps is not None: self.imitation_data_max_steps = imitation_data_max_steps
        if imitation_coef is not None: self.imitation_coef = imitation_coef

        if self.imitation_data_percentage <= 0:
            self.imitation_buffer = None
            return

        # Priority: 1. Argument, 2. Instance Attribute
        level_name = level_name or getattr(self, "level_name", None)
        
        # Fallback to env resolution if not provided
        if not level_name and self.env is not None:
            # VecEnv might have 'level_name' attribute if it's SyncVecEnv/SubprocVecEnv wrapping our envs
            level_name = getattr(self.env, "level_name", None)
            
            # Try to get from first wrapped env if it's a VecEnv
            if not level_name and hasattr(self.env, "get_attr"):
                try:
                    attr_vals = self.env.get_attr("level_name", indices=[0])
                    if attr_vals:
                        level_name = attr_vals[0]
                except:
                    pass
            
        if level_name:
            self.imitation_buffer = ImitationReplayBuffer(level_name)
            if self.imitation_buffer.total_size == 0:
                self.imitation_buffer = None
                logger.opt(colors=True).warning(f"<yellow>Imitation buffer for {level_name} is EMPTY. Check translated replays.</yellow>")
            else:
                logger.opt(colors=True).info(f"<yellow>Imitation learning ENABLED for level {level_name}. Level Start: {self._start_timesteps}, Max Steps: {self.imitation_data_max_steps}</yellow>")
                logger.info(f"Settings: Percentage: {self.imitation_data_percentage}%, Coef: {self.imitation_coef}")
        else:
            self.imitation_buffer = None
            logger.warning("Imitation learning: Could not resolve level_name. Buffer disabled.")

        
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        if isinstance(self.action_space, gym.spaces.Discrete):
             self.policy.action_dist = MaskableCategoricalDistribution(self.action_space.n)
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
             self.policy.action_dist = MaskableMultiCategoricalDistribution(self.action_space.nvec)
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
             self.policy.action_dist = MaskableBernoulliDistribution(self.action_space.n)
        else:
             raise NotImplementedError("Only Discrete, MultiDiscrete and MultiBinary action spaces are supported for masking")

        if not hasattr(self.policy, "lstm_actor"):
            raise RuntimeError(
                f"Error: The policy {type(self.policy)} is not a recurrent policy. "
                "Ensure you are using a recurrent policy (e.g., CnnLstmPolicy) with MaskableRecurrentPPO."
            )

        lstm = self.policy.lstm_actor
        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        
        def create_rnn_tuple():
            h = torch.zeros(single_hidden_state_shape, device=self.device)
            if self.rnn_type == "lstm":
                c = torch.zeros(single_hidden_state_shape, device=self.device)
                return (h, c)
            return (h,)

        self._last_lstm_states = RNNStates(create_rnn_tuple(), create_rnn_tuple())

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = MaskableRecurrentRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MaskableRecurrentRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        
        lstm_states = deepcopy(self._last_lstm_states)

        with torch.no_grad():
            while n_steps < n_rollout_steps:
                if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    self.policy.reset_noise(env.num_envs)

                if is_masking_supported(env):
                    action_masks = get_action_masks(env)
                else:
                    action_masks = None
                
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
                episode_starts_tensor = torch.as_tensor(self._last_episode_starts).float().to(self.device)
                
                action_masks_tensor = None
                if action_masks is not None:
                     action_masks_tensor = torch.as_tensor(action_masks).to(self.device)

                actions, values, log_probs, lstm_states = self.policy.forward(
                    obs_tensor,
                    lstm_states,
                    episode_starts_tensor,
                    action_masks=action_masks_tensor,
                    deterministic=False
                )
                
                actions = actions.cpu().numpy()
                clipped_actions = actions
                if isinstance(self.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                new_obs, rewards, dones, infos = env.step(clipped_actions)
                self.num_timesteps += env.num_envs

                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

                self._update_info_buffer(infos)
                n_steps += 1

                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = actions.reshape(-1, 1)

                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with torch.no_grad():
                            def get_idx_state(states):
                                return tuple(s[:, idx : idx + 1, :].contiguous() for s in states)
                                
                            terminal_lstm_state = get_idx_state(lstm_states.vf)
                            episode_starts = torch.tensor([False], dtype=torch.float32, device=self.device)
                            terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                        rewards[idx] += self.gamma * terminal_value

                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    lstm_states=self._last_lstm_states,
                    action_masks=action_masks
                )
                
                self._last_obs = new_obs
                self._last_episode_starts = dones
                self._last_lstm_states = lstm_states

        with torch.no_grad():
            episode_starts = torch.tensor(dones, dtype=torch.float32, device=self.device)
            values = self.policy.predict_values(torch.as_tensor(new_obs).to(self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        
        # Optimization: move rollout buffer to GPU once
        self.rollout_buffer.to_device(self.device)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        imitation_losses = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                mask = rollout_data.mask > 1e-8
                action_masks = rollout_data.action_masks 

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                    action_masks=action_masks
                )
                
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.mean(torch.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                
                value_loss = torch.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob[mask])
                else:
                    entropy_loss = -torch.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                # --- IMITATION LOSS ---
                imitation_loss = torch.tensor(0.0, device=self.device)
                # Check relative steps since level start
                steps_in_level = self.num_timesteps - getattr(self, "_start_timesteps", 0)
                if self.imitation_buffer and steps_in_level < self.imitation_data_max_steps:
                    if np.random.rand() < (self.imitation_data_percentage / 100.0):
                        # Sample FULL sequence for state-preserving recurrent training
                        # This solves the "Amnesia" problem by carrying hidden states forward
                        expert_data = self.imitation_buffer.sample_full_episode()
                        
                        if expert_data:
                            if self._n_updates % 10 == 0:
                                logger.opt(colors=True).info(f"<yellow>Imitation sampling expert episode ({expert_data['observations'].shape[1]} steps). Level Step: {steps_in_level}/{self.imitation_data_max_steps}</yellow>")
                            expert_obs = torch.as_tensor(expert_data["observations"], device=self.device).float()
                            expert_acts = torch.as_tensor(expert_data["actions"], device=self.device).long()
                            expert_starts = torch.as_tensor(expert_data["episode_starts"], device=self.device).float()
                            expert_masks = expert_data["action_masks"]
                            if expert_masks is not None:
                                expert_masks = torch.as_tensor(expert_masks, device=self.device)
                                # BAND-AID: "Trust the Expert"
                                # If the expert took an action, it MUST be valid, even if our heuristic mask says otherwise.
                                # This fixes the "7 million loss" caused by strict masking in the translator.
                                expert_masks.scatter_(2, expert_acts.unsqueeze(2), True)

                            # Processing setup
                            # We process the full episode in chunks of 'seq_len' (Truncated BPTT)
                            seq_len = 64
                            T = expert_obs.shape[1] # Total steps in episode
                            n_expert_seq = 1 # We sample 1 full episode
                            
                            # Initialize Hidden States (Zeros at start of episode)
                            num_layers = self.policy.lstm_actor.num_layers
                            hidden_size = self.policy.lstm_actor.hidden_size
                            
                            h = torch.zeros((num_layers, n_expert_seq, hidden_size), device=self.device)
                            if self.rnn_type == "lstm":
                                c = torch.zeros((num_layers, n_expert_seq, hidden_size), device=self.device)
                                expert_lstm_states = RNNStates(pi=(h, c), vf=(h, c))
                            else:
                                expert_lstm_states = RNNStates(pi=(h,), vf=(h,))

                            accumulated_loss = torch.tensor(0.0, device=self.device)
                            n_chunks = 0

                            # Loop through episode in chunks
                            for t in range(0, T, seq_len):
                                end_t = min(t + seq_len, T)
                                current_seq_len = end_t - t
                                if current_seq_len <= 0: break

                                # Slices for this chunk
                                obs_chunk = expert_obs[:, t:end_t] # (1, seq_len, ...)
                                acts_chunk = expert_acts[:, t:end_t]
                                starts_chunk = expert_starts[:, t:end_t]
                                masks_chunk = expert_masks[:, t:end_t] if expert_masks is not None else None

                                # Flatten for API (Batch=1 * SeqLen)
                                obs_flat = obs_chunk.reshape(-1, *obs_chunk.shape[2:])
                                acts_flat = acts_chunk.reshape(-1)
                                starts_flat = starts_chunk.reshape(-1)
                                masks_flat = masks_chunk.reshape(-1, masks_chunk.shape[-1]) if masks_chunk is not None else None

                                # Forward Pass: Get Distribution AND New States
                                # We use get_distribution because evaluate_actions discards states
                                distribution, new_lstm_states = self.policy.get_distribution(
                                    obs_flat, expert_lstm_states, starts_flat, action_masks=masks_flat
                                )
                                
                                # Calculate Loss
                                expert_log_prob = distribution.log_prob(acts_flat)
                                chunk_loss = -torch.mean(expert_log_prob)
                                accumulated_loss += chunk_loss
                                n_chunks += 1

                                # Update States for Next Chunk (Detach to implement Truncated BPTT)
                                # We preserve the VALUES of the state (Memory), but cut the Gradient history.
                                def detach_state(s):
                                    if isinstance(s, torch.Tensor): return s.detach()
                                    return tuple(x.detach() for x in s)
                                
                                pi_state = detach_state(new_lstm_states.pi)
                                vf_state = detach_state(new_lstm_states.vf)
                                expert_lstm_states = RNNStates(pi=pi_state, vf=vf_state)

                            # Average the loss over the number of chunks to scale it independently of episode length
                            if n_chunks > 0:
                                imitation_loss = (accumulated_loss / n_chunks) * self.imitation_coef
                                imitation_losses.append(imitation_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + imitation_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(((torch.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        # Invalidate cache after training
        self.rollout_buffer.clear_vram_cache()

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if imitation_losses:
            self.logger.record("train/imitation_loss", np.mean(imitation_losses))

        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.policy.predict(observation, state, episode_start, deterministic, action_masks=action_masks)

    @classmethod
    def load(cls, path: str, env: Optional[GymEnv] = None, device: Union[torch.device, str] = "auto", custom_objects: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Overridden load to automatically handle architecture mismatches (e.g. Non-Recurrent -> Recurrent).
        This makes the class a drop-in replacement that fixes loading errors without pipeline changes.
        """
        try:
            # Try standard load first
            return super().load(path, env=env, device=device, custom_objects=custom_objects, **kwargs)
        except (RuntimeError, ValueError, TypeError) as e:
            # If it fails, retry with non-strict parameter loading and no optimizer
            logger.warning(f"Standard load failed, attempting architecture-agnostic recovery for MaskableRecurrentPPO: {e}")
            
            if custom_objects is None:
                custom_objects = {}
            
            # Signal SB3 to skip optimizer loading
            custom_objects["optimizer"] = None
            
            # Temporary monkeypatch to force non-strict loading
            original_set_params = cls.set_parameters
            def non_strict_set_params(self, params, exact_match=True, device=None):
                return original_set_params(self, params, exact_match=False, device=device)
            
            try:
                cls.set_parameters = non_strict_set_params
                return super().load(path, env=env, device=device, custom_objects=custom_objects, **kwargs)
            finally:
                cls.set_parameters = original_set_params