import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, List, Any, Optional, Union, Type, Tuple
from src.agent.maskable_recurrent_ppo import MaskableRecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from src.agent.models.unit_control_network import UnitControlNetwork

class UniversalRecurrentPolicy(MaskableRecurrentActorCriticPolicy):
    """
    A custom Maskable Recurrent policy for the Universal architecture.
    Combines LSTM/GRU with the shared spatial head.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, 
                 lr_schedule: Any, *args, **kwargs):
        # Universal grid size (e.g. 32x32)
        self.spatial_resolution = 32
        # Remove from kwargs to avoid conflicts with parent __init__
        kwargs.pop("spatial_resolution", None)
        
        # Default lstm_hidden_size to 512 to match previous feature_dim
        if "lstm_hidden_size" not in kwargs:
            kwargs["lstm_hidden_size"] = 512
            
        # Ensure shared features extractor to match previous stages
        if "share_features_extractor" not in kwargs:
            kwargs["share_features_extractor"] = True
            
        # Force UnitControlNetwork for Dict spaces if not specified
        if "features_extractor_class" not in kwargs and isinstance(observation_space, gym.spaces.Dict):
            kwargs["features_extractor_class"] = UnitControlNetwork
            
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_action_net(self) -> None:
        """
        Overrides the action_net with a single flat linear layer.
        SB3's MultiDiscrete distribution will slice this automatically.
        The input features come from the LSTM/GRU output (self.lstm_output_dim).
        """
        # MultiDiscrete([n_commands, 1024]) requires sum([n_cmds, 1024]) logits
        total_logits = sum(self.action_space.nvec)
        self.action_net = nn.Linear(self.lstm_output_dim, total_logits)
        
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """
        Standard MultiDiscrete distribution creation.
        Note: MaskableRecurrentActorCriticPolicy uses MaskableDistribution 
        which handles MultiDiscrete correctly if configured in _setup_model.
        """
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def _disable_spatial_mask(self, action_masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Universal architecture shares the spatial head across commands, making a single
        spatial mask invalid. This ensures the spatial head is never masked.
        """
        if action_masks is None:
            return None
        # action_masks shape is [batch_size, n_cmds + n_spatial]
        n_cmds = self.action_space.nvec[0]
        # Force the spatial part of the mask (everything after n_cmds) to be True
        modified_masks = action_masks.clone()
        modified_masks[:, n_cmds:] = True
        return modified_masks

    def forward(self, obs, lstm_states, episode_starts, action_masks=None, deterministic=False):
        action_masks = self._disable_spatial_mask(action_masks)
        return super().forward(obs, lstm_states, episode_starts, action_masks, deterministic)

    def get_distribution(self, obs, lstm_states, episode_starts, action_masks=None):
        action_masks = self._disable_spatial_mask(action_masks)
        return super().get_distribution(obs, lstm_states, episode_starts, action_masks)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts, action_masks=None):
        action_masks = self._disable_spatial_mask(action_masks)
        return super().evaluate_actions(obs, actions, lstm_states, episode_starts, action_masks)
