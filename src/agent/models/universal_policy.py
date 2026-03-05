import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, List, Any, Optional, Union, Type, Tuple
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class UniversalActorCriticPolicy(MaskableActorCriticPolicy):
    """
    A custom Maskable policy for the Universal architecture.
    Uses a shared spatial head for all command types.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, 
                 lr_schedule: Any, *args, **kwargs):
        # Universal grid size (e.g. 16x16)
        self.spatial_resolution = kwargs.pop("spatial_resolution", 16)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_action_net(self) -> None:
        """
        Overrides the action_net with a single flat linear layer.
        SB3's MultiDiscrete distribution will slice this automatically.
        """
        # MultiDiscrete([n_commands, 256]) requires sum([n_cmds, 256]) logits
        total_logits = sum(self.action_space.nvec)
        self.action_net = nn.Linear(self.features_dim, total_logits)
        
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """
        Standard MultiDiscrete distribution creation.
        """
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)
