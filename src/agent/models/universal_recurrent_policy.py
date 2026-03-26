from functools import partial
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, List, Any, Optional, Union, Tuple
from stable_baselines3.common.type_aliases import Schedule
from src.agent.maskable_recurrent_ppo import MaskableRecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.models.spatial_decoder import SpatialDecoder

class UniversalRecurrentPolicy(MaskableRecurrentActorCriticPolicy):
    """
    A custom Maskable Recurrent policy for the Universal architecture.
    Combines LSTM/GRU with a spatial decoder for the shared 32x32 head.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, 
                 lr_schedule: Any, *args, **kwargs):
        # Universal grid size (e.g. 32x32)
        self.spatial_resolution = 32
        kwargs.pop("spatial_resolution", None)

        if "lstm_hidden_size" not in kwargs:
            kwargs["lstm_hidden_size"] = 512

        if "share_features_extractor" not in kwargs:
            kwargs["share_features_extractor"] = True

        if "features_extractor_class" not in kwargs and isinstance(observation_space, gym.spaces.Dict):
            kwargs["features_extractor_class"] = UnitControlNetwork

        fe_kw = dict(kwargs.get("features_extractor_kwargs") or {})
        fe_kw["agent_architecture"] = "universal"
        kwargs["features_extractor_kwargs"] = fe_kw

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        n_cmds = int(self.action_space.nvec[0])
        assert int(self.action_space.nvec[1]) == self.spatial_resolution ** 2
        self.command_head = nn.Linear(latent_dim_pi, n_cmds)
        self.spatial_decoder = SpatialDecoder(
            latent_dim_pi,
            fused_channels=128,
            spatial_resolution=self.spatial_resolution,
        )
        delattr(self, "action_net")
        if self.ortho_init:
            self.command_head.apply(partial(self.init_weights, gain=0.01))
            self.spatial_decoder.apply(partial(self.init_weights, gain=0.01))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        fe = self.features_extractor
        fused = getattr(fe, "_spatial_fused", None)
        if fused is None:
            fused = torch.zeros(
                latent_pi.shape[0], 128, 16, 16,
                device=latent_pi.device, dtype=latent_pi.dtype,
            )
        spatial_logits = self.spatial_decoder(latent_pi, fused)
        cmd_logits = self.command_head(latent_pi)
        action_logits = torch.cat([cmd_logits, spatial_logits], dim=-1)
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
