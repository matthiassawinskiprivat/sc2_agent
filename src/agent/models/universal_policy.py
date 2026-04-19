from functools import partial
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Any, Optional
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from src.agent.models.spatial_decoder import SpatialDecoder, tactical_center_spatial_bias

class UniversalActorCriticPolicy(MaskableActorCriticPolicy):
    """
    A custom Maskable policy for the Universal architecture.
    Uses a spatial decoder for the shared 32x32 spatial head.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, 
                 lr_schedule: Any, *args, **kwargs):
        self.spatial_resolution = 32
        kwargs.pop("spatial_resolution", None)
        kwargs.pop("rnn_type", None)
        kwargs.pop("lstm_hidden_size", None)
        kwargs.pop("n_lstm_layers", None)
        kwargs.pop("shared_lstm", None)
        kwargs.pop("enable_critic_lstm", None)

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
        spatial_logits = spatial_logits + tactical_center_spatial_bias(
            self.spatial_resolution, latent_pi
        ).unsqueeze(0)
        cmd_logits = self.command_head(latent_pi)
        action_logits = torch.cat([cmd_logits, spatial_logits], dim=-1)
        return self.action_dist.proba_distribution(action_logits=action_logits)
