from functools import partial
from typing import Any, Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from src.agent.build_constants import (
    BUILD_MAP_RESOLUTION,
    BUILD_PLACEMENT_CELLS,
    COMMAND_VECTOR_DIM,
    MACRO_VECTOR_DIM,
    MAX_BUILD_WORKERS,
    WORKER_FEAT_DIM,
)
from src.agent.models.build_network import BuildNetwork


class BuildFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor wrapping BuildNetwork."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, **kwargs):
        super().__init__(observation_space, features_dim)
        map_channels = observation_space.spaces["local_map"].shape[0]
        self.map_res = int(kwargs.get("map_resolution", BUILD_MAP_RESOLUTION))
        self.thinking_steps = int(kwargs.get("thinking_steps", 3))
        self.net = BuildNetwork(
            map_channels=map_channels,
            worker_feat_dim=WORKER_FEAT_DIM,
            hidden_dim=features_dim,
            map_resolution=self.map_res,
        )
        self._worker_logits: torch.Tensor | None = None
        self._placement_logits: torch.Tensor | None = None

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        worker_features = observations["worker_features"]
        command_vec = observations["command"]
        local_map = observations["local_map"]

        batch = worker_features.shape[0]
        device = worker_features.device
        hidden_state = torch.zeros(batch, self.net.hidden_dim, device=device)

        unit_type_idx = (command_vec[:, 8] * 2000.0).long().clamp(0, 199)
        target_xy = command_vec[:, 9:11]

        _, placement_logits, hidden, worker_logits = self.net(
            worker_features=worker_features,
            command_type=unit_type_idx.unsqueeze(-1),
            command_target=target_xy,
            local_map=local_map,
            hidden_state=hidden_state,
            worker_mask=None,
        )
        self._worker_logits = worker_logits
        self._placement_logits = placement_logits
        return hidden


class BuildActorCriticPolicy(MaskableActorCriticPolicy):
    """Maskable PPO policy for build agent: worker pointer + placement grid."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Any,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("features_extractor_class", BuildFeaturesExtractor)
        fe_kw = dict(kwargs.get("features_extractor_kwargs") or {})
        fe_kw.setdefault("features_dim", 128)
        fe_kw.setdefault("map_resolution", BUILD_MAP_RESOLUTION)
        kwargs["features_extractor_kwargs"] = fe_kw
        kwargs.pop("spatial_resolution", None)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        if self.ortho_init:
            pass
        delattr(self, "action_net")

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        fe = self.features_extractor
        worker_logits = getattr(fe, "_worker_logits", None)
        placement_logits = getattr(fe, "_placement_logits", None)
        if worker_logits is None or placement_logits is None:
            n_workers = int(self.action_space.nvec[0])
            worker_logits = torch.zeros(
                latent_pi.shape[0], n_workers, device=latent_pi.device
            )
            placement_logits = torch.zeros(
                latent_pi.shape[0], BUILD_PLACEMENT_CELLS, device=latent_pi.device
            )
        action_logits = torch.cat([worker_logits, placement_logits], dim=-1)
        return self.action_dist.proba_distribution(action_logits=action_logits)
