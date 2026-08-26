"""Maskable PPO policy for the learned manager agent."""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from src.agent.manager_constants import (
    MANAGER_MACRO_VECTOR_DIM,
    MANAGER_MAP_RESOLUTION,
)
from src.agent.models.manager_network import ManagerNetwork


class ManagerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, **kwargs):
        super().__init__(observation_space, features_dim)
        map_channels = observation_space.spaces["map"].shape[0]
        macro_dim = observation_space.spaces["macro_vector"].shape[0]
        map_res = int(kwargs.get("map_resolution", MANAGER_MAP_RESOLUTION))
        self.net = ManagerNetwork(
            map_channels=map_channels,
            macro_dim=macro_dim,
            hidden_dim=features_dim,
            map_resolution=map_res,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(observations["map"], observations["macro_vector"])


class ManagerActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Any,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("features_extractor_class", ManagerFeaturesExtractor)
        fe_kw = dict(kwargs.get("features_extractor_kwargs") or {})
        fe_kw.setdefault("features_dim", 128)
        fe_kw.setdefault("map_resolution", MANAGER_MAP_RESOLUTION)
        kwargs["features_extractor_kwargs"] = fe_kw
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
