import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Union, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.components.reasoning_core import ReasoningCore


class UnitControlNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512, thinking_steps: int = 3,
                 agent_architecture: str = "original", action_heads: list = None, grid_size: int = 128, local_radius: int = 5):
        super().__init__(observation_space, features_dim)
        self.agent_architecture = agent_architecture

        if isinstance(observation_space, gym.spaces.Dict):
            map_space = observation_space.spaces["map"]
            self.command_dim = observation_space.spaces["command"].shape[0]
            self.is_dict_space = True
        else:
            map_space = observation_space
            self.command_dim = 0
            self.is_dict_space = False

        n_input_channels = map_space.shape[0]

        self.cnn_trunk = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
        )

        self.tactical_enabled = False
        n_tactical_channels = n_input_channels
        if self.agent_architecture == "universal" or (self.is_dict_space and "tactical" in observation_space.spaces):
            self.tactical_enabled = True
            if self.is_dict_space and "tactical" in observation_space.spaces:
                n_tactical_channels = observation_space.spaces["tactical"].shape[0]

        self.use_spatial_fusion = (self.agent_architecture == "universal") and self.tactical_enabled

        if self.tactical_enabled:
            if self.use_spatial_fusion:
                self.tactical_cnn_to16 = nn.Sequential(
                    nn.Conv2d(n_tactical_channels, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                )
            else:
                self.tactical_cnn_legacy = nn.Sequential(
                    nn.Conv2d(n_tactical_channels, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )

        if self.use_spatial_fusion:
            n_flatten = 0
        else:
            with torch.no_grad():
                _, h, w = map_space.shape
                sample = torch.zeros(1, n_input_channels, h, w, dtype=torch.float32)
                trunk_out = self.cnn_trunk(sample)
                n_flatten = trunk_out.flatten(1).shape[1]

        n_tactical_flatten = (64 * 8 * 8) if (self.tactical_enabled and not self.use_spatial_fusion) else 0

        if self.use_spatial_fusion:
            n_vector_in = 64 + 64 + self.command_dim
            self.fused_spatial_channels = 128
            self.linear = nn.Sequential(
                nn.Linear(n_vector_in, features_dim),
                nn.ReLU(),
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(n_flatten + n_tactical_flatten + self.command_dim, features_dim),
                nn.ReLU(),
            )

        self.reasoning = ReasoningCore(input_dim=features_dim, hidden_dim=features_dim)
        self.thinking_steps = thinking_steps
        self._spatial_fused: Optional[torch.Tensor] = None

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        self._spatial_fused = None

        if self.is_dict_space:
            map_obs = observations["map"]
            command_obs = observations["command"]
            tactical_obs = observations.get("tactical")
        else:
            map_obs = observations
            command_obs = None
            tactical_obs = None

        if self.use_spatial_fusion:
            global_16 = self.cnn_trunk(map_obs)
            batch_size = map_obs.shape[0]
            if tactical_obs is not None:
                t16 = self.tactical_cnn_to16(tactical_obs)
            else:
                t16 = torch.zeros(batch_size, 64, 16, 16, device=map_obs.device, dtype=map_obs.dtype)
            self._spatial_fused = torch.cat([global_16, t16], dim=1)
            gap_g = global_16.mean(dim=(2, 3))
            gap_t = t16.mean(dim=(2, 3))
            parts = [gap_g, gap_t]
            if command_obs is not None:
                parts.append(command_obs)
            combined = torch.cat(parts, dim=1)
        else:
            batch_size = map_obs.shape[0]
            global_features = torch.flatten(self.cnn_trunk(map_obs), 1)
            features_to_cat = [global_features]
            if self.tactical_enabled and not self.use_spatial_fusion:
                if tactical_obs is not None:
                    tactical_features = self.tactical_cnn_legacy(tactical_obs)
                else:
                    tactical_features = torch.zeros(
                        (batch_size, 4096), device=map_obs.device, dtype=map_obs.dtype
                    )
                features_to_cat.append(tactical_features)
            if command_obs is not None:
                features_to_cat.append(command_obs)
            combined = torch.cat(features_to_cat, dim=1)

        x = self.linear(combined)
        thought = self.reasoning(input_tensor=x, hidden_state=x, thinking_steps=self.thinking_steps)
        return thought
