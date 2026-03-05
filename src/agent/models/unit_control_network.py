import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Union, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.components.reasoning_core import ReasoningCore
from src.utils.layer_constants import LayerConstants

class UnitControlNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512, thinking_steps: int = 3,
                 agent_architecture: str = "original", action_heads: list = None, grid_size: int = 128, local_radius: int = 5):
        super().__init__(observation_space, features_dim)
        self.agent_architecture = agent_architecture
        
        # Branch based on space type
        if isinstance(observation_space, gym.spaces.Dict):
            map_space = observation_space.spaces["map"]
            self.command_dim = observation_space.spaces["command"].shape[0]
            self.is_dict_space = True
        else:
            map_space = observation_space
            self.command_dim = 0
            self.is_dict_space = False

        n_input_channels = map_space.shape[0]
        
        # 1. CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(map_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.command_dim, features_dim),
            nn.ReLU()
        )
        # ...

        # 2. Reasoning Core
        self.reasoning = ReasoningCore(input_dim=features_dim, hidden_dim=features_dim)
        self.thinking_steps = thinking_steps

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if self.is_dict_space:
            map_obs = observations["map"]
            command_obs = observations["command"]
        else:
            map_obs = observations
            command_obs = None

        # 1. Encode Spatial
        spatial_features = self.cnn(map_obs)

        # 2. Concatenate with High-Level Command if it exists
        if command_obs is not None:
            combined = torch.cat([spatial_features, command_obs], dim=1)
        else:
            combined = spatial_features

        # 3. Project to features_dim
        x = self.linear(combined)

        # 4. Reason
        thought = x
        thought = self.reasoning(input_tensor=x, hidden_state=thought, thinking_steps=self.thinking_steps)

        return thought