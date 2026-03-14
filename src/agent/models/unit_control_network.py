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
        
        # 1. Global CNN Encoder
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

        # 1.1 Tactical CNN Encoder (Zoomed 32x32 centered on active unit)
        self.tactical_enabled = False
        n_tactical_flatten = 0
        
        # Universal architecture ALWAYS expects a tactical stream
        if self.agent_architecture == "universal" or (self.is_dict_space and "tactical" in observation_space.spaces):
            self.tactical_enabled = True
            
            # Channel count: either from space, or default to n_input_channels
            if self.is_dict_space and "tactical" in observation_space.spaces:
                n_tactical_channels = observation_space.spaces["tactical"].shape[0]
            else:
                n_tactical_channels = n_input_channels
                
            self.tactical_cnn = nn.Sequential(
                nn.Conv2d(n_tactical_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16x16
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 8x8
                nn.ReLU(),
                nn.Flatten()
            )
            # Tactical output is always 64 * 8 * 8 = 4096
            n_tactical_flatten = 4096

        # Final Feature Projection
        # n_flatten (9216) + n_tactical_flatten (4096) + command (10) = 13322
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_tactical_flatten + self.command_dim, features_dim),
            nn.ReLU()
        )

        # 2. Reasoning Core
        self.reasoning = ReasoningCore(input_dim=features_dim, hidden_dim=features_dim)
        self.thinking_steps = thinking_steps

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if self.is_dict_space:
            map_obs = observations["map"]
            command_obs = observations["command"]
            tactical_obs = observations.get("tactical")
        else:
            map_obs = observations
            command_obs = None
            tactical_obs = None

        # 1. Encode Global View
        global_features = self.cnn(map_obs)

        # 1.1 Encode Tactical View
        if self.tactical_enabled:
            if tactical_obs is not None:
                tactical_features = self.tactical_cnn(tactical_obs)
            else:
                # Fallback to zeros if tactical is missing but expected
                batch_size = map_obs.shape[0]
                tactical_features = torch.zeros((batch_size, 4096), device=map_obs.device)
        else:
            tactical_features = None

        # 2. Fusion: Concatenate Global + Tactical + Command
        features_to_cat = [global_features]
        if tactical_features is not None:
            features_to_cat.append(tactical_features)
        if command_obs is not None:
            features_to_cat.append(command_obs)

        combined = torch.cat(features_to_cat, dim=1)

        # 3. Project to Shared Latent Space
        x = self.linear(combined)

        # 4. Iterative Reasoning
        thought = x
        thought = self.reasoning(input_tensor=x, hidden_state=thought, thinking_steps=self.thinking_steps)

        return thought
