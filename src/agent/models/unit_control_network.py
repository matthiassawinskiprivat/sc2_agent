import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.components.reasoning_core import ReasoningCore
from src.utils.layer_constants import LayerConstants

class UnitControlNetwork(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Unit Control Agent.
    Includes:
    1. CNN Encoder (Spatial Processing)
    2. Reasoning Core (System 2 / Recurrent Depth)
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, thinking_steps: int = 3):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # 1. CNN Encoder (Standard NatureCNN-like but preserved spatial info longer)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute shape by doing a forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        
        # 2. Reasoning Core
        self.reasoning = ReasoningCore(input_dim=features_dim, hidden_dim=features_dim)
        self.thinking_steps = thinking_steps

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalization has been moved back to the ObservationTransformer (CPU)
        x = observations
        
        # 1. Encode
        x = self.cnn(x)
        x = self.linear(x)
        
        # 2. Reason (Loop x on itself)
        thought = x
        thought = self.reasoning(input_tensor=x, hidden_state=thought, thinking_steps=self.thinking_steps)
        
        return thought