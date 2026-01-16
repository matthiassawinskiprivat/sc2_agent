import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.components.reasoning_core import ReasoningCore

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
        
        # Memory State (Hidden State for GRU)
        # We need to manage this manually or let it reset?
        # SB3 doesn't natively pass hidden states to FeaturesExtractor unless using LSTM Policy.
        # For simplicity in this "Pseudo-Recurrent" setup (Depth Recurrence), 
        # we initialize hidden state from the CNN output itself (or zero).
        # Since ReasoningCore loops over the SAME input to refine it, 
        # we can treat the Initial Memory as "Zero" or "CNN Output".
        
        # Paper approach: "The L-module converges to a local equilibrium".
        # We can use the CNN output as the initial 'H' and loop it.
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. Encode
        x = self.cnn(observations)
        x = self.linear(x)
        
        # 2. Reason (Loop x on itself)
        # Initial State: We can start with x itself as the "thought".
        thought = x
        
        # We assume independent reasoning per step (stateless across time for now, 
        # to match SB3 non-LSTM interfaces easily). 
        # The recurrence is DEPTH-based (thinking hard about THIS frame).
        thought = self.reasoning(input_tensor=x, hidden_state=thought, thinking_steps=self.thinking_steps)
        
        return thought
