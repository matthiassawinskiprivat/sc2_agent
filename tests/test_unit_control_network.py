import pytest
import torch
import gymnasium as gym
import numpy as np
from src.agent.models.unit_control_network import UnitControlNetwork

def test_unit_control_network_shapes():
    """Verify the custom network initializes and forwards correctly."""
    # 1. Setup
    obs_space = gym.spaces.Box(low=0, high=255, shape=(44, 128, 128), dtype=np.uint8)
    features_dim = 256
    model = UnitControlNetwork(obs_space, features_dim=features_dim, thinking_steps=3)
    
    # 2. Input
    batch_size = 4
    dummy_obs = torch.randn(batch_size, 44, 128, 128)
    
    # 3. Forward
    output = model(dummy_obs)
    
    # 4. Assert
    assert output.shape == (batch_size, features_dim)
    assert not torch.isnan(output).any()

def test_unit_control_network_reasoning_depth():
    """Verify thinking steps logic runs."""
    obs_space = gym.spaces.Box(low=0, high=255, shape=(10, 64, 64), dtype=np.uint8)
    # 1 Step
    model_1 = UnitControlNetwork(obs_space, thinking_steps=1)
    out_1 = model_1(torch.randn(1, 10, 64, 64))
    
    # 5 Steps
    model_5 = UnitControlNetwork(obs_space, thinking_steps=5)
    out_5 = model_5(torch.randn(1, 10, 64, 64))
    
    assert out_1.shape == out_5.shape
