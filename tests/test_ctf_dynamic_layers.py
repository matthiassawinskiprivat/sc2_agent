import pytest
import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from src.utils.layer_constants import LayerConstants

def test_ctf_env_accumulates_layers():
    # We use a level name from the pipeline
    env = CaptureTheFlagEnv(level_name="capture_the_flag_0")
    
    # Check if observation space matches the accumulated layers count
    expected_layers = env.config_manager.get_cumulative_active_layers("capture_the_flag_0")
    
    # Verify that the environment is using the ConfigManager's output
    assert env.active_layers == expected_layers
    assert env.observation_space.shape[0] == len(expected_layers)
    
    # Verify that it's NOT using the old hardcoded base_layers count
    assert not hasattr(env, "base_layers")

def test_ctf_single_env_accumulates_layers():
    env = CaptureTheFlagSingleEnv(level_name="capture_the_flag_0")
    expected_layers = env.config_manager.get_cumulative_active_layers("capture_the_flag_0")
    
    assert env.active_layers == expected_layers
    assert env.observation_space.shape[0] == len(expected_layers)
    assert not hasattr(env, "base_layers")
