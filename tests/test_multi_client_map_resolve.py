import pytest
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from unittest.mock import MagicMock
import os
from pathlib import Path

def test_ctf_env_map_resolution():
    """Test that MultiClientEnv can resolve maps in the project directory."""
    level_name = "capture_the_flag"
    env = CaptureTheFlagEnv(level_name=level_name)
    
    # We expect the map '12_32x32_Full' to be found in train/maps/capture_the_flag/
    assert env.map_name == "12_32x32_Full"
    
    resolved_map = env._resolve_map(env.map_name)
    assert resolved_map is not None
    assert "12_32x32_Full.SC2Map" in str(resolved_map.path)
    assert Path(resolved_map.path).exists()

def test_ctf_env_attributes():
    """Test that spaces are correctly initialized."""
    env = CaptureTheFlagEnv()
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.observation_space.shape == (7, 128, 128)
