import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

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


def test_ctf_single_env_uses_precomputed_tactical_layers_without_config_walk():
    """Async/sync pipeline flattens cumulative lists onto training_conf for subproc workers."""
    precomputed = ["allied_unit_flying", "enemy_a_dps_base"]
    mock_cm = MagicMock()
    mock_cm.get_agent_architecture.return_value = "universal"
    mock_cm.get_cumulative_active_tactical_layers.side_effect = AssertionError(
        "subproc worker must not re-walk pipeline YAML when lists are precomputed"
    )

    with patch("train.envs.capture_the_flag_single_env.ConfigManager", return_value=mock_cm):
        env = CaptureTheFlagSingleEnv(
            level_name="ctf_level_04",
            training_conf={
                "config_dir": "config_universal",
                "agent_architecture": "universal",
                "active_layers": precomputed,
                "active_tactical_layers": precomputed,
                "action_heads": ["move", "attack"],
                "categorical_heads": [],
            },
        )

    assert env.active_tactical_layers == precomputed
    mock_cm.get_cumulative_active_tactical_layers.assert_not_called()
