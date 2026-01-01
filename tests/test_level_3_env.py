import pytest
from unittest.mock import MagicMock, patch
from train.envs.level_3_env import Level3Env

@pytest.fixture
def mock_training_conf():
    return {
        "targets": 2,
        "grid_size": 32,
        "max_steps": 100,
        "rewards": {"success": 10}
    }

@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
def test_level3_initialization(mock_discover_maps, mock_config_manager, mock_training_conf):
    """
    Test that Level3Env initializes with correct parameters.
    """
    # Setup mocks
    mock_discover_maps.return_value = ["map_01"]
    
    # Configure ConfigManager mock
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {
        "env": {"targets": 2, "rewards": {"success": 10}}
    }
    mock_cm_instance.get_cumulative_active_layers.return_value = []
    
    # Init Env
    env = Level3Env(
        level_name="level_3",
        visualize=False,
        training_conf=mock_training_conf,
        max_steps=100
    )
    
    # Check targets
    assert env.targets == 2
    assert env.max_steps == 100
    assert env.level_name == "level_3"

@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
def test_level3_default_targets(mock_discover_maps, mock_config_manager):
    """
    Test that Level3Env defaults targets to 1 if not specified.
    """
    mock_discover_maps.return_value = ["map_01"]
    
    # Configure ConfigManager mock
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {}
    mock_cm_instance.get_cumulative_active_layers.return_value = []

    conf = {}
    env = Level3Env(level_name="level_3", training_conf=conf)
    assert env.targets == 1
