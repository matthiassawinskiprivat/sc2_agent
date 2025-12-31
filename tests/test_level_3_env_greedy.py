import pytest
from unittest.mock import MagicMock, patch
from train.envs.level_3_env import Level3Env

@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
def test_level3_reward_integration(mock_discover_maps, mock_config_manager):
    mock_discover_maps.return_value = ["map_01"]
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {}
    mock_cm_instance.get_cumulative_active_layers.return_value = []

    env = Level3Env(level_name="level_3")
    env._bot = MagicMock()
    
    # Mock the new method in bot
    env._bot.calculate_distance_reward.return_value = 12.34
    
    reward = env._calculate_avg_distance_reward()
    
    assert reward == 12.34
    assert env._bot.calculate_distance_reward.called
