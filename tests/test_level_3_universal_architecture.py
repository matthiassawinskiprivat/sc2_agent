import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from gymnasium import spaces
from train.envs.level_3_env import Level3Env
from train.envs.level_0_env import Level0Env
from src.level_3_bot import Level3Bot

@pytest.fixture
def mock_training_conf():
    return {
        "agent_architecture": "universal",
        "active_layers": ["allied_unit_density", "enemy_unit_density", "terrain_pathable", "command_objective"],
        "action_heads": ["move_global", "move_local"],
        "level_name": "level_3",
        "env": {
            "targets": 2,
            "unit_type": "ZERGLING"
        }
    }

def test_level_3_env_universal_space(mock_training_conf):
    """Checks if the environment correctly sets up spaces for universal architecture."""
    with patch("train.envs.level_0_env.ConfigManager") as MockConfig, \
         patch.object(Level0Env, "_discover_maps", return_value=["test.SC2Map"]):
        instance = MockConfig.return_value
        instance.get_cumulative_active_layers.return_value = mock_training_conf["active_layers"]
        instance.get_cumulative_action_heads.return_value = mock_training_conf["action_heads"]
        instance.get_cumulative_categorical_heads.return_value = []
        instance.get_agent_architecture.return_value = "universal"
        instance.get_level_config.return_value = {"env": mock_training_conf["env"]}
        instance.get_cumulative_active_tactical_layers.return_value = mock_training_conf["active_layers"]

        env = Level3Env(training_conf=mock_training_conf)
        
        assert isinstance(env.observation_space, spaces.Dict)
        assert "map" in env.observation_space.spaces
        assert "tactical" in env.observation_space.spaces
        assert "command" in env.observation_space.spaces
        
        assert isinstance(env.action_space, spaces.MultiDiscrete)
        assert np.array_equal(env.action_space.nvec, [2, 1024])

def test_level_3_env_reset_observation_shape(mock_training_conf):
    """Checks if reset returns a dict observation for universal architecture."""
    with patch("train.envs.level_0_env.ConfigManager") as MockConfig, \
         patch("train.envs.level_0_env.SC2Process"), \
         patch("train.envs.level_0_env.Client"), \
         patch.object(Level0Env, "_discover_maps", return_value=["test.SC2Map"]):
        
        instance = MockConfig.return_value
        instance.get_cumulative_active_layers.return_value = mock_training_conf["active_layers"]
        instance.get_cumulative_action_heads.return_value = mock_training_conf["action_heads"]
        instance.get_cumulative_categorical_heads.return_value = []
        instance.get_agent_architecture.return_value = "universal"
        instance.get_level_config.return_value = {"env": mock_training_conf["env"]}
        instance.get_cumulative_active_tactical_layers.return_value = mock_training_conf["active_layers"]

        env = Level3Env(training_conf=mock_training_conf)
        
        # Mock bot
        mock_bot = MagicMock(spec=Level3Bot)
        mock_bot.get_active_zerglings.return_value = [MagicMock(tag=1), MagicMock(tag=2)]
        
        # Mock observation dict
        mock_obs = {
            "map": np.zeros((4, 128, 128)),
            "tactical": np.zeros((4, 32, 32)),
            "command": np.zeros(10)
        }
        mock_bot.get_observation_for_unit.return_value = (mock_obs, {})
        mock_bot._get_observation.return_value = (mock_obs, {})
        
        env._bot = mock_bot
        env._loop = MagicMock()
        
        # Mock super().reset() via patching the method in Level0Env
        with patch("train.envs.level_0_env.Level0Env.reset", return_value=(mock_obs, {})):
            obs, info = env.reset()
            
            assert isinstance(obs, dict)
            assert "map" in obs
            assert "tactical" in obs
            assert "command" in obs

def test_level_3_env_step_termination_obs_shape(mock_training_conf):
    """Verifies the bug where step returns np.zeros instead of a dict on termination."""
    with patch("train.envs.level_0_env.ConfigManager") as MockConfig, \
         patch.object(Level0Env, "_discover_maps", return_value=["test.SC2Map"]):
        instance = MockConfig.return_value
        instance.get_cumulative_active_layers.return_value = mock_training_conf["active_layers"]
        instance.get_cumulative_action_heads.return_value = mock_training_conf["action_heads"]
        instance.get_cumulative_categorical_heads.return_value = []
        instance.get_agent_architecture.return_value = "universal"
        instance.get_level_config.return_value = {"env": mock_training_conf["env"]}
        instance.get_cumulative_active_tactical_layers.return_value = mock_training_conf["active_layers"]

        env = Level3Env(training_conf=mock_training_conf)
        mock_bot = MagicMock(spec=Level3Bot)

        mock_beacon_units = MagicMock()
        mock_beacon_units.amount = 0
        mock_bot.all_units = MagicMock(return_value=mock_beacon_units)
        mock_bot.beacon_type = MagicMock()

        env._bot = mock_bot
        env._loop = MagicMock()
        
        env.active_unit_tags = []
        
        def mock_run(coro):
            import asyncio
            return asyncio.run(coro)
        
        env._loop.run_until_complete.side_effect = mock_run
        
        action = [0, 0]
        
        obs, reward, done, truncated, info = env.step(action)
        
        assert isinstance(obs, dict), f"Expected dict observation, got {type(obs)}"

if __name__ == "__main__":
    pytest.main([__file__])
