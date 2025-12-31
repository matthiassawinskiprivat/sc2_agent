import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from train.envs.level_3_env import Level3Env
from sc2.ids.unit_typeid import UnitTypeId

@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
def test_level3_reward_flow(mock_discover_maps, mock_config_manager):
    # Setup mocks
    mock_discover_maps.return_value = ["map_01"]
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {
        "env": {"targets": 2, "rewards": {"success": 50, "step_penalty": -0.1}}
    }
    mock_cm_instance.get_cumulative_active_layers.return_value = []
    
    # Init env
    env = Level3Env(level_name="level_3")
    
    # Mock bot
    env._bot = MagicMock()
    env._bot.apply_action = AsyncMock()
    env._bot.get_observation_for_unit.return_value = (np.zeros(env.observation_space.shape), {})
    env._bot.game_info = MagicMock()
    
    env._client = AsyncMock()
    env._client.observation = AsyncMock(return_value=MagicMock())

    # Mock active units
    z1 = MagicMock(tag=101)
    z2 = MagicMock(tag=102)
    env._bot.get_active_zerglings.return_value = [z1, z2]
    env.active_unit_tags = [101, 102]
    env.current_unit_index = 0

    # Mock bot.calculate_distance_reward
    env._bot.calculate_distance_reward.return_value = 5.0
    # Mock bot.check_collections
    env._bot.check_collections = AsyncMock(return_value=25.0)

    # Patch GameState and ProtoWrapper to avoid crashing on mock data
    with patch("sc2.game_state.GameState"), \
         patch("train.envs.level_3_env.ProtoWrapper"):
        
        # Run one cycle for Unit 101
        loop = asyncio.new_event_loop()
        obs, reward, done, truncated, info = loop.run_until_complete(env._async_step(123))
        
        # First unit action shouldn't step the game yet
        assert reward == 0.0
        assert env.current_unit_index == 1
        assert not env._client.step.called
        
        # Run second cycle for Unit 102
        obs, reward, done, truncated, info = loop.run_until_complete(env._async_step(456))
        
        # Second unit action should step the game and return reward
        assert env._client.step.called
        # Reward = collection_reward (25.0) + dist_reward (5.0) = 30.0
        assert reward == 30.0
        assert env.current_unit_index == 0