import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from train.envs.level_3_env import Level3Env

@pytest.fixture
def mock_training_conf():
    return {
        "targets": 2,
        "grid_size": 32,
        "max_steps": 100,
        "rewards": {
            "success": 100.0,
            "distance_multiplier": 1.0,
            "step_penalty": -0.1
        },
        "active_layers": []
    }

class MockUnit:
    def __init__(self, tag):
        self.tag = tag

@pytest.mark.asyncio
@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
async def test_level_3_intermediate_reward_attribution(mock_discover_maps, mock_config_manager, mock_training_conf):
    """
    Test that Level3Env delivers rewards for each unit in the cycle, not just at the end.
    This test verifies individual rewards are implemented.
    """
    mock_discover_maps.return_value = ["map_01"]
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {"env": {"targets": 2, "rewards": mock_training_conf["rewards"]}}
    mock_cm_instance.get_cumulative_active_layers.return_value = []

    # Initialize Env
    env = Level3Env(level_name="level_3", training_conf=mock_training_conf)
    
    # Mock Bot
    mock_bot = MagicMock()
    mock_bot.apply_action = AsyncMock()
    env._bot = mock_bot
    mock_bot.grid_size = 32
    
    # Setup 2 units
    unit1 = MockUnit(101)
    unit2 = MockUnit(102)
    mock_bot.get_active_zerglings.return_value = [unit1, unit2]
    
    # Mock observation generation
    mock_bot.get_observation_for_unit.return_value = (np.zeros((1, 32, 32)), {})
    
    # Mock reward calculation
    mock_bot.calculate_distance_reward.side_effect = [5.0, 3.0] # Reward for unit 1 and unit 2
    mock_bot.check_collections = AsyncMock(side_effect=[{}, {102: 10.0}]) # Collection reward for unit 2 only
    
    # Mock internal SC2 client
    env._client = MagicMock()
    env._client.actions = AsyncMock()
    env._client.step = AsyncMock()
    env._client.observation = AsyncMock()
    
    # Mock GameState
    with patch("sc2.game_state.GameState") as mock_game_state:
        # Mock state refresh
        env._bot._prepare_step = MagicMock()
        
        # Initialize active tags
        env.active_unit_tags = [101, 102]
        env.current_unit_index = 0
        
        # --- Step 1: Unit 1 moves ---
        obs1, reward1, done1, truncated1, info1 = await env._async_step(action=1)
        
        assert reward1 == 5.0 - 0.1, f"Expected 4.9, got {reward1}"

        # --- Step 2: Unit 2 moves ---
        obs2, reward2, done2, truncated2, info2 = await env._async_step(action=1)
        
        assert reward2 == 3.0 + 10.0 - 0.1, f"Expected 12.9, got {reward2}"