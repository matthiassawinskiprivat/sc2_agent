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
async def test_level_3_reward_leakage_repro(mock_discover_maps, mock_config_manager, mock_training_conf):
    """
    REPRODUCTION: Test that rewards for unit 2 are lost if they occur during unit 1's turn.
    """
    mock_discover_maps.return_value = ["map_01"]
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {"env": {"targets": 2, "rewards": mock_training_conf["rewards"]}}
    mock_cm_instance.get_cumulative_active_layers.return_value = []
    mock_cm_instance.get_cumulative_action_heads.return_value = ["move"]
    mock_cm_instance.get_cumulative_categorical_heads.return_value = []

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
    
    # Mock observation generation
    mock_bot.get_observation_for_unit.return_value = (np.zeros((1, 32, 32)), dict())
    
    # Mock internal SC2 client
    env._client = MagicMock()
    env._client.actions = AsyncMock()
    env._client.step = AsyncMock()
    env._client.observation = AsyncMock()
    
    # Mock GameState
    with patch("sc2.game_state.GameState"), patch("train.envs.level_0_env.ProtoWrapper"):
        env._bot._prepare_step = MagicMock()
        
        # Scenario: 
        # Round 1: Unit 1 acts. Game steps. Unit 2 (moving from previous round) hits its beacon.
        # check_collections returns {102: 100.0}.
        
        mock_bot.get_active_zerglings.side_effect = [
            [unit1, unit2], # Start of Step 1
            [unit1],        # End of Step 1 (unit 2 killed/collected)
            [unit1],        # End of Step 1 termination check
            [unit1],        # Start of Step 2
            [unit1],        # End of Step 2
            [unit1],        # End of Step 2 termination check
        ]
        
        mock_bot.calculate_distance_reward.side_effect = [5.0, 3.0]
        # Unit 2 reward appears during Unit 1's turn
        mock_bot.check_collections = AsyncMock(side_effect=[{102: 100.0}, {}]) 
        
        env.active_unit_tags = [101, 102]
        env.current_unit_index = 0
        
        # --- Step 1: Unit 1 moves ---
        obs1, reward1, done1, truncated1, info1 = await env._async_step(action=1)
        
        # CURRENT BEHAVIOR: reward1 will be 4.9 (5.0 - 0.1). 
        # Unit 2's 100.0 reward is lost!
        print(f"Step 1 Reward: {reward1}")
        
        # --- Step 2: Unit 2's turn should be skipped because it's dead ---
        # Actually Level3Env will skip it automatically.
        
        # Verify if Unit 2 reward was captured (in current broken implementation it won't be)
        # We want to catch this failure.
        assert reward1 >= 104.9, f"Reward leakage! Expected at least 104.9, got {reward1}"
