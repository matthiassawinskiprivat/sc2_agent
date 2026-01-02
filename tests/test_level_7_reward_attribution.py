import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from train.envs.level_7_env import Level7Env

@pytest.fixture
def mock_training_conf():
    return {
        "targets": 2,
        "grid_size": 16,
        "max_steps": 100,
        "rewards": {
            "damage_reward": 0.1,
            "kill_reward": 10.0,
            "distance_multiplier": 1.0,
            "step_penalty": -0.1,
            "damage_penalty": -0.1,
            "kill_penalty": -10.0
        },
        "env": {
            "unit_type": "MARINE",
            "enemy_type": "ZERGLING"
        }
    }

class MockUnit:
    def __init__(self, tag):
        self.tag = tag

@pytest.mark.asyncio
@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
async def test_level_7_intermediate_reward_attribution(mock_discover_maps, mock_config_manager, mock_training_conf):
    """
    Test that Level7Env delivers individual rewards (combat + distance) for each unit.
    """
    mock_discover_maps.return_value = ["map_01"]
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {"env": mock_training_conf["env"], "rewards": mock_training_conf["rewards"]}
    mock_cm_instance.get_cumulative_active_layers.return_value = []

    # Initialize Env
    env = Level7Env(level_name="level_7", training_conf=mock_training_conf)
    
    # Mock Bot
    mock_bot = MagicMock()
    mock_bot.apply_action = AsyncMock()
    env._bot = mock_bot
    mock_bot.grid_size = 16
    mock_bot.allied_unit_type = 1 # Marine
    mock_bot.enemy_unit_type = 2 # Zergling
    
    # Setup 2 units
    unit1 = MockUnit(101)
    unit2 = MockUnit(102)
    mock_bot.get_active_units.return_value = [unit1, unit2]
    
    # Mock observation generation
    mock_bot.get_observation_for_unit.return_value = (np.zeros((1, 16, 16)), {})
    
    # Mock reward calculations
    mock_bot.calculate_distance_reward.side_effect = [5.0, 3.0] 
    mock_bot.calculate_combat_reward.side_effect = [2.0, 10.0] # Unit 1 did 2.0 damage, Unit 2 did 10.0 kill
    mock_bot.calculate_allied_attrition_penalty.side_effect = [-0.5, 0.0] # Unit 1 took 0.5 damage
    
    # Mock internal SC2 client
    env._client = MagicMock()
    env._client.actions = AsyncMock()
    env._client.step = AsyncMock()
    env._client.observation = AsyncMock()
    
    # Mock GameState
    with patch("sc2.game_state.GameState"), \
         patch("train.envs.level_0_env.ProtoWrapper"):
        # Mock state refresh
        env._bot._prepare_step = MagicMock()
        
        # Initialize active tags
        env.active_unit_tags = [101, 102]
        env.current_unit_index = 0
        
        # --- Step 1: Unit 1 acts ---
        obs1, reward1, done1, truncated1, info1 = await env._async_step(action=1)
        
        # Expected reward: 5.0 (dist) + 2.0 (combat) - 0.5 (attrition) - 0.1 (step) = 6.4
        assert reward1 != 0.0
        assert reward1 == pytest.approx(6.4)

        # --- Step 2: Unit 2 acts ---
        obs2, reward2, done2, truncated2, info2 = await env._async_step(action=1)
        
        # Expected reward: 3.0 (dist) + 10.0 (combat) + 0.0 (attrition) - 0.1 (step) = 12.9
        assert reward2 == pytest.approx(12.9)
