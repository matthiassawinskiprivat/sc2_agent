import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from train.envs.level_drop_env import LevelDropEnv
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_level_drop_termination_total_annihilation():
    with patch('train.envs.level_0_env.ConfigManager') as mock_cm, \
         patch('sc2.game_state.GameState') as mock_gs:
        
        mock_cm.return_value.get_cumulative_active_layers.return_value = []
        mock_cm.return_value.get_cumulative_action_heads.return_value = ["move"]
        mock_cm.return_value.get_level_config.return_value = {"env": {"grid_size": 128}}
        
        env = LevelDropEnv(level_name="level_drop_0")
        
        bot = MagicMock()
        bot.apply_action = AsyncMock()
        env._bot = bot
        env._client = AsyncMock()
        env._loop = MagicMock()
        
        env.training_conf = {"rewards": {"win_reward": 50.0, "loss_penalty": -10.0}}
        
        # Scenario: 0 units, 1 structure
        bot.enemy_units.amount = 0
        bot.enemy_structures.amount = 1
        
        # Controlled units list
        mock_marine = MagicMock(tag=1)
        bot.get_active_units.return_value = [mock_marine]
        
        bot.calculate_combat_reward.return_value = 0.0
        bot.calculate_distance_reward.return_value = 0.0
        bot.calculate_allied_attrition_penalty.return_value = 0.0
        bot.calculate_drop_reward.return_value = 0.0
        bot.get_observation_for_unit.return_value = (np.zeros((1, 128, 128)), dict())
        
        env.active_unit_tags = [1]
        env.current_unit_index = 0
        
        obs, reward, done, truncated, info = await env._async_step(action=0)
        
        assert done is False, "Should not terminate if enemy structures still exist!"

@pytest.mark.asyncio
async def test_level_drop_termination_victory():
    """
    Verifies that LevelDropEnv terminates when EVERYTHING is destroyed.
    """
    with patch('train.envs.level_0_env.ConfigManager') as mock_cm, \
         patch('sc2.game_state.GameState') as mock_gs:
        
        mock_cm.return_value.get_cumulative_active_layers.return_value = []
        mock_cm.return_value.get_cumulative_action_heads.return_value = ["move"]
        mock_cm.return_value.get_level_config.return_value = {"env": {"grid_size": 128}}
        
        env = LevelDropEnv(level_name="level_drop_0")
        bot = MagicMock()
        bot.apply_action = AsyncMock()
        env._bot = bot
        env._client = AsyncMock()
        
        env.training_conf = {"rewards": {"win_reward": 50.0, "loss_penalty": -10.0}}
        
        # Scenario: 0 units, 0 structures
        bot.enemy_units.amount = 0
        bot.enemy_structures.amount = 0
        
        mock_marine = MagicMock(tag=1)
        bot.get_active_units.return_value = [mock_marine]
        
        bot.calculate_combat_reward.return_value = 0.0
        bot.calculate_distance_reward.return_value = 0.0
        bot.calculate_allied_attrition_penalty.return_value = 0.0
        bot.calculate_drop_reward.return_value = 0.0
        bot.get_observation_for_unit.return_value = (np.zeros((1, 128, 128)), dict())
        
        env.active_unit_tags = [1]
        env.current_unit_index = 0
        
        obs, reward, done, truncated, info = await env._async_step(action=0)
        
        assert done is True, "Should terminate when all enemy units and structures are gone."
        # Reward is win_reward (50) + step_penalty (-0.05)
        assert reward >= 49.0