import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from train.envs.level_3_env import Level3Env
from src.level_3_bot import Level3Bot
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_level3_env_targets_propagation():
    """
    Test that Level3Env correctly propagates 'targets' from env config to Level3Bot.
    """
    # Mock ConfigManager to return targets: 2
    mock_config = {
        "env": {
            "targets": 2,
            "grid_size": 128,
            "rewards": {"success": 50.0}
        }
    }
    
    with patch('src.utils.config_manager.ConfigManager.get_level_config', return_value=mock_config), \
         patch('src.utils.config_manager.ConfigManager.get_cumulative_active_layers', return_value=[]):
        
        # training_conf usually comes from level_config["training"]
        training_conf = {"some_param": True} 
        
        env = Level3Env(level_name="level_3", training_conf=training_conf)
        
        # Check if targets was added to training_conf or handled elsewhere
        # If it's NOT in training_conf, Level3Bot won't see it when created in _setup_game_session
        assert env.training_conf.get("targets") == 2
