import pytest
from train.envs.level_7_env import Level7Env
from src.level_7_bot import Level7Bot
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

def test_level_7_env_action_space_is_doubled():
    # Grid size 16 -> 16*16 = 256. Level 7 -> 2 * 256 = 512.
    mock_cm = MagicMock()
    mock_cm.get_cumulative_active_layers.return_value = []
    mock_cm.get_cumulative_action_heads.return_value = ["move", "attack"]
    
    with patch('train.envs.level_0_env.ConfigManager', return_value=mock_cm), \
         patch('train.envs.level_0_env.Level0Env._discover_maps', return_value=['map1']):
        env = Level7Env(grid_size=16, level_name="level_7")
    
    assert env.action_space.n == 512

@pytest.mark.asyncio
async def test_level_7_bot_apply_action_splitting():
    # Verify that the bot correctly splits actions into move/attack
    training_conf = {
        "targets": 1,
        "env": {"action_heads": ["move", "attack"]}
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 10 # 100 area
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (10, 10)
    bot.units = MagicMock()
    unit = MagicMock()
    bot.units.find_by_tag.return_value = unit
    
    # Action 50 -> Move
    bot.do = MagicMock()
    await bot.apply_action(1, 50)
    unit.move.assert_called_once()
    unit.attack.assert_not_called()
    
    # Action 150 -> Attack
    unit.reset_mock()
    await bot.apply_action(1, 150)
    unit.attack.assert_called_once()
    unit.move.assert_not_called()

def test_level_7_bot_reward_logic():
    training_conf = {
        "rewards": {"damage_reward": 0.1, "kill_reward": 10.0}
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
    
    # Damage test
    enemy = MagicMock(tag=1, health=40, shield=0, health_max=50, shield_max=0)
    bot.enemy_units = MagicMock(return_value=[enemy])
    bot.previous_health = {1: 50}
    
    reward = bot.calculate_combat_reward()
    assert reward == 1.0 # (50-40) * 0.1
    assert bot.previous_health[1] == 40
    
    # Kill test
    bot.enemy_units = MagicMock(return_value=[]) # Enemy gone
    reward = bot.calculate_combat_reward()
    assert reward == 10.0
    assert 1 not in bot.previous_health