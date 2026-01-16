import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from train.envs.level_drop_env import LevelDropEnv
from src.level_drop_bot import LevelDropBot

def test_level_drop_env_heads_accumulation():
    """
    Verify that LevelDropEnv accumulates action heads from Level 0 and Level 7.
    """
    # Grid size 10 -> grid_area 100
    # Expected heads: move (lvl0), attack (lvl7), load, unload (lvl_drop)
    # Total = 4 heads * 100 = 400 actions
    
    # We need to mock the ConfigManager to return the accumulated list
    # since we are testing the Env's initialization logic.
    mock_cm = MagicMock()
    mock_cm.get_cumulative_active_layers.return_value = []
    mock_cm.get_cumulative_action_heads.return_value = ["move", "attack", "load", "unload"]
    mock_cm.get_level_config.return_value = {"env": {"rewards": {}}}
    
    from sc2.data import Race
    mock_game_info = MagicMock()
    mock_game_info.player_races = {1: Race.Terran, 2: Race.Zerg}

    with patch('train.envs.level_0_env.ConfigManager', return_value=mock_cm), \
         patch('train.envs.level_0_env.SC2Process'), \
         patch('train.envs.level_0_env.Client'), \
         patch('sc2.game_info.GameInfo', return_value=mock_game_info), \
         patch('train.envs.level_0_env.Level0Env._discover_maps', return_value=['map1']):
        
        env = LevelDropEnv(grid_size=10, level_name="level_drop")
        
    assert env.action_space.n == 400
    assert env.action_heads == ["move", "attack", "load", "unload"]

@pytest.mark.asyncio
async def test_level_drop_bot_action_mapping():
    """
    Verify that LevelDropBot correctly maps actions based on the cumulative heads.
    """
    training_conf = {
        "env": {"action_heads": ["move", "attack", "load", "unload"]}
    }
    bot = LevelDropBot(conn=MagicMock(), training_conf=training_conf)
    bot.grid_size = 10
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (5, 5)
    
    unit = MagicMock()
    # Mock cargo_used for unload test
    unit.cargo_used = 1
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = unit
    bot.do = MagicMock()
    
    # Action 50 -> head 0 (move)
    await bot.apply_action(1, 50)
    unit.move.assert_called_once()
    
    # Action 150 -> head 1 (attack)
    unit.reset_mock()
    await bot.apply_action(1, 150)
    unit.attack.assert_called_once()
    
    # Action 250 -> head 2 (load)
    unit.reset_mock()
    # Mock no transport nearby for simple move fallback test
    bot.units.filter.return_value = []
    await bot.apply_action(1, 250)
    # LevelDropBot 'load' logic falls back to move if no transport found
    unit.move.assert_called_once()
    
    # Action 350 -> head 3 (unload)
    unit.reset_mock()
    from sc2.ids.ability_id import AbilityId
    await bot.apply_action(1, 350)
    unit.assert_called_with(AbilityId.UNLOADALL)