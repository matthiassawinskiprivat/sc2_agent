import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from train.envs.level_3_env import Level3Env
from sc2.ids.unit_typeid import UnitTypeId
import numpy as np

def test_sequential_control():
    asyncio.run(_test_sequential_control_impl())

async def _test_sequential_control_impl():
    # Setup
    env = Level3Env(training_conf={"targets": 2})
    env._bot = MagicMock() # Level3Bot
    env._client = AsyncMock()
    
    # Mock bot methods
    env._bot.get_active_zerglings.return_value = [
        MagicMock(tag=101),
        MagicMock(tag=102)
    ]
    env._bot.apply_action = AsyncMock()
    env._bot.check_collections = AsyncMock(return_value=1.0)
    env._bot.get_observation_for_unit.return_value = (np.zeros((1, 64, 64)), {})
    
    # Mock units count for done condition
    # units(TYPE).amount
    mock_units = MagicMock()
    mock_units.amount = 1 # Not 0, so not done
    env._bot.units.return_value = mock_units
    
    env._bot.previous_distances = {}
    
    # Mock distance reward calc
    env._calculate_avg_distance_reward = MagicMock(return_value=0.5)
    
    # Mock bot _prepare_step to avoid needing real state
    env._bot._prepare_step = MagicMock()
    
    # Patch GameState to avoid protobuf parsing errors
    with patch("sc2.game_state.GameState") as MockGameState, \
         patch("train.envs.level_0_env.Level0Env.reset", return_value=(np.zeros((1, 64, 64)), {})):
        env.reset()
        
        # We need to make sure GameState patch is active during _async_step too.
        # So we should put the loop inside the with block.
        
        assert env.current_unit_index == 0
        assert env.active_unit_tags == [101, 102]
        
        # 2. Step 1 (Unit 101)
        action1 = 123
        obs, reward, done, truncated, info = await env._async_step(action1)
        
        # Verify action applied to Unit 101
        env._bot.apply_action.assert_called_with(101, action1)
        # Verify index advanced
        assert env.current_unit_index == 1
        # Verify NO physics step
        assert not env._client.step.called
        assert reward == 0.0
        
        # 3. Step 2 (Unit 102)
        action2 = 456
        obs, reward, done, truncated, info = await env._async_step(action2)
        
        # Verify action applied to Unit 102
        env._bot.apply_action.assert_called_with(102, action2)
        # Verify index reset
        assert env.current_unit_index == 0
        # Verify physics step
        assert env._client.step.called
        
        # Verify total reward = collection (1.0) + distance (0.5) = 1.5
        assert reward == 1.5
