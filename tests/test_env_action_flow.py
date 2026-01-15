import pytest
import numpy as np
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from train.envs.level_0_env import Level0Env
from sc2.position import Point2

@pytest.mark.asyncio
async def test_level0_env_action_transmission():
    """
    Test that Level0Env actually sends actions to the client.
    """
    # Mocking necessary components
    with patch('train.envs.level_0_env.SC2Process'), \
         patch('train.envs.level_0_env.Client'), \
         patch('train.envs.level_0_env.SC2Bot') as MockBotClass:
        
        # Setup mock bot
        mock_bot = MockBotClass.return_value
        mock_bot.apply_action = AsyncMock()
        mock_bot._actions = None # Force fallback to 'actions'
        mock_bot.actions = []
        mock_bot.zergling = MagicMock()
        mock_bot.agent_system.unit_agents = {
            "agent_0": MagicMock()
        }
        mock_bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (10, 20)
        mock_bot.game_info._proto = MagicMock()

        # Setup mock client
        mock_client = AsyncMock()

        # Mock GameState to avoid sc2 library internal errors
        with patch('sc2.game_state.GameState'):
            # Initialize env
            env = Level0Env()
            env._bot = mock_bot
            env._client = mock_client
            env._loop = asyncio.get_event_loop()

            # Define a mock action
            action = 123
            mock_cmd = "move_command"

            # Mock apply_action to simulate real behavior for this test
            async def mock_apply_action(tag, act):
                mock_bot.actions.append(mock_cmd)
            mock_bot.apply_action.side_effect = mock_apply_action

            # Execute _async_step
            await env._async_step(action)

            # Verify apply_action called
            mock_bot.apply_action.assert_called_once_with(mock_bot.zergling.tag, action)
            
            # VERIFY: client.actions should have been called with the actions list
            mock_client.actions.assert_called_once()
            args, _ = mock_client.actions.call_args
            assert mock_cmd in args[0]
            
            # VERIFY: actions should be cleared after transmission
            assert len(mock_bot.actions) == 0
