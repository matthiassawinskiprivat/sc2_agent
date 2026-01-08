
import pytest
import asyncio
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from unittest.mock import MagicMock, AsyncMock
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

@pytest.fixture
def mock_sc2_process():
    process = MagicMock()
    controller = MagicMock()
    controller.create_game = AsyncMock()
    process.__aenter__ = AsyncMock(return_value=controller)
    process.__aexit__ = AsyncMock()
    process.ws = MagicMock()
    return process

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.join_game = AsyncMock()
    client.step = AsyncMock()
    client._execute = AsyncMock(return_value=MagicMock(game_info=MagicMock()))
    client.get_game_data = AsyncMock()
    client.observation = AsyncMock(return_value=MagicMock(observation=MagicMock()))
    client.debug_create_unit = AsyncMock()
    client.debug_set_unit_value = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_god_mode_spawning(mock_sc2_process, mock_client):
    env = CaptureTheFlagSingleEnv(
        sc2_process_cls=lambda **kwargs: mock_sc2_process,
        client_cls=lambda ws: mock_client
    )
    
    # Mock bot attributes that would be set during on_start
    env.bot = MagicMock()
    env.bot.client = mock_client
    env.bot.game_info = MagicMock()
    env.bot.game_info.playable_area = MagicMock(width=100, height=100)
    
    # Mock _refresh_bot_state to avoid GameState initialization errors
    env._refresh_bot_state = AsyncMock()
    
    await env._spawn_initial_units()
    
    # Verify debug_create_unit was called for both players (owner 1 and 2)
    # We expect several calls: Pylons (owner 1 & 2), Units (owner 1 & 2)
    calls = mock_client.debug_create_unit.call_args_list
    
    # Check for owner 1 and 2 in the arguments
    owners_seen = set()
    for call in calls:
        commands = call[0][0]
        for cmd in commands:
            owners_seen.add(cmd[3]) # cmd format: [unit_id, count, pos, owner]
            
    assert 1 in owners_seen
    assert 2 in owners_seen

@pytest.mark.asyncio
async def test_god_mode_commanding(mock_sc2_process, mock_client):
    env = CaptureTheFlagSingleEnv(
        sc2_process_cls=lambda **kwargs: mock_sc2_process,
        client_cls=lambda ws: mock_client
    )
    
    env.bot = MagicMock()
    env.bot.client = mock_client
    env.bot.apply_action = AsyncMock()
    
    # Mock player units
    env.player_units = {1: [101, 102], 2: [201, 202]}
    
    # Player 1 turn
    env.active_player_id = 1
    env.current_unit_index = 0
    await env._apply_action(12345) # spatial action
    env.bot.apply_action.assert_called_with(101, 12345)
    
    # Player 2 turn
    env.active_player_id = 2
    env.current_unit_index = 1
    await env._apply_action(54321)
    env.bot.apply_action.assert_called_with(202, 54321)
