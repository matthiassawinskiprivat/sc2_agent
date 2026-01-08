import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from train.envs.capture_the_flag_env import CaptureTheFlagEnv

@pytest.mark.asyncio
async def test_ctf_env_asymmetric_setup():
    """Verify that bots are correctly initialized with trainee/opponent roles."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"), \
         patch("train.envs.capture_the_flag_env.CaptureTheFlagBot") as MockBot:
        
        # Setup mock bots
        bot1 = MockBot.return_value
        bot1.on_start = AsyncMock()
        bot2 = MagicMock()
        bot2.on_start = AsyncMock()
        MockBot.side_effect = [bot1, bot2]
        
        env = CaptureTheFlagEnv()
        
        # Mock model
        mock_model = MagicMock()
        env.set_model(mock_model)
        
        # Mocking methods called in _setup_game_session
        env._manager = MagicMock()
        client1 = MagicMock()
        client1._execute = AsyncMock()
        client2 = MagicMock()
        client2._execute = AsyncMock()
        env._manager.clients = [client1, client2]
        
        # Mock responses for game_info and game_data
        client1.get_game_data = AsyncMock()
        client2.get_game_data = AsyncMock()
        client1.debug_show_map = AsyncMock()
        client2.debug_show_map = AsyncMock()
        
        env._resolve_map = MagicMock(return_value="map")
        env._manager.create_join_game = AsyncMock()
        env._synchronized_step = AsyncMock()
        env._refresh_bot_states = AsyncMock()
        env._spawn_initial_units = AsyncMock()
        env._refresh_unit_lists = AsyncMock()
        
        # Mock internal sc2 helpers that are called during prepare_bot
        with patch("sc2.game_info.GameInfo"), \
             patch("sc2.player.Bot"):
            
            await env._setup_game_session()
            
            # Check roles
            assert bot1.is_trainee is True
            assert bot2.is_trainee is False
            
            # Check model snapshotting
            assert bot2.model == mock_model
            assert bot1.model is None # Trainee doesn't need internal model
