import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from train.envs.multi_client_env import MultiClientEnv
from src.utils.multi_client_manager import MultiClientSC2Manager

class MockManager:
    def __init__(self):
        self.launch_processes = AsyncMock()
        self.create_join_game = AsyncMock()
        self.close = AsyncMock()
        self.host_client = AsyncMock()
        self.join_client = AsyncMock()
        self.clients = [self.host_client, self.join_client]

def test_multi_client_env_reset():
    """Test environment reset launches and starts game (Sync test)."""
    with patch('train.envs.multi_client_env.MultiClientSC2Manager', return_value=MockManager()) as mock_cls:
        env = MultiClientEnv(map_name="TestMap")
        
        # Mock _launch_sc2_processes and _setup_game_session to avoid real loop execution issues if possible
        # But wait, env.reset() creates a NEW loop.
        # If we run this test synchronously (no @pytest.mark.asyncio), there is no running loop.
        
        # We need to mock the coroutines that reset calls
        with patch.object(env, '_launch_sc2_processes', new_callable=AsyncMock) as mock_launch:
             with patch.object(env, '_setup_game_session', new_callable=AsyncMock) as mock_setup:
                  with patch.object(env, '_get_joint_observation', new_callable=AsyncMock) as mock_obs:
                        mock_obs.return_value = [0, 0]
                        
                        obs, info = env.reset()
                        
                        assert mock_launch.called
                        assert mock_setup.called
                        assert mock_obs.called

def test_multi_client_env_step():
    """Test synchronized stepping (Sync test)."""
    with patch('train.envs.multi_client_env.MultiClientSC2Manager', return_value=MockManager()) as mock_cls:
        env = MultiClientEnv(map_name="TestMap")
        
        # We need to initialize the loop manually or call reset first
        # But reset might be heavy with mocks.
        # Let's just mock the loop directly for step test.
        env._loop = MagicMock()
        env._loop.run_until_complete = MagicMock(return_value=([0], [0], [False], [False], {}))
        
        obs, reward, done, truncated, info = env.step([1, 2])
        
        assert env._loop.run_until_complete.called