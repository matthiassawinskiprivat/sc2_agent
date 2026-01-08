
import pytest
import asyncio
from train.envs.single_client_env import SingleClientEnv
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_sc2_process():
    process = MagicMock()
    process.__aenter__ = AsyncMock(return_value=MagicMock())
    process.__aexit__ = AsyncMock()
    process.ws = MagicMock()
    return process

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.join_game = AsyncMock()
    client.step = AsyncMock()
    return client

def test_single_client_env_init():
    # This should fail because SingleClientEnv doesn't exist yet
    env = SingleClientEnv()
    assert env.worker_index == 0

def test_single_client_env_reset(mock_sc2_process, mock_client):
    # Mocking dependencies
    env = SingleClientEnv(
        sc2_process_cls=lambda **kwargs: mock_sc2_process,
        client_cls=lambda **kwargs: mock_client
    )
    
    # We need to mock _setup_game_session and _get_joint_observation 
    # as they are usually implemented by subclasses
    env._setup_game_session = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value={"obs": "data"})
    
    # Mock launch_process on manager
    env._manager.launch_process = AsyncMock()
    
    # We need to mock run_until_complete on the loop to just return the coro result
    # since we can't easily run it while pytest-asyncio is active.
    # Alternatively, just don't use pytest.mark.asyncio for this test.
    
    obs, info = env.reset()
    
    assert obs == {"obs": "data"}
    env._setup_game_session.assert_called_once()
    env._manager.launch_process.assert_called_once()
