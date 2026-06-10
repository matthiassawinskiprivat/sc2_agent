
import pytest
import asyncio
from train.envs.single_client_env import SingleClientEnv
from unittest.mock import MagicMock, AsyncMock, patch

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


def test_single_client_env_reset_retries_after_startup_failure(mock_sc2_process, mock_client):
    env = SingleClientEnv(
        sc2_process_cls=lambda **kwargs: mock_sc2_process,
        client_cls=lambda **kwargs: mock_client,
        training_conf={"env": {"startup_max_retries": 3}},
    )
    env._get_joint_observation = AsyncMock(return_value={"obs": "data"})

    call_count = {"n": 0}

    async def flaky_setup():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("create_game failed")

    env._setup_game_session = flaky_setup
    env._manager.launch_process = AsyncMock()
    env.close = MagicMock()

    with patch("train.envs.single_client_env.time.sleep"):
        obs, info = env.reset()

    assert obs == {"obs": "data"}
    assert call_count["n"] == 2
    assert env._manager.launch_process.call_count == 2
    assert env.close.call_count == 1


def test_single_client_env_reset_raises_after_max_startup_retries(mock_sc2_process, mock_client):
    env = SingleClientEnv(
        sc2_process_cls=lambda **kwargs: mock_sc2_process,
        client_cls=lambda **kwargs: mock_client,
        training_conf={"env": {"startup_max_retries": 2}},
    )
    env._setup_game_session = AsyncMock(side_effect=RuntimeError("SC2 won't start"))
    env._manager.launch_process = AsyncMock()
    env.close = MagicMock()

    with patch("train.envs.single_client_env.time.sleep"):
        with pytest.raises(RuntimeError, match="SC2 won't start"):
            env.reset()

    assert env._manager.launch_process.call_count == 2
    assert env.close.call_count == 2
