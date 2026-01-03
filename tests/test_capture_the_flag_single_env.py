
import pytest
import asyncio
import numpy as np
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from unittest.mock import MagicMock, AsyncMock
from sc2.ids.unit_typeid import UnitTypeId

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
    # Mock _execute for RequestGameInfo
    client._execute = AsyncMock(return_value=MagicMock(game_info=MagicMock()))
    client.get_game_data = AsyncMock()
    client.observation = AsyncMock(return_value=MagicMock(observation=MagicMock()))
    return client

def test_ctf_single_env_init():
    env = CaptureTheFlagSingleEnv()
    assert env.worker_index == 0
    assert env.observation_space.shape[0] == 23

def test_ctf_single_env_reset(mock_sc2_process, mock_client):
    env = CaptureTheFlagSingleEnv(
        sc2_process_cls=lambda **kwargs: mock_sc2_process,
        client_cls=lambda ws: mock_client
    )
    
    # Mock internal methods that interact with SC2
    env._setup_game_session = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value=np.zeros((23, 128, 128)))
    env._manager.launch_process = AsyncMock()
    
    obs, info = env.reset()
    
    assert obs is not None
    assert obs.shape == (23, 128, 128)
    env._setup_game_session.assert_called_once()
