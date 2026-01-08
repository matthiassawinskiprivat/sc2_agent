
import pytest
import asyncio
import numpy as np
from loguru import logger
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

def test_ctf_single_env_stress_resets():
    """Stress test with 10 actual resets (not mocked)."""
    env = CaptureTheFlagSingleEnv(visualize=False)
    
    try:
        for i in range(10):
            logger.info(f"Stress Cycle {i+1}/10...")
            obs, info = env.reset()
            assert obs.shape == (23, 128, 128)
            # Verify unit visibility
            assert len(env.player_units[1]) > 0
            assert len(env.player_units[2]) > 0
            
            # Step a few times
            for _ in range(5):
                action = env.action_space.sample()
                env.step(action)
                
            logger.info(f"Cycle {i+1} successful.")
    finally:
        env.close()
