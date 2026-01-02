import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from train.envs.capture_the_flag_env import CaptureTheFlagEnv

def test_ctf_env_init():
    """Test CTF environment initialization."""
    env = CaptureTheFlagEnv(
        minerals=500,
        vespene=200,
        flag_distance_min=20.0,
        unit_pool=["ZERGLING", "ROACH"]
    )
    assert env.minerals == 500
    assert env.vespene == 200
    assert env.flag_distance_min == 20.0
    assert env.unit_pool == ["ZERGLING", "ROACH"]
    # Observation space should be 7 layers
    assert env.observation_space.shape == (7, 128, 128)

def test_ctf_env_spawning_logic():
    """Test the resource-based spawning logic."""
    env = CaptureTheFlagEnv(minerals=100, vespene=0, unit_pool=["ZERGLING"])
    # Zergling costs 25 minerals (standard SC2)
    # 100 minerals should spawn 4 zerglings
    
    spawn_list = env._calculate_spawns(env.minerals, env.vespene)
    
    assert len(spawn_list) == 4
    assert all(u == "ZERGLING" for u in spawn_list)

@pytest.mark.asyncio
async def test_ctf_env_observation_shape():
    """Test that the joint observation has 7 layers."""
    env = CaptureTheFlagEnv()
    
    # Mock bots
    env.bot1 = MagicMock()
    env.bot2 = MagicMock()
    
    # Mock bot observation return
    mock_obs = np.zeros((7, 128, 128), dtype=np.uint8)
    env.bot1.get_observation.return_value = mock_obs
    
    env.active_player_id = 1
    env.player_units = {1: [123], 2: [456]}
    
    obs = await env._get_joint_observation()
    
    assert obs.shape == (7, 128, 128)
    assert env.bot1.active_unit_tag == 123