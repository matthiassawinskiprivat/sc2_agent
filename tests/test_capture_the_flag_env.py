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
    spawn_list = env._calculate_spawns(env.minerals, env.vespene)
    assert len(spawn_list) == 4
    assert all(u == "ZERGLING" for u in spawn_list)

@pytest.mark.asyncio
async def test_ctf_env_layer_highlighting():
    """Verify that command_assignment and command_objective layers work."""
    env = CaptureTheFlagEnv()
    env.bot1 = MagicMock()
    env.player_units = {1: [111]}
    env.active_player_id = 1
    env.current_unit_index = 0
    
    # Mock observation with 7 layers
    # Layers: 0: Allied G, 1: Enemy G, 2: Allied H, 3: Enemy H, 4: Assignment, 5: Objective, 6: Pathable
    # Wait, my config has:
    # 0: allied_g_dps_base
    # 1: enemy_g_dps_base
    # 2: allied_health
    # 3: enemy_health
    # 4: command_assignment
    # 5: command_objective
    # 6: terrain_pathable
    
    obs = np.zeros((7, 128, 128), dtype=np.uint8)
    obs[4, 10, 10] = 255 # Assignment
    obs[5, 20, 20] = 255 # Objective
    
    env.bot1.get_observation.return_value = obs
    
    joint_obs = await env._get_joint_observation()
    
    assert joint_obs[4].max() == 255
    assert joint_obs[5].max() == 255
    assert env.bot1.active_unit_tag == 111
