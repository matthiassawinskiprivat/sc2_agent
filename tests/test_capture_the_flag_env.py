import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from train.envs.capture_the_flag_env import CaptureTheFlagEnv

def test_ctf_env_init():
    """Test CTF environment initialization."""
    env = CaptureTheFlagEnv(
        mineral_budget=500,
        vespene_budget=200,
        flag_distance_min=20.0,
        unit_pool=["ZERGLING", "ROACH"]
    )
    assert env.mineral_budget == 500
    assert env.vespene_budget == 200
    assert env.flag_distance_min == 20.0
    assert env.unit_pool == ["ZERGLING", "ROACH"]
    # Observation space should match active_layers
    assert env.observation_space.shape == (len(env.active_layers), 128, 128)

def test_ctf_env_spawning_logic():
    """Test the resource-based spawning logic."""
    env = CaptureTheFlagEnv(mineral_budget=100, vespene_budget=0, unit_pool=["ZERGLING"])
    spawn_list = env._calculate_spawns(env.mineral_budget, env.vespene_budget)
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
    
    num_layers = len(env.active_layers)
    obs = np.zeros((num_layers, 128, 128), dtype=np.uint8)
    
    # Identify indices for assignment and objective
    try:
        assign_idx = env.active_layers.index("command_assignment")
        obj_idx = env.active_layers.index("command_objective")
    except ValueError:
        # Fallback to old hardcoded indices if layers were somehow different
        assign_idx, obj_idx = 4, 5
        
    obs[assign_idx, 10, 10] = 255 # Assignment
    obs[obj_idx, 20, 20] = 255 # Objective
    
    env.bot1.get_observation.return_value = obs
    
    joint_obs = await env._get_joint_observation()
    
    assert joint_obs[assign_idx].max() == 255
    assert joint_obs[obj_idx].max() == 255
    assert env.bot1.active_unit_tag == 111
