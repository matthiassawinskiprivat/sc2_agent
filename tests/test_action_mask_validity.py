
import pytest
import numpy as np
import os
import sys
from stable_baselines3.common.monitor import Monitor

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.envs.level_0_env import Level0Env
from train.train_pipeline import MaskableSubprocVecEnv

def make_env(rank):
    def _init():
        return Monitor(
            Level0Env(
                level_name="level_0",
                visualize=False,
                grid_size=32,
                worker_index=rank,
                num_workers=2
            )
        )
    return _init

def test_action_mask_validity():
    """
    Assert that get_action_masks() always returns at least one True value per environment.
    """
    num_envs = 2
    env_fns = [make_env(i) for i in range(num_envs)]
    env = MaskableSubprocVecEnv(env_fns, start_method='spawn')
    
    try:
        env.reset()
        for _ in range(20):
            masks = env.get_action_masks()
            assert masks.shape == (num_envs, 1024)
            assert masks.dtype == bool
            
            for i in range(num_envs):
                assert np.any(masks[i]), f"Env {i} has all actions masked (all False)!"
            
            actions = np.array([env.action_space.sample() for _ in range(num_envs)])
            env.step(actions)
            
    finally:
        env.close()

if __name__ == "__main__":
    test_action_mask_validity()
    print("Action mask validity test passed!")
