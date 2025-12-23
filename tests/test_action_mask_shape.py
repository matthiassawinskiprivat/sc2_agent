
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

@pytest.mark.parametrize("num_envs", [2])
def test_maskable_subproc_vec_env_mask_shape(num_envs):
    """
    Test if MaskableSubprocVecEnv returns action masks of the correct shape.
    This specifically verifies the fix for the shape mismatch error.
    """
    env_fns = [make_env(i) for i in range(num_envs)]
    env = MaskableSubprocVecEnv(env_fns, start_method='spawn')
    
    try:
        env.reset()
        masks = env.get_action_masks()
        
        # Each environment has grid_size=32, so 32*32 = 1024 actions.
        # With 2 environments, the shape should be (2, 1024).
        expected_shape = (num_envs, 32 * 32)
        
        assert masks.shape == expected_shape, f"Expected shape {expected_shape}, but got {masks.shape}"
        assert masks.dtype == bool
        
        # Test step returns correct masks via env_method if needed (MaskablePPO uses get_action_masks)
        actions = np.array([env.action_space.sample() for _ in range(num_envs)])
        env.step(actions)
        
        masks_after_step = env.get_action_masks()
        assert masks_after_step.shape == expected_shape
        
    finally:
        env.close()

if __name__ == "__main__":
    # Manually run the test if script is executed directly
    try:
        test_maskable_subproc_vec_env_mask_shape(2)
        print("Test passed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
