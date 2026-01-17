import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from stable_baselines3.common.preprocessing import is_image_space
import pytest
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.envs.level_0_env import Level0Env

def test_obs_space_performance():
    """
    Measures the performance of SB3's is_image_space check on the ACTUAL Level0Env observation space.
    """
    # Instantiate the real environment (mocking visuals to avoid SC2 launch)
    # Note: Level0Env init does not launch SC2 if we don't call reset(), usually.
    # But it does load config.
    try:
        env = Level0Env(visualize=False, grid_size=32)
        print(f"Env initialized. Obs Space: {env.observation_space}")
    except Exception as e:
        pytest.fail(f"Failed to instantiate Level0Env: {e}")

    iterations = 1000
    
    # Measure is_image_space on the real env space
    start = time.perf_counter()
    for _ in range(iterations):
        is_image_space(env.observation_space)
    end = time.perf_counter()
    time_taken = (end - start) * 1000
    print(f"\nLevel0Env Space Check: {time_taken:.4f} ms for {iterations} calls")
    
    # Threshold: Should be very fast (sub-1ms) because it returns False immediately for float32
    assert time_taken < 1.0, f"Check is still too slow! Took {time_taken:.4f} ms"
    
    # Verify properties
    assert env.observation_space.dtype == np.float32, "Observation space dtype should be float32"
    assert np.all(env.observation_space.low == 0.0), "Low bound should be 0.0"
    assert np.all(env.observation_space.high == 1.0), "High bound should be 1.0"

if __name__ == "__main__":
    test_obs_space_performance()