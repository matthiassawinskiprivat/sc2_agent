
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from stable_baselines3.common.preprocessing import is_image_space
import pytest

def test_obs_space_performance():
    """
    Measures the performance of SB3's is_image_space check on the project's observation space.
    The current implementation (uint8 with large bounds) is expected to be slower than
    an optimized float32 implementation.
    """
    channels = 44
    height = 128
    width = 128
    
    # Current project definition (uint8)
    space_uint8 = spaces.Box(low=0, high=255, shape=(channels, height, width), dtype=np.uint8)
    
    # Target optimization (float32)
    space_float32 = spaces.Box(low=0.0, high=1.0, shape=(channels, height, width), dtype=np.float32)
    
    iterations = 1000
    
    # Measure uint8
    start = time.perf_counter()
    for _ in range(iterations):
        is_image_space(space_uint8)
    end = time.perf_counter()
    time_uint8 = (end - start) * 1000
    print(f"\n uint8 (Current): {time_uint8:.4f} ms for {iterations} calls")
    
    # Measure float32
    start = time.perf_counter()
    for _ in range(iterations):
        is_image_space(space_float32)
    end = time.perf_counter()
    time_float32 = (end - start) * 1000
    print(f"float32 (Optimized): {time_float32:.4f} ms for {iterations} calls")
    
    # Assert that float32 is significantly faster (at least 10x)
    assert time_float32 < time_uint8 / 10, f"Optimization check failed: float32 was not significantly faster ({time_float32:.4f} vs {time_uint8:.4f} ms)"

if __name__ == "__main__":
    test_obs_space_performance()
