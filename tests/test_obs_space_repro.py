
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from stable_baselines3.common.preprocessing import is_image_space
import sys
import os

# Mock a large observation space similar to the project
def test_obs_space_performance():
    channels = 44
    height = 128
    width = 128
    
    # Scenario A: Scalar bounds (Fast)
    space_scalar = spaces.Box(low=0, high=255, shape=(channels, height, width), dtype=np.uint8)
    
    # Scenario B: Array bounds (Slow)
    # Some wrappers or transformations might convert scalars to full arrays
    low_array = np.zeros((channels, height, width), dtype=np.uint8)
    high_array = np.full((channels, height, width), 255, dtype=np.uint8)
    space_array = spaces.Box(low=low_array, high=high_array, dtype=np.uint8)
    
    # Scenario C: float32 (Fastest?)
    space_float = spaces.Box(low=0.0, high=1.0, shape=(channels, height, width), dtype=np.float32)
    
    print(f"Testing is_image_space performance for shape {space_scalar.shape}...")
    
    start = time.perf_counter()
    res_a = is_image_space(space_scalar)
    end = time.perf_counter()
    time_a = (end - start) * 1000
    print(f"uint8 Scalar bounds: {time_a:.4f} ms (Result: {res_a})")
    
    start = time.perf_counter()
    res_c = is_image_space(space_float)
    end = time.perf_counter()
    time_c = (end - start) * 1000
    print(f"float32 Scalar bounds: {time_c:.4f} ms (Result: {res_c})")
    
    start = time.perf_counter()
    res_b = is_image_space(space_array)
    end = time.perf_counter()
    time_b = (end - start) * 1000
    print(f"Array bounds: {time_b:.4f} ms (Result: {res_b})")
    
    # In some environments, the check might be called thousands of times
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        is_image_space(space_array)
    end = time.perf_counter()
    avg_time_b = ((end - start) / iterations) * 1000
    print(f"Array bounds (avg over {iterations} iterations): {avg_time_b:.4f} ms")

if __name__ == "__main__":
    test_obs_space_performance()
