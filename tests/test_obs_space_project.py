
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from stable_baselines3.common.preprocessing import is_image_space
import sys

def test_project_case():
    channels = 44
    height = 128
    width = 128
    
    # Exact definition from Level0Env
    observation_space = spaces.Box(low=0, high=255, shape=(channels, height, width), dtype=np.uint8)
    
    print(f"Testing project space {observation_space.shape}...")
    start = time.perf_counter()
    # Call it many times to see cumulative effect
    iterations = 1000
    for _ in range(iterations):
        is_image_space(observation_space)
    end = time.perf_counter()
    total_time = (end - start) * 1000
    print(f"Total time for {iterations} calls: {total_time:.4f} ms")
    print(f"Avg time per call: {total_time/iterations:.4f} ms")

if __name__ == "__main__":
    test_project_case()
