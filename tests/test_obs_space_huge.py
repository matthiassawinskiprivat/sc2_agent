
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from stable_baselines3.common.preprocessing import is_image_space
import sys

def test_slow_case():
    # Large number of channels?
    channels = 1000
    height = 256
    width = 256
    
    low = np.zeros((channels, height, width), dtype=np.uint8)
    high = np.full((channels, height, width), 255, dtype=np.uint8)
    
    space = spaces.Box(low=low, high=high, dtype=np.uint8)
    
    print(f"Testing huge space {space.shape}...")
    start = time.perf_counter()
    is_image_space(space)
    end = time.perf_counter()
    print(f"Time: {(end - start)*1000:.4f} ms")

if __name__ == "__main__":
    test_slow_case()
