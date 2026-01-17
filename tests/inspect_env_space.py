
import numpy as np
from train.envs.level_0_env import Level0Env

def inspect_env_space():
    env = Level0Env(visualize=False, grid_size=32)
    obs_space = env.observation_space
    
    print(f"Space Type: {type(obs_space)}")
    print(f"Dtype: {obs_space.dtype}")
    print(f"Low Type: {type(obs_space.low)}")
    print(f"High Type: {type(obs_space.high)}")
    
    if isinstance(obs_space.low, np.ndarray):
        print(f"Low Shape: {obs_space.low.shape}")
        # Check if it's all zeros
        print(f"Low is all zeros: {np.all(obs_space.low == 0)}")
        
    if isinstance(obs_space.high, np.ndarray):
        print(f"High Shape: {obs_space.high.shape}")
        # Check if it's all 255
        print(f"High is all 255: {np.all(obs_space.high == 255)}")

if __name__ == "__main__":
    inspect_env_space()
