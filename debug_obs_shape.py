
import os
import sys
sys.path.append(os.getcwd())
from train.envs.level_0_env import Level0Env
from unittest.mock import MagicMock

def debug_obs_shape():
    print("Initializing Level0Env...")
    # Mocking external dependencies to avoid launching SC2
    env = Level0Env(level_name="level_0", visualize=False, sc2_process_cls=MagicMock(), client_cls=MagicMock())
    
    print(f"Observation Space Shape: {env.observation_space.shape}")
    print(f"Active Layers: {env.training_conf.get('active_layers')}")
    
    if env.observation_space.shape[0] == 4:
        print("SUCCESS: Observation space has 4 channels as expected.")
    else:
        print(f"FAILURE: Expected 4 channels, got {env.observation_space.shape[0]}.")

if __name__ == "__main__":
    debug_obs_shape()
