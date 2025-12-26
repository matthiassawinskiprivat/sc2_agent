
import os
import sys
sys.path.append(os.getcwd())
from train.envs.level_0_env import Level0Env
from unittest.mock import MagicMock

def debug_obs_shape():
    for level in ["level_0", "level_1"]:
        print(f"\n--- Checking {level} ---")
        # Mocking external dependencies to avoid launching SC2
        env = Level0Env(level_name=level, visualize=False, sc2_process_cls=MagicMock(), client_cls=MagicMock())
        
        print(f"Observation Space Shape: {env.observation_space.shape}")
        print(f"Active Layers: {env.training_conf.get('active_layers')}")
        
        expected_channels = 4 if level == "level_0" else 5
        if env.observation_space.shape[0] == expected_channels:
            print(f"SUCCESS: {level} has {expected_channels} channels as expected.")
        else:
            print(f"FAILURE: Expected {expected_channels} channels, got {env.observation_space.shape[0]}.")

if __name__ == "__main__":
    debug_obs_shape()
