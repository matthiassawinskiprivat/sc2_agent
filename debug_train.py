import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import sys

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.envs.level_0_env import Level0Env

def debug_train():
    """
    A minimal training script to debug the Level0Env.
    """
    print("--- Starting Minimal Debug Training ---")

    env = None  # Initialize env to None
    try:
        # Create a single environment
        env = Level0Env(visualize=False, level_name="level_0")

        # Create a PPO model
        model = PPO("MlpPolicy", env, verbose=1)

        # Train for a small number of steps
        model.learn(total_timesteps=1000)

        print("--- Minimal Debug Training Finished ---")

    except Exception as e:
        print(f"An error occurred during debug training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            env.close()

if __name__ == "__main__":
    debug_train()
