import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
import os
import sys

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.envs.level_7_env import Level7Env

def debug_train_level_7():
    """
    A minimal training script to debug the Level7Env with doubled action space.
    """
    print("--- Starting Level 7 Debug Training ---")

    env = None
    try:
        # Create a single environment
        env = Level7Env(visualize=False, level_name="level_7", grid_size=16)

        # Create a MaskablePPO model (as used in the project)
        # Note: Level 7 has 2 * 16 * 16 = 512 actions
        model = MaskablePPO("CnnPolicy", env, verbose=1)

        print(f"Action space: {env.action_space.n}")

        # Train for a small number of steps
        # This will trigger collect_rollouts and action masking
        model.learn(total_timesteps=100)

        print("--- Level 7 Debug Training Finished ---")

    except Exception as e:
        print(f"An error occurred during debug training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if env:
            env.close()

if __name__ == "__main__":
    debug_train_level_7()
