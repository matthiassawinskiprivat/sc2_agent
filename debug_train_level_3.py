import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import os
import sys

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.envs.level_3_env import Level3Env

def debug_train_level_3():
    """
    A minimal training script to debug the Level3Env.
    """
    print("--- Starting Minimal Debug Training (Level 3) ---")

    env = None
    try:
        # Create a single environment
        env = Level3Env(visualize=False, level_name="level_3")

        # Create a PPO model
        model = PPO("CnnPolicy", env, verbose=1, n_steps=128)

        # Train for a small number of steps
        model.learn(total_timesteps=512)

        print("--- Minimal Debug Training Finished ---")

    except Exception as e:
        print(f"An error occurred during debug training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            env.close()

if __name__ == "__main__":
    debug_train_level_3()
