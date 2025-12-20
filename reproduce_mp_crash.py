
import sys
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

# Add path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train.envs.minimal_sc2_env import MinimalSC2Env

def make_env(rank: int):
    """
    Utility function for multiprocessed env.
    
    :param rank: index of the subprocess
    """
    def _init():
        env = MinimalSC2Env(
            visualize=False
        )
        return env
    return _init

def main():
    print("--- Attempting Multi-Processing Training ---")
    
    # number of environments
    num_cpu = 2 
    
    # Create the vectorized environment
    # On Windows, this defaults to 'spawn', which requires picklability
    try:
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    except Exception as e:
        print(f"Failed to create SubprocVecEnv: {e}")
        raise e

    print("SubprocVecEnv created successfully.")

    try:
        model = PPO("MlpPolicy", env, verbose=1)
        print("Model created.")
        
        print("Starting learn...")
        model.learn(total_timesteps=100)
        print("Learn finished.")
        
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    # This is crucial for Windows multiprocessing
    main()
