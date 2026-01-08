import pytest
import numpy as np
import threading
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from train.train_pipeline import MaskableSubprocVecEnv

# Mock environment that contains a non-picklable object (thread lock)
class NonPicklableEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        # Thread lock is NOT picklable
        self.lock = threading.Lock()

    def reset(self, seed=None, options=None):
        return np.zeros((1,), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((1,), dtype=np.uint8), 0.0, False, False, {}

    def action_masks(self):
        return np.array([True, True])

def make_env():
    return NonPicklableEnv()

def test_get_attr_pickle_error_repro():
    """
    Reproduce the TypeError: cannot pickle '_thread.lock' object
    when get_attr tries to fetch a bound method from a non-picklable object.
    """
    # Use 'spawn' to ensure we test cross-process pickling (default on Windows)
    venv = MaskableSubprocVecEnv([make_env], start_method='spawn')
    try:
        venv.reset()
        
        print("\nAttempting to get bound method 'action_masks'...")
        # SB3's is_masking_supported calls get_attr('action_masks')
        # In a standard SubprocVecEnv, this fetches the bound method from the remote process.
        # Pickling a bound method requires pickling the 'self' object (the Env).
        # Since NonPicklableEnv has a lock, it should FAIL.
        
        try:
            res = venv.get_attr('action_masks')
            print(f"Result: {res}")
            # If we reached here, it didn't crash (maybe some versions of pickle handle it or it's not a bound method yet)
            # But the user reported it DOES crash.
        except Exception as e:
            print(f"Caught expected error: {type(e).__name__}: {e}")
            assert "pickle" in str(e).lower() or "thread.lock" in str(e).lower()
            return

    finally:
        venv.close()

if __name__ == "__main__":
    test_get_attr_pickle_error_repro()
