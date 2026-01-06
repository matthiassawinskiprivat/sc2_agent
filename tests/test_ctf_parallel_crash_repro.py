import pytest
from unittest.mock import MagicMock
import numpy as np
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym

# Simple mock environment that behaves like CTF but is picklable
class MockCTFEnv(gym.Env):
    def __init__(self, **kwargs):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(23, 128, 128), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(1024)
        
        # This mirrors the old buggy CTFEnv where action_masks is only a method
        # If we add self.action_masks = None, it should work.
        if kwargs.get("with_fix", False):
            self.action_masks = None

    def reset(self, seed=None, options=None):
        return np.zeros((23, 128, 128), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((23, 128, 128), dtype=np.uint8), 0.0, False, False, {}

    def action_masks(self):
        return np.ones(1024, dtype=bool)

def make_env(with_fix):
    def _init():
        return MockCTFEnv(with_fix=with_fix)
    return _init

def test_get_attr_pickling_logic():
    """
    Definitively test if adding the attribute prevents the pickling crash.
    """
    # 1. Test WITHOUT FIX (Should crash or return method)
    print("\nTesting WITHOUT FIX...")
    venv_buggy = SubprocVecEnv([make_env(with_fix=False)], start_method='spawn')
    try:
        venv_buggy.reset()
        res = venv_buggy.get_attr('action_masks')
        print(f"Buggy Result type: {type(res[0])}")
        if callable(res[0]):
            print("Confirmed: Returns bound method (potentially dangerous for communications)")
    except Exception as e:
        print(f"Buggy crashed as expected: {e}")
    finally:
        venv_buggy.close()

    # 2. Test WITH FIX (Should return None and not crash)
    print("\nTesting WITH FIX...")
    venv_fixed = SubprocVecEnv([make_env(with_fix=True)], start_method='spawn')
    try:
        venv_fixed.reset()
        res = venv_fixed.get_attr('action_masks')
        print(f"Fixed Result: {res}")
        assert res[0] is None, f"Expected None (the attribute), got {res[0]}"
        print("Confirmed: Returns the attribute, satisfying the check safely.")
    except Exception as e:
        pytest.fail(f"Fixed version CRASHED: {e}")
    finally:
        venv_fixed.close()

if __name__ == "__main__":
    test_get_attr_pickling_logic()
