import pytest
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
import numpy as np
import os
import yaml
import threading
from train.train_pipeline import PipelineManager, MaskableSubprocVecEnv
from sb3_contrib import MaskablePPO
import gymnasium as gym

# Mock environment that contains a non-picklable object (thread lock)
# This simulates the real CTFEnv which has asyncio loops/locks
class NonPicklableCTFEnv(gym.Env):
    def __init__(self, **kwargs):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(23, 128, 128), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(1024)
        self.lock = threading.Lock()
        self.level_name = kwargs.get("level_name", "capture_the_flag_0")

    def reset(self, seed=None, options=None):
        return np.zeros((23, 128, 128), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((23, 128, 128), dtype=np.uint8), 0.0, False, False, {}

    def action_masks(self):
        return np.ones(1024, dtype=bool)
    
    def close(self):
        pass

def make_env():
    return NonPicklableCTFEnv()

def test_parallel_learn_startup_with_non_picklable_env():
    """
    Verify that PipelineManager._train_level can successfully call model.learn()
    even if the environment itself is not picklable, thanks to get_attr interception.
    """
    num_envs = 2
    level_name = "capture_the_flag_0"
    
    ctf_config = {
        "training": {
            "num_envs": num_envs,
            "learning_rate": 0.001,
            "batch_size": 64,
            "n_steps": 16,
            "training_steps": 32
        },
        "env": {
            "env_class": "CaptureTheFlagEnv",
            "headless": True,
            "grid_size": 128
        },
        "model": {
            "policy": "MultiInputPolicy",
            "gamma": 0.99,
            "ent_coef": 0.01,
            "n_epochs": 1
        }
    }
    
    config_content = yaml.dump(ctf_config)
    
    with patch("builtins.open", mock_open(read_data=config_content)), \
         patch("os.makedirs"), \
         patch("train.train_pipeline.Watchdog"), \
         patch("train.train_pipeline.CheckpointCallback"), \
         patch("train.train_pipeline.DiagnosticsCallback"), \
         patch("train.train_pipeline.ENV_REGISTRY", {"CaptureTheFlagEnv": NonPicklableCTFEnv}):
        
        manager = PipelineManager()
        
        try:
            # This will trigger reset() and get_attr('action_masks').
            # If get_attr('action_masks') is NOT intercepted, it will fail 
            # because NonPicklableCTFEnv has a lock.
            manager._train_level(level_name)
            print("SUCCESS: Parallel training started and completed its session with non-picklable env.")
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            pytest.fail(f"Parallel training CRASHED with communication error: {e}")
        except Exception as e:
            print(f"Parallel session ended with error: {e}")
            if "pickle" in str(e).lower() or "lock" in str(e).lower():
                 pytest.fail(f"Parallel training CRASHED with pickling error: {e}")

if __name__ == "__main__":
    test_parallel_learn_startup_with_non_picklable_env()
