
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
import pytest
import time

class MinimalMPEnv(gym.Env):
    """
    A minimal environment to test multiprocessing stability.
    It doesn't use SC2, but simulates a slow reset to mimic SC2's behavior.
    """
    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 10, 10), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(10)

    def reset(self, seed=None, options=None):
        # Simulate some setup time
        time.sleep(0.1)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

def make_env(rank):
    def _init():
        return MinimalMPEnv(rank)
    return _init

def test_minimal_mp_stability():
    """
    Test if SubprocVecEnv can launch and run 2 simple environments.
    This verifies the basic MP infrastructure works.
    """
    num_envs = 2
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    try:
        obs = env.reset()
        assert obs.shape == (num_envs, 1, 10, 10)
        
        actions = np.array([env.action_space.sample() for _ in range(num_envs)])
        obs, rewards, dones, infos = env.step(actions)
        
        assert obs.shape == (num_envs, 1, 10, 10)
        assert len(rewards) == num_envs
    finally:
        env.close()

import asyncio

class AsyncDummyEnv(gym.Env):
    """
    A dummy environment that uses asyncio, mimicking Level0Env.
    """
    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 10, 10), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(10)
        self.loop = None

    def reset(self, seed=None, options=None):
        # This mimics the broken pattern in Level0Env:
        # creating a new loop in every reset.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._async_setup())
        except Exception as e:
            print(f"Error in AsyncDummyEnv reset: {e}")
            raise
        
        return self.observation_space.sample(), {}

    async def _async_setup(self):
        # Simulate some async work
        await asyncio.sleep(0.1)

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def close(self):
        if self.loop:
            self.loop.close()
            self.loop = None

def make_async_env(rank):
    def _init():
        return AsyncDummyEnv(rank)
    return _init

def test_async_mp_stability():
    """
    Test if SubprocVecEnv can handle an environment that uses asyncio.
    """
    num_envs = 2
    env = SubprocVecEnv([make_async_env(i) for i in range(num_envs)])
    
    try:
        obs = env.reset()
        assert obs.shape == (num_envs, 1, 10, 10)
    finally:
        env.close()
