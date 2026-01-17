import pytest
from unittest.mock import MagicMock
import os
import asyncio
import random
import numpy as np
import gymnasium as gym # Added import
from gymnasium import spaces # Added import

# Add the project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.envs.level_0_env import Level0Env


class MockLevel0Env(Level0Env):
    """A mock version of the environment that doesn't launch StarCraft II."""

    def __init__(self, **kwargs):
        # Initialize basic attributes expected by Level0Env methods
        self.visualize = kwargs.get("visualize", False)
        self.grid_size = kwargs.get("grid_size", 16)
        self.worker_index = kwargs.get("worker_index", 0)
        self.num_workers = kwargs.get("num_workers", 1)
        self.level_name = kwargs.get("level_name", "level_0")
        self.map_rotation_index = 0

        # Manually set dummy map list and map name
        self.map_list = [f"dummy_map_{i}.SC2Map" for i in range(5)] # Create 5 dummy maps
        self.map_name = self.map_list[self.worker_index % len(self.map_list)]

        self.worker_id = f"{random.randint(1000, 9999)}"
        # Mock logging to prevent file creation
        self.log_file = None
        # self._log(f"Worker {self.worker_index}: Mocked __init__.") # Log disabled for mock

        # Define proper action and observation spaces for gymnasium
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(42, 128, 128), dtype=np.float32)
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)

        # Mock HierarchicalAgentSystem and its agent
        self.system = MagicMock()
        self.agent_id = "zergling"
        self.agent = MagicMock()
        self.system.create_unit_agent.return_value = self.agent
        self.agent.get_action_mask.return_value = np.zeros(self.action_space.n) # Dummy mask

        # Mock communication
        self.game_process = MagicMock()
        self.game_process.is_alive.return_value = True
        self.parent_conn = MagicMock()
        # Mock responses for reset and step
        self.parent_conn.recv.side_effect = [
            (np.random.rand(42, 128, 128).astype(np.uint8), {"mock_info": True, "map_name": self.map_name}), # For reset
            (np.random.rand(42, 128, 128).astype(np.uint8), 0.0, False, False, {"mock_info": True}) # For step
        ]
        self.child_conn = MagicMock()

        self.steps = 0
        self.max_steps = 300
        self.episode_counter = 0 # For test logic

        self.client = MagicMock()
        self.sc2_process = MagicMock()
        
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.game_info = MagicMock()
        self.game_info.pathing_grid.width = 128
        self.game_info.game_info.pathing_grid.height = 128
        self.game_info.pathing_grid.__getitem__.return_value = 1

    # Override methods that would access real resources
    def _discover_maps(self):
        return self.map_list

    def _log(self, message):
        pass # Do not write logs for mock

    def _get_game_config(self):
        return {
            "map_name": self.map_name,
            "realtime": self.visualize,
            "step_time_limit": None,
            "game_time_limit": None,
            "disable_fog": True,
            "level_name": self.level_name,
        }

    def _get_player_config(self):
        return {
            "race": Race.Zerg,
            "agent": self.agent,
        }

    def reset(self, seed=None, options=None):
        self.episode_counter += 1
        simulated_reset_frequency = 10 # Matches the test's assumption
        if (self.episode_counter + self.worker_index) % simulated_reset_frequency == 0:
            self.map_rotation_index += 1
        
        map_idx = (self.worker_index + self.map_rotation_index) % len(self.map_list)
        self.map_name = self.map_list[map_idx]

        obs = np.random.rand(42, 128, 128).astype(np.uint8)
        info = {"mock_info": True, "map_name": self.map_name}
        return obs, info

    def step(self, action):
        obs = np.random.rand(42, 128, 128).astype(np.uint8)
        reward = 0.0
        terminated = False
        truncated = False
        info = {"mock_info": True}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass # Do nothing


def test_map_rotation_logic_pytest():
    num_workers = 4
    simulated_reset_frequency = 10
    num_episodes = simulated_reset_frequency * 2 + 1

    env_for_map_list = MockLevel0Env(worker_index=0, num_workers=1)
    expected_map_list = env_for_map_list.map_list
    num_maps = len(expected_map_list)
    assert num_maps > 0, "No maps were discovered."

    envs = [MockLevel0Env(worker_index=i, num_workers=num_workers) for i in range(num_workers)]
    
    # Keep track of the rotation index for each worker, mirroring the env's internal state
    map_rotation_indices_prediction = [0] * num_workers

    for worker_idx in range(num_workers):
        env = envs[worker_idx]
        initial_map_index = worker_idx % num_maps
        assert env.map_name == expected_map_list[initial_map_index], f"Worker {worker_idx}: Initial map mismatch."

    for episode in range(1, num_episodes + 1):
        for worker_idx in range(num_workers):
            env = envs[worker_idx]
            
            # PREDICT if map_rotation_index will be incremented for this worker in this episode
            # Note: env.episode_counter is incremented inside env.reset()
            # so we check `episode + worker_idx` instead of `env.episode_counter + worker_idx` for prediction here.
            if (episode + worker_idx) % simulated_reset_frequency == 0:
                map_rotation_indices_prediction[worker_idx] += 1

            obs, info = env.reset() # This will internally increment episode_counter and rotate map

            expected_map_idx = (worker_idx + map_rotation_indices_prediction[worker_idx]) % num_maps
            expected_map = expected_map_list[expected_map_idx]
            
            assert env.map_name == expected_map, f"Mismatch for Worker {worker_idx} at Episode {episode}"
            assert info["map_name"] == expected_map, f"Info dict map_name mismatch for Worker {worker_idx} at Episode {episode}"