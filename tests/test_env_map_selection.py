import unittest
from unittest.mock import patch, MagicMock
import os
import asyncio
import random

# Add the project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.envs.level_0_env import Level0Env, HARD_RESET_COUNT





class MockLevel0Env(Level0Env):

    """A mock version of the environment that doesn't launch StarCraft II."""

    def __init__(self, **kwargs):

        # We need to call the real __init__ to set up map lists, etc.

        super().__init__(**kwargs)

        # Prevent the real environment from trying to launch SC2 in these tests

        self.client = MagicMock()

        self.sc2_process = MagicMock()

        

        # Mock the event loop to prevent crashes

        try:

            self.loop = asyncio.get_running_loop()

        except RuntimeError:  # 'RuntimeError: There is no current event loop...'

            self.loop = asyncio.new_event_loop()

            asyncio.set_event_loop(self.loop)



        # Mock game_info needed for _calculate_spawn_points

        self.game_info = MagicMock()

        self.game_info.pathing_grid.width = 128

        self.game_info.pathing_grid.height = 128

        self.game_info.pathing_grid.__getitem__.return_value = 1





    def _launch_client(self):

        # Override to do nothing, but set client to a mock object

        self.client = MagicMock()

        self._log(f"Worker {self.worker_index}: Mock launch with map: {self.map_name}")

        pass



    async def _soft_reset_logic(self):

        # Override to do nothing

        pass



    def close(self):

        # Override to do nothing, and prevent real close logic

        self._log(f"Worker {self.worker_index}: Mock close.")

        self.client = None

        self.sc2_process = None

        

    def _get_observation(self):

        # Return a dummy observation, this is crucial to prevent calls to the (None) client

        return "dummy_obs"





class TestMapSelection(unittest.TestCase):



    def test_map_rotation_logic(self):
        num_workers = 4
        num_episodes = 151 # Test for a few full cycles of hard resets

        # Discover maps once to get the expected order
        env_for_map_list = MockLevel0Env(worker_index=0, num_workers=1)
        expected_map_list = env_for_map_list.map_list
        num_maps = len(expected_map_list)
        self.assertGreater(num_maps, 0, "No maps were discovered.")

        # --- Simulation ---
        # Create a set of mock environments, one for each worker
        envs = [MockLevel0Env(worker_index=i, num_workers=num_workers) for i in range(num_workers)]
        
        # Keep track of the rotation index for each worker, mirroring the env's internal state
        map_rotation_indices = [0] * num_workers

        # Check initial map assignment
        for i in range(num_workers):
            envs[i].episode_counter = 0 # Reset for predictability
            initial_map_index = i % num_maps
            self.assertEqual(envs[i].map_name, expected_map_list[initial_map_index])
            print(f"Initial Check - Worker {i}: Map '{envs[i].map_name}' (Correct)")

        # Simulate running episodes
        for episode in range(1, num_episodes):
            for worker_idx in range(num_workers):
                env = envs[worker_idx]

                # PREDICT: Check if a hard reset will occur in the upcoming reset() call.
                # The environment's episode_counter will be incremented to `episode` inside reset().
                if episode > 1 and (episode + worker_idx) % HARD_RESET_COUNT == 0:
                    map_rotation_indices[worker_idx] += 1
                    print(f"Predicting hard reset for worker {worker_idx} at episode {episode}. New rotation index: {map_rotation_indices[worker_idx]}")

                # ACT: The reset method contains the core logic we want to test
                env.reset()

                # VERIFY: Determine the expected map based on our prediction
                current_map_rotation = map_rotation_indices[worker_idx]
                expected_map_idx = (worker_idx + current_map_rotation) % num_maps
                expected_map = expected_map_list[expected_map_idx]
                
                # Assert that the map name is correct
                self.assertEqual(env.map_name, expected_map, f"Mismatch for Worker {worker_idx} at Episode {episode}")
        
        print("\nMap rotation logic test passed successfully!")



if __name__ == '__main__':

    unittest.main()
