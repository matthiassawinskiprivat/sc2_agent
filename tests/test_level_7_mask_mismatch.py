import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from train.envs.level_7_env import Level7Env

class TestLevel7MaskMismatch:
    @pytest.fixture
    def mock_env(self):
        with patch('train.envs.level_0_env.SC2Process'), \
             patch('train.envs.level_0_env.Client'), \
             patch('train.envs.level_0_env.ConfigManager'), \
             patch('train.envs.level_7_env.Level7Bot'), \
             patch('sc2.game_state.GameState'), \
             patch('sc2.pixel_map.PixelMap'), \
             patch('train.envs.level_0_env.Level0Env._discover_maps', return_value=['map1']):
            
            env = Level7Env(
                grid_size=16, 
                level_name="level_7",
                training_conf={"targets": 1}
            )
            # Setup necessary mocks
            env._bot = MagicMock()
            env._bot.apply_action = AsyncMock()
            env._client = AsyncMock()
            env._loop = MagicMock()
            
            # Setup initial state
            env.active_unit_tags = [1]
            env.current_unit_index = 0
            
            # Mock the bot's action_masks() method to return the WRONG size
            # This simulates the current buggy behavior
            # Grid size 16 -> 256 (Move only). 
            # Env action space is 512 (Move + Attack).
            buggy_mask = np.ones(256, dtype=bool) 
            env._bot.action_masks.return_value = buggy_mask
            
            return env

    def test_mask_shape_mismatch_reproduction(self, mock_env):
        """
        This test reproduces the issue where the bot returns a mask that is smaller
        than the environment's action space.
        """
        # The environment expects the mask to match action_space.n
        expected_shape = mock_env.action_space.n # 512
        
        # In the bug scenario, action_masks() returns 256.
        received_mask = mock_env.action_masks()
        
        # This assertion verifies we can reproduce the mismatch.
        # Ideally, we want the TEST to pass if the BUG is present (to confirm reproduction)
        # OR fail if the system is supposed to work.
        # Given "Write Failing Tests", we want this test to Assert that the mask IS correct,
        # so it fails now and passes later when we fix it.
        
        assert len(received_mask) == expected_shape, \
            f"Mask shape mismatch! Expected {expected_shape}, got {len(received_mask)}"