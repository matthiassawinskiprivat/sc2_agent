import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from train.envs.level_7_env import Level7Env
from src.level_7_bot import Level7Bot

class TestLevel7MaskMismatch:
    @pytest.fixture
    def mock_env(self):
        mock_cm = MagicMock()
        mock_cm.get_cumulative_active_layers.return_value = ["layer1"]
        mock_cm.get_cumulative_action_heads.return_value = ["move", "attack"]
        mock_cm.get_level_config.return_value = {"env": {"rewards": {}}}

        with patch('train.envs.level_0_env.SC2Process'), \
             patch('train.envs.level_0_env.Client'), \
             patch('train.envs.level_0_env.ConfigManager', return_value=mock_cm), \
             patch('sc2.game_state.GameState'), \
             patch('sc2.pixel_map.PixelMap'), \
             patch('train.envs.level_0_env.Level0Env._discover_maps', return_value=['map1']):
            
            env = Level7Env(
                grid_size=16, 
                level_name="level_7",
                training_conf={"targets": 1}
            )
            # Setup necessary mocks
            env._client = AsyncMock()
            env._loop = MagicMock()
            
            # Use real Level7Bot but mock its dependencies
            bot = Level7Bot(conn=MagicMock(), training_conf={"targets": 1})
            
            # Setup agent system mock properly
            mock_agent_system = MagicMock()
            mock_unit_agent = MagicMock()
            mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
            bot.agent_system = mock_agent_system
            
            # Mock agent's move mask (GridSize^2 = 256)
            mock_move_mask = np.ones(256, dtype=bool)
            mock_unit_agent.get_action_mask.return_value = mock_move_mask
            
            env._bot = bot
            
            return env

    def test_mask_shape_fixed(self, mock_env):
        """
        Verifies that Level7Env.action_masks() now returns the correct doubled size.
        """
        # Expected: 2 * 16 * 16 = 512
        expected_shape = mock_env.action_space.n 
        assert expected_shape == 512
        
        received_mask = mock_env.action_masks()
        
        assert len(received_mask) == expected_shape, \
            f"Mask shape mismatch! Expected {expected_shape}, got {len(received_mask)}"
        
        # Verify it's a concatenation of move masks
        half = expected_shape // 2
        assert np.array_equal(received_mask[:half], received_mask[half:])