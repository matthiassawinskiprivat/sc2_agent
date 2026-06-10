import pytest
import numpy as np
from gymnasium import spaces
from train.sync_vec_env import SyncVecEnv
from unittest.mock import MagicMock, patch

class TestSyncVecEnvLogging:
    @patch("train.sync_vec_env.logger")
    def test_step_wait_logging(self, mock_logger):
        """Test logging inside step_wait."""
        # Setup mock environments
        obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        mock_env1 = MagicMock()
        mock_env1.step.return_value = (np.zeros(1, dtype=np.float32), 0, False, False, {})
        mock_env1.observation_space = obs_space
        mock_env1.action_space = spaces.Discrete(2)
        mock_env1.render_mode = None
        
        mock_env2 = MagicMock()
        mock_env2.step.return_value = (np.zeros(1, dtype=np.float32), 0, False, False, {})
        mock_env2.observation_space = obs_space
        mock_env2.action_space = spaces.Discrete(2)
        mock_env2.render_mode = None
        
        env_fns = [lambda: mock_env1, lambda: mock_env2]
        venv = SyncVecEnv(env_fns)
        
        venv.step_async(np.array([0, 0]))
        venv.step_wait()
        
        # Verify step logging
        # We check if any call to debug contains our strings
        all_messages = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Stepping environment 0" in msg for msg in all_messages)
        assert any("Stepping environment 1" in msg for msg in all_messages)

    @patch("train.sync_vec_env.logger")
    def test_reset_logging(self, mock_logger):
        """Test logging inside reset."""
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(1, dtype=np.float32), {})
        mock_env.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        mock_env.action_space = spaces.Discrete(2)
        
        env_fns = [lambda: mock_env]
        venv = SyncVecEnv(env_fns)
        
        venv.reset()
        
        # Verify reset logging
        all_messages = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Resetting environment 0" in msg for msg in all_messages)

    @patch("train.sync_vec_env.logger")
    def test_close_logging(self, mock_logger):
        """Test logging inside close."""
        mock_env = MagicMock()
        mock_env.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        mock_env.action_space = spaces.Discrete(2)
        
        env_fns = [lambda: mock_env]
        venv = SyncVecEnv(env_fns)
        
        venv.close()
        
        # Verify close logging
        all_messages = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Closing environment 0" in msg for msg in all_messages)
