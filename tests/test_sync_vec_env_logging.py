import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from train.sync_vec_env import SyncVecEnv

class TestSyncVecEnvLogging:
    @patch("train.sync_vec_env.logger")
    def test_step_wait_logging(self, mock_logger):
        """Test logging inside step_wait."""
        # Setup mock environments
        mock_env1 = MagicMock()
        mock_env1.step.return_value = (np.zeros(1), 0, False, False, {})
        mock_env1.observation_space = MagicMock()
        mock_env1.action_space = MagicMock()
        mock_env1.render_mode = None
        
        mock_env2 = MagicMock()
        mock_env2.step.return_value = (np.zeros(1), 0, False, False, {})
        mock_env2.render_mode = None
        
        env_fns = [lambda: mock_env1, lambda: mock_env2]
        venv = SyncVecEnv(env_fns)
        
        venv.step_async(np.array([0, 0]))
        venv.step_wait()
        
        # Verify step logging
        log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Stepping environment 0" in msg for msg in log_calls), "Should log stepping env 0"
        assert any("Stepping environment 1" in msg for msg in log_calls), "Should log stepping env 1"

    @patch("train.sync_vec_env.logger")
    def test_reset_logging(self, mock_logger):
        """Test logging inside reset."""
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(1), {})
        mock_env.observation_space = MagicMock()
        mock_env.action_space = MagicMock()
        
        env_fns = [lambda: mock_env]
        venv = SyncVecEnv(env_fns)
        
        venv.reset()
        
        # Verify reset logging
        log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Resetting environment 0" in msg for msg in log_calls), "Should log resetting env 0"

    @patch("train.sync_vec_env.logger")
    def test_close_logging(self, mock_logger):
        """Test logging inside close."""
        mock_env = MagicMock()
        mock_env.observation_space = MagicMock()
        mock_env.action_space = MagicMock()
        
        env_fns = [lambda: mock_env]
        venv = SyncVecEnv(env_fns)
        
        venv.close()
        
        # Verify close logging
        log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Closing environment 0" in msg for msg in log_calls), "Should log closing env 0"
