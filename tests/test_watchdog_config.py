import pytest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys

# Project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_pipeline import PipelineManager

def test_watchdog_configurable_timeout():
    """
    Verify that the watchdog initialization uses the configurable timeout.
    """
    with patch('train.train_pipeline.Watchdog') as MockWatchdog:
        with patch('train.train_pipeline.yaml.safe_load') as mock_yaml:
            with patch('train.train_pipeline.open', mock_open(read_data="dummy")):
                with patch('train.train_pipeline.os.makedirs', MagicMock()):
                    
                    mock_vec_env = MagicMock()
                    mock_vec_env.__name__ = "MaskableSyncVecEnv"
                    
                    with patch('train.train_pipeline.MaskableSyncVecEnv', mock_vec_env):
                        # 1. Test Default (no config value)
                        mock_yaml.return_value = {
                            "training": {"num_envs": 1, "training_steps": 1, "learning_rate": 0.001, "batch_size": 1, "n_steps": 1},
                            "env": {"grid_size": 16, "active_layers": ["allied_unit_density"]},
                            "model": {"policy": "CnnPolicy", "gamma": 0.99, "ent_coef": 0.01, "n_epochs": 1}
                        }
                        
                        manager = PipelineManager()
                        
                        mock_env_class = MagicMock()
                        mock_env_class.__name__ = "MockEnv"
                        
                        # Mock environment and model creation to ensure we reach watchdog
                        with patch('train.train_pipeline.ENV_REGISTRY', {"Level0Env": mock_env_class}):
                            with patch('train.train_pipeline.MaskablePPO') as MockPPO:
                                # Mock learn to avoid real SC2
                                MockPPO.return_value.learn = MagicMock()
                                # Mock environment instance
                                mock_env_inst = MagicMock()
                                mock_env_inst.observation_space.shape = (5, 128, 128)
                                mock_env_inst.action_space.n = 1024
                                with patch('train.train_pipeline.MaskableSyncVecEnv', return_value=mock_env_inst) as mock_vec_init:
                                    mock_vec_init.__name__ = "MaskableSyncVecEnv"
                                    
                                    try:
                                        manager._train_level("level_0")
                                    except Exception as e:
                                        print(f"Exception during test: {e}")
                                        pass
                                    
                                    # Default should now be 3600.0
                                    MockWatchdog.assert_called()
                                    _, kwargs = MockWatchdog.call_args
                                    assert kwargs.get('timeout') == 3600.0
                                
                        MockWatchdog.reset_mock()
                        
                        # 2. Test Explicit Config
                        mock_yaml.return_value["training"]["watchdog_timeout"] = 500.0
                        with patch('train.train_pipeline.ENV_REGISTRY', {"Level0Env": mock_env_class}):
                            with patch('train.train_pipeline.MaskablePPO', MagicMock()):
                                with patch('train.train_pipeline.MaskableSyncVecEnv', return_value=mock_env_inst) as mock_vec_init:
                                    mock_vec_init.__name__ = "MaskableSyncVecEnv"
                                    try:
                                        manager._train_level("level_0")
                                    except:
                                        pass
                                    
                                    # Should use config value
                                    MockWatchdog.assert_called()
                                    _, kwargs = MockWatchdog.call_args
                                    assert kwargs.get('timeout') == 500.0

if __name__ == "__main__":
    test_watchdog_configurable_timeout()
    print("Watchdog config test passed!")