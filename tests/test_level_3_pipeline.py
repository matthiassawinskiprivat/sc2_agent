import pytest
from unittest.mock import MagicMock, patch
import yaml
from train.train_pipeline import PipelineManager
from train.envs.level_3_env import Level3Env

@patch("train.train_pipeline.MaskablePPO")
@patch("train.train_pipeline.MaskableSubprocVecEnv")
@patch("train.train_pipeline.Monitor")
def test_level3_pipeline_integration(mock_monitor, mock_vec_env, mock_ppo):
    """
    Test that PipelineManager instantiates Level3Env when processing level_3.
    """
    mock_vec_env.__name__ = "MaskableSubprocVecEnv"
    
    # We rely on the real config/level_3.yaml file being present.
    # But we need to make sure PipelineManager finds it.
    
    with patch("glob.glob", return_value=["config/level_3.yaml"]):
        manager = PipelineManager(config_dir="config")
        
        # We need to ensure that when we call _train_level, it reads the file.
        # But yaml.safe_load needs to work. 
        # Since I am not patching open(), it will read the REAL file.
        
        # We mock _train_level dependencies.
        # We need to catch the environment creation.
        
        try:
            manager._train_level("level_3")
        except Exception as e:
            print(f"DEBUG: Exception during _train_level: {e}")
            # PPO or other parts might fail due to mocks
            pass
        
        # Verify VecEnv was created
        if mock_vec_env.call_args:
            env_fns = mock_vec_env.call_args[0][0]
            created_env = env_fns[0]()
            
            # Check if the wrapped env is Level3Env
            wrapped_env = mock_monitor.call_args[0][0]
            assert isinstance(wrapped_env, Level3Env)
            assert wrapped_env.targets == 2
        else:
             pytest.fail("VecEnv not created")
