import pytest
from unittest.mock import MagicMock, patch, mock_open
import yaml
import os
from train.train_pipeline import PipelineManager

@pytest.fixture
def mock_pipeline_manager():
    # Mock the pipeline.yaml read in __init__
    mock_pipeline_yaml = "levels: [capture_the_flag_0]"
    with patch("builtins.open", mock_open(read_data=mock_pipeline_yaml)):
        with patch("glob.glob", return_value=["config/capture_the_flag_0.yaml"]):
            manager = PipelineManager(config_dir="config")
            return manager

def test_ctf_num_envs_scaling_respected(mock_pipeline_manager):
    """
    Verify that num_envs is respected for CTF after the fix.
    """
    ctf_config = {
        "training": {
            "num_envs": 2,
            "learning_rate": 0.001,
            "batch_size": 64,
            "n_steps": 128,
            "training_steps": 100
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
            "n_epochs": 10
        }
    }
    
    config_content = yaml.dump(ctf_config)
    
    # We need to mock a lot of things inside _train_level to avoid actual SC2 launch
    with patch("builtins.open", mock_open(read_data=config_content)), \
         patch("os.makedirs"), \
         patch("train.train_pipeline.MaskableSyncVecEnv") as mock_sync_vec, \
         patch("train.train_pipeline.MaskableSubprocVecEnv") as mock_subproc_vec, \
         patch("train.train_pipeline.MaskablePPO") as mock_ppo, \
         patch("train.train_pipeline.Watchdog"), \
         patch("train.train_pipeline.CheckpointCallback"), \
         patch("train.train_pipeline.DiagnosticsCallback"):
        
        # Configure mocks to have __name__
        mock_sync_vec.__name__ = "MaskableSyncVecEnv"
        mock_subproc_vec.__name__ = "MaskableSubprocVecEnv"
        
        # Mock env instance
        mock_env = MagicMock()
        mock_env.observation_space.shape = (23, 128, 128)
        mock_env.action_space.n = 1024
        mock_sync_vec.return_value = mock_env
        mock_subproc_vec.return_value = mock_env
        
        # Call the method
        mock_pipeline_manager._train_level("capture_the_flag_0")
        
        # After the fix, it should use MaskableSubprocVecEnv because num_envs > 1 and headless
        mock_subproc_vec.assert_called_once()
        env_fns = mock_subproc_vec.call_args[0][0]
        assert len(env_fns) == 2, f"Expected 2 environments, but got {len(env_fns)}"

if __name__ == "__main__":
    pytest.main([__file__])
