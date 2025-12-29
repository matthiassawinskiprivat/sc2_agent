import pytest
import torch
from unittest.mock import MagicMock, patch
import sys
import os

# Mock modules before importing pipeline
sys.modules["train.envs.level_0_env"] = MagicMock()
sys.modules["train.sync_vec_env"] = MagicMock()
sys.modules["src.utils.watchdog"] = MagicMock()
sys.modules["src.utils.sc2_cleanup"] = MagicMock()

from train.train_pipeline import PipelineManager

@patch("train.train_pipeline.MaskablePPO")
@patch("train.train_pipeline.yaml")
@patch("train.train_pipeline.os")
@patch("train.train_pipeline.glob")
@patch("train.train_pipeline.graft_model_weights")
def test_pipeline_invokes_grafting(mock_graft, mock_glob, mock_os, mock_yaml, mock_ppo):
    """
    Verify that _train_level calls graft_model_weights with new_action_dim
    when it detects a mismatch in action space.
    """
    # Setup Mocks
    mock_os.path.join.side_effect = lambda *args: "/".join(args)
    mock_os.path.exists.return_value = False
    
    # Mock Config
    mock_yaml.safe_load.return_value = {
        "training": {"num_envs": 1, "learning_rate": 0.001, "batch_size": 64, "n_steps": 1024, "training_steps": 100},
        "env": {"grid_size": 64, "headless": True},
        "model": {"policy": "CnnPolicy", "gamma": 0.99, "ent_coef": 0.01, "n_epochs": 4}
    }
    
    # Mock Env (Action space mismatch: Saved=1024, Env=4096)
    mock_env = MagicMock()
    mock_env.observation_space.shape = (4, 128, 128)
    mock_env.action_space.n = 4096 # 64x64
    
    # MaskableSubprocVecEnv or SyncVecEnv will be instantiated. 
    # We need to intercept the env creation. 
    
    mock_vec_env_cls = MagicMock()
    mock_vec_env_cls.__name__ = "MaskableSubprocVecEnv"
    mock_vec_env_cls.return_value = mock_env
    
    with patch("train.train_pipeline.MaskableSubprocVecEnv", new=mock_vec_env_cls):
        
        # Mock Loaded Model (Saved=1024)
        mock_saved_model = MagicMock()
        mock_saved_policy = MagicMock()
        mock_saved_model.policy = mock_saved_policy
        mock_saved_policy.observation_space.shape = (4, 128, 128)
        mock_saved_policy.action_space.n = 1024 # 32x32
        
        # Mock state_dict
        mock_state_dict = {"action_net.weight": torch.randn(1024, 512)}
        mock_saved_policy.state_dict.return_value = mock_state_dict
        
        mock_ppo.load.return_value = mock_saved_model
        
        # Configure new model's load_state_dict return value
        mock_ppo.return_value.policy.load_state_dict.return_value = ([], [])
        
        # Instantiate Manager
        manager = PipelineManager(config_dir="config")
        
        # Run _train_level with a model path
        manager._train_level("level_1", model_path_to_load="dummy_path.zip")
        
        # VERIFICATION
        
        # 1. Check if graft_model_weights was called with new_action_dim=4096
        # Note: We expect it to be called ONCE if both mismatches handled together, 
        # or TWICE if handled separately.
        # But our plan is to handle action mismatch via grafting too.
        
        # Verify call args
        mock_graft.assert_called()
        call_args_list = mock_graft.call_args_list
        
        # Ensure at least one call passed new_action_dim
        found_action_graft = False
        for args, kwargs in call_args_list:
            if kwargs.get("new_action_dim") == 4096:
                found_action_graft = True
                assert args[0] == mock_state_dict
        
        assert found_action_graft, "graft_model_weights was not called with new_action_dim=4096"
        
        # 2. Check if a NEW model was initialized (because mismatch was found)
        # Verify MaskablePPO constructor was called (once)
        assert mock_ppo.call_count == 1 
        # Verify MaskablePPO.load was called (once)
        assert mock_ppo.load.call_count == 1
        
        # 3. Check if load_state_dict was called on the new model's policy
        # The new model instance is the return value of the constructor call
        new_model_instance = mock_ppo.return_value
        new_model_instance.policy.load_state_dict.assert_called()