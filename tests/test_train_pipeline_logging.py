import pytest
from unittest.mock import MagicMock, patch, ANY
import os
from train.train_pipeline import PipelineManager

class TestTrainPipelineLogging:
    @patch("train.train_pipeline.logger")
    @patch("train.train_pipeline.MaskablePPO")
    @patch("train.train_pipeline.SyncVecEnv")
    @patch("train.train_pipeline.yaml")
    @patch("builtins.open")
    @patch("train.train_pipeline.glob.glob")
    @patch("train.train_pipeline.os.makedirs")
    def test_train_level_logging_and_callback(self, mock_makedirs, mock_glob, mock_open, mock_yaml, mock_sync_vec_env, mock_ppo, mock_logger):
        """
        Test that _train_level adds diagnostics logging and callback.
        """
        # Setup mocks
        mock_sync_vec_env.__name__ = "SyncVecEnv"
        
        mock_yaml.safe_load.side_effect = [
            {"levels": ["level_0"]}, # pipeline.yaml
            { # level_0.yaml
                "training": {
                    "num_envs": 1,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "n_steps": 128,
                    "training_steps": 1000
                },
                "env": {
                    "headless": False,
                    "grid_size": 128
                },
                "model": {
                    "gamma": 0.99,
                    "ent_coef": 0.01,
                    "n_epochs": 10,
                    "policy": "MlpPolicy"
                }
            }
        ]
        mock_glob.return_value = ["config/level_0.yaml"]
        
        manager = PipelineManager(config_dir="config")
        
        # Mock get_level_status to return NEEDS_TRAINING
        with patch.object(manager, "get_level_status", return_value=("NEEDS_TRAINING", None)):
             manager._train_level("level_0")
        
        # Verify logging of start
        start_log_found = False
        for call in mock_logger.info.call_args_list:
            if "Starting Training for Level: level_0" in call[0][0]:
                start_log_found = True
                break
        assert start_log_found, "Should log start of training level"

        # Verify DiagnosticsCallback is in the callback list passed to learn
        # model.learn(..., callback=CallbackList([...]), ...)
        # We need to inspect the callback argument
        mock_model = mock_ppo.return_value
        learn_call = mock_model.learn.call_args
        assert learn_call is not None, "model.learn should be called"
        
        _, kwargs = learn_call
        callback_list = kwargs.get("callback")
        assert callback_list is not None, "CallbackList should be passed"
        
        # We can't easily check the type of objects inside CallbackList if it's a real object wrapping mocks, 
        # but we can check if my new class is instantiated if I mock it.
        # So I should patch the new callback class I intend to create.

    @patch("train.train_pipeline.DiagnosticsCallback")
    @patch("train.train_pipeline.logger")
    @patch("train.train_pipeline.MaskablePPO")
    @patch("train.train_pipeline.SyncVecEnv")
    @patch("train.train_pipeline.yaml")
    @patch("builtins.open")
    @patch("train.train_pipeline.glob.glob")
    @patch("train.train_pipeline.os.makedirs")
    def test_diagnostics_callback_added(self, mock_makedirs, mock_glob, mock_open, mock_yaml, mock_sync_vec_env, mock_ppo, mock_logger, mock_diag_callback):
        # Setup mocks same as above
        mock_sync_vec_env.__name__ = "SyncVecEnv"
        
        mock_yaml.safe_load.side_effect = [
            {"levels": ["level_0"]},
            {
                "training": {
                    "num_envs": 1, 
                    "training_steps": 1000,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "n_steps": 128
                },
                "env": {"headless": False, "grid_size": 128},
                "model": {"gamma": 0.99, "ent_coef": 0.01, "n_epochs": 10, "policy": "MlpPolicy"}
            }
        ]
        mock_glob.return_value = ["config/level_0.yaml"]
        
        manager = PipelineManager(config_dir="config")
        
        with patch.object(manager, "get_level_status", return_value=("NEEDS_TRAINING", None)):
             manager._train_level("level_0")
             
        mock_diag_callback.assert_called_once(), "DiagnosticsCallback should be instantiated"
