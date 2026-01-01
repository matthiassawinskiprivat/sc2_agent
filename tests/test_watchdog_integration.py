import pytest
from unittest.mock import MagicMock, patch
import time
import threading
from train.train_pipeline import PipelineManager
from src.utils.watchdog import Watchdog

class TestWatchdogIntegration:
    @patch("train.train_pipeline.Watchdog")
    @patch("train.train_pipeline.DiagnosticsCallback")
    @patch("train.train_pipeline.MaskablePPO")
    @patch("train.train_pipeline.SyncVecEnv")
    @patch("train.train_pipeline.yaml")
    @patch("builtins.open")
    @patch("train.train_pipeline.glob.glob")
    @patch("train.train_pipeline.os.makedirs")
    def test_watchdog_lifecycle_in_pipeline(self, mock_makedirs, mock_glob, mock_open, mock_yaml, mock_sync_vec_env, mock_ppo, mock_diag_callback, mock_watchdog):
        """Test that watchdog is started and stopped in the training loop."""
        # Setup mocks
        mock_sync_vec_env.__name__ = "SyncVecEnv"
        mock_yaml.safe_load.side_effect = [
            {"levels": ["level_0"]},
            {
                "training": {"num_envs": 1, "training_steps": 100, "learning_rate": 0.001, "batch_size": 64, "n_steps": 128},
                "env": {"headless": False, "grid_size": 128},
                "model": {"gamma": 0.99, "ent_coef": 0.01, "n_epochs": 10, "policy": "MlpPolicy"}
            }
        ]
        
        manager = PipelineManager(config_dir="config")
        with patch.object(manager, "get_level_status", return_value=("NEEDS_TRAINING", None)):
             manager._train_level("level_0")
             
        # Verify watchdog was started
        mock_watchdog.return_value.start.assert_called_once()
        # Verify watchdog was stopped
        mock_watchdog.return_value.stop.assert_called_once()
        # Verify watchdog was passed to DiagnosticsCallback
        mock_diag_callback.assert_called_once_with(check_freq=1024, watchdog=mock_watchdog.return_value)

    def test_watchdog_actually_triggers_on_hang(self):
        """Test that watchdog triggers if a real hang occurs (short timeout)."""
        event = threading.Event()
        def on_trigger():
            event.set()
            
        wd = Watchdog(timeout=0.2, callback=on_trigger)
        wd.start()
        
        # Simulate a hang longer than timeout by waiting for the event
        triggered = event.wait(timeout=1.0)
        
        assert triggered, "Watchdog failed to trigger during simulated hang"
        wd.stop()
