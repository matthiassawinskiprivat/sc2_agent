import pytest
import os
import shutil
import yaml
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from train.train_pipeline import PipelineManager

def test_pipeline_log_cleanup(tmp_path):
    # 1. Setup temp directories
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    log_dir_base = tmp_path / "logs"
    log_dir_base.mkdir()
    
    # 2. Create pipeline.yaml
    pipeline_conf = {
        "levels": ["level_0", "level_1", "level_2"]
    }
    with open(config_dir / "pipeline.yaml", "w") as f:
        yaml.dump(pipeline_conf, f)
        
    # 3. Create dummy logs
    (log_dir_base / "level_0").mkdir()
    (log_dir_base / "level_1").mkdir()
    (log_dir_base / "level_2").mkdir()
    
    # Create some files inside
    (log_dir_base / "level_0" / "events.out").touch()
    (log_dir_base / "level_1" / "events.out").touch()
    
    # 4. Initialize Manager
    manager = PipelineManager(
        config_dir=str(config_dir),
        model_dir_base=str(model_dir),
        log_dir_base=str(log_dir_base)
    )
    
    # 5. Execute cleanup for level_1
    manager._cleanup_logs("level_1")
    
    # 6. Verify
    assert (log_dir_base / "level_0").exists(), "Level 0 logs should NOT be deleted"
    assert not (log_dir_base / "level_1").exists(), "Level 1 logs SHOULD be deleted"
    assert not (log_dir_base / "level_2").exists(), "Level 2 logs SHOULD be deleted"

@patch("train.train_pipeline.subprocess.Popen")
def test_pipeline_start_tensorboard(mock_popen, tmp_path):
    # 1. Setup temp directories
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    log_dir_base = tmp_path / "logs"
    log_dir_base.mkdir()
    
    # 2. Create pipeline.yaml
    pipeline_conf = {"levels": ["level_0"]}
    with open(config_dir / "pipeline.yaml", "w") as f:
        yaml.dump(pipeline_conf, f)
        
    # 3. Initialize Manager
    manager = PipelineManager(
        config_dir=str(config_dir),
        model_dir_base=str(model_dir),
        log_dir_base=str(log_dir_base)
    )
    
    # 4. Start TensorBoard
    manager._start_tensorboard()
    
    # 5. Verify Popen was called with correct arguments
    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    cmd = args[0]
    assert "--logdir" in cmd
    assert str(log_dir_base) in cmd
    assert "tensorboard" in " ".join(cmd).lower()
