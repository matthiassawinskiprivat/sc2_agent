import pytest
import os
import shutil
import yaml
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from train.train_pipeline import PipelineManager, cleanup_worker_logs

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


def test_cleanup_worker_logs(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "worker_1111.log").write_text("old")
    (logs_dir / "worker_9999.log").write_text("old")
    (logs_dir / "other.log").write_text("keep")
    (logs_dir / "level_0").mkdir()

    deleted = cleanup_worker_logs(str(logs_dir))

    assert deleted == 2
    assert not (logs_dir / "worker_1111.log").exists()
    assert not (logs_dir / "worker_9999.log").exists()
    assert (logs_dir / "other.log").exists()
    assert (logs_dir / "level_0").exists()


def test_cleanup_worker_logs_missing_dir(tmp_path):
    assert cleanup_worker_logs(str(tmp_path / "no_such_logs")) == 0


@patch("train.train_pipeline.kill_switch_sc2")
@patch("train.train_pipeline.cleanup_worker_logs")
def test_pipeline_run_cleans_worker_logs_on_start(mock_cleanup_workers, mock_kill, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    log_dir_base = tmp_path / "logs"
    log_dir_base.mkdir()
    with open(config_dir / "pipeline.yaml", "w") as f:
        yaml.dump({"levels": ["level_0"]}, f)
    (model_dir / "level_0").mkdir()
    (model_dir / "level_0" / "final_model.zip").touch()

    manager = PipelineManager(
        config_dir=str(config_dir),
        model_dir_base=str(model_dir),
        log_dir_base=str(log_dir_base),
    )
    manager.run()

    mock_cleanup_workers.assert_called_once_with()

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
