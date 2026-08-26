"""Build pipeline must not call the global SC2 kill switch (concurrent training)."""

from unittest.mock import patch

import yaml

from train.train_build import BuildPipelineManager
from train.train_pipeline import PipelineManager


def test_build_pipeline_disables_global_sc2_kill_flag():
    assert BuildPipelineManager.global_sc2_kill_on_boundaries is False
    assert PipelineManager.global_sc2_kill_on_boundaries is True


def test_build_pipeline_run_skips_kill_switch(tmp_path):
    (tmp_path / "pipeline.yaml").write_text(
        yaml.dump({"levels": ["level_build_zerg_0"]}),
        encoding="utf-8",
    )
    (tmp_path / "level_build_zerg_0.yaml").write_text(
        yaml.dump(
            {
                "training": {"num_envs": 1, "training_steps": 1},
                "env": {},
                "model": {},
            }
        ),
        encoding="utf-8",
    )
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir()
    logs.mkdir()
    # Mark level complete so run() never enters _train_level
    done = models / "level_build_zerg_0"
    done.mkdir()
    (done / "final_model.zip").write_bytes(b"x")

    mgr = BuildPipelineManager(
        config_dir=str(tmp_path),
        model_dir_base=str(models),
        log_dir_base=str(logs),
    )
    with patch("train.train_pipeline.kill_switch_sc2") as mock_kill, patch(
        "train.train_pipeline.cleanup_worker_logs"
    ):
        mgr.run()
        mock_kill.assert_not_called()


def test_default_pipeline_run_still_calls_kill_switch(tmp_path):
    (tmp_path / "pipeline.yaml").write_text(
        yaml.dump({"levels": ["level_0"]}),
        encoding="utf-8",
    )
    (tmp_path / "level_0.yaml").write_text(
        yaml.dump(
            {
                "training": {"num_envs": 1, "training_steps": 1},
                "env": {},
                "model": {},
            }
        ),
        encoding="utf-8",
    )
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir()
    logs.mkdir()
    done = models / "level_0"
    done.mkdir()
    (done / "final_model.zip").write_bytes(b"x")

    mgr = PipelineManager(
        config_dir=str(tmp_path),
        model_dir_base=str(models),
        log_dir_base=str(logs),
    )
    with patch("train.train_pipeline.kill_switch_sc2") as mock_kill, patch(
        "train.train_pipeline.cleanup_worker_logs"
    ):
        mgr.run()
        assert mock_kill.call_count == 2  # start + end
