"""Build pipeline truncates TB/CSV logs when resuming from a checkpoint."""

from unittest.mock import MagicMock, patch

import yaml

from train.train_build import BuildPipelineManager


def _write_minimal_build_config(tmp_path):
    (tmp_path / "pipeline.yaml").write_text(
        yaml.dump({"levels": ["level_build_zerg_0"]}),
        encoding="utf-8",
    )
    (tmp_path / "level_build_zerg_0.yaml").write_text(
        yaml.dump(
            {
                "training": {
                    "num_envs": 1,
                    "training_steps": 100,
                    "n_steps": 64,
                    "batch_size": 32,
                    "learning_rate": 0.0003,
                    "device": "cuda:0",
                },
                "env": {
                    "env_class": "ZergBuildEnv",
                    "headless": True,
                    "max_steps": 10,
                },
                "model": {
                    "gamma": 0.99,
                    "ent_coef": 0.01,
                    "n_epochs": 1,
                    "policy_kwargs": {},
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_train_level_truncates_logs_on_resume(tmp_path):
    _write_minimal_build_config(tmp_path)
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir()
    logs.mkdir()
    ckpt = models / "level_build_zerg_0" / "build_2048_steps.zip"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")

    mgr = BuildPipelineManager(
        config_dir=str(tmp_path),
        model_dir_base=str(models),
        log_dir_base=str(logs),
    )

    with patch(
        "train.training_log_utils.truncate_logs_for_checkpoint_resume"
    ) as mock_trunc, patch(
        "train.async_train_universal.free_port"
    ) as mock_free, patch(
        "train.train_build.EnvFactory", return_value=MagicMock()
    ), patch(
        "train.train_build.MaskableSyncVecEnv", return_value=MagicMock()
    ), patch(
        "train.train_build.resolve_training_device", return_value="cuda:0"
    ), patch(
        "train.train_build.warmup_cuda_if_needed"
    ), patch(
        "train.train_build.sync_rollout_buffer_config",
        return_value=(MagicMock, {}),
    ), patch(
        "train.train_build.load_build_checkpoint"
    ) as mock_load, patch(
        "train.train_build.CheckpointCallback"
    ), patch(
        "train.train_build.CallbackList", return_value=MagicMock()
    ):
        model = MagicMock()
        mock_load.return_value = model
        mock_trunc.return_value = {"csv_rows_dropped": 1, "tb_scalars_dropped": 2}

        mgr._train_level(
            "level_build_zerg_0",
            model_path_to_load=str(ckpt),
            reset_timesteps=False,
        )

        mock_free.assert_called_once_with(6006)
        mock_trunc.assert_called_once_with(
            str(logs / "level_build_zerg_0"), str(ckpt)
        )
        model.learn.assert_called_once()


def test_build_train_level_skips_truncate_on_fresh_level(tmp_path):
    _write_minimal_build_config(tmp_path)
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir()
    logs.mkdir()

    mgr = BuildPipelineManager(
        config_dir=str(tmp_path),
        model_dir_base=str(models),
        log_dir_base=str(logs),
    )

    with patch(
        "train.training_log_utils.truncate_logs_for_checkpoint_resume"
    ) as mock_trunc, patch(
        "train.async_train_universal.free_port"
    ) as mock_free, patch(
        "train.train_build.EnvFactory", return_value=MagicMock()
    ), patch(
        "train.train_build.MaskableSyncVecEnv", return_value=MagicMock()
    ), patch(
        "train.train_build.resolve_training_device", return_value="cuda:0"
    ), patch(
        "train.train_build.warmup_cuda_if_needed"
    ), patch(
        "train.train_build.sync_rollout_buffer_config",
        return_value=(MagicMock, {}),
    ), patch(
        "train.train_build.MaskablePPO"
    ) as mock_ppo, patch(
        "train.train_build.CheckpointCallback"
    ), patch(
        "train.train_build.CallbackList", return_value=MagicMock()
    ):
        model = MagicMock()
        mock_ppo.return_value = model

        mgr._train_level(
            "level_build_zerg_0",
            model_path_to_load=None,
            reset_timesteps=True,
        )

        mock_free.assert_not_called()
        mock_trunc.assert_not_called()
        model.learn.assert_called_once()
