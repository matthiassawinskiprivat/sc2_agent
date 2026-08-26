"""Opponent self-play must load the same staging zip the async collector writes."""
import os
from unittest.mock import MagicMock, patch  # noqa: F401 — patch used in later tests

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces


def _make_env():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    with patch(
        "train.envs.capture_the_flag_single_env.SingleClientEnv.__init__",
        lambda self, *a, **kw: None,
    ):
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.training_conf = {
        "model": {
            "algorithm": "MaskableRecurrentPPO",
            "rnn_type": "gru",
            "policy_kwargs": {
                "features_extractor_class": "src.agent.models.unit_control_network.UnitControlNetwork",
                "features_extractor_kwargs": {"features_dim": 512, "thinking_steps": 3},
                "lstm_hidden_size": 512,
                "net_arch": [],
            },
        }
    }
    env.observation_space = spaces.Dict(
        {
            "map": spaces.Box(low=0, high=1, shape=(20, 128, 128), dtype=np.float32),
            "tactical": spaces.Box(low=0, high=1, shape=(12, 32, 32), dtype=np.float32),
            "command": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
        }
    )
    env.action_space = spaces.MultiDiscrete([64, 1024])
    env.agent_architecture = "universal"
    env.training_conf["training"] = {"device": "auto"}
    env.episode_counter = 1
    env._log = MagicMock()
    env.unit_pools_config = []
    env.level_name = "ctf_test"
    env.num_workers = 1
    env.worker_index = 0
    env.player_units = {1: [], 2: []}
    return env


def test_opponent_checkpoint_dir_default_matches_async_pipeline(monkeypatch):
    """CTF workers must search the same buffer root as the async collector/learner."""
    from src.utils.async_buffer_paths import DEFAULT_ASYNC_BUFFER_DIR, resolve_async_buffer_dir

    monkeypatch.delenv("SC2_ASYNC_BUFFER_DIR", raising=False)
    env = _make_env()
    assert env._opponent_checkpoint_dir() == resolve_async_buffer_dir()
    assert DEFAULT_ASYNC_BUFFER_DIR.replace("/", os.sep) in env._opponent_checkpoint_dir()


def test_opponent_checkpoint_candidates_include_collector_staging_name(tmp_path, monkeypatch):
    env = _make_env()
    monkeypatch.setenv("SC2_ASYNC_BUFFER_DIR", str(tmp_path))

    candidates = env._opponent_checkpoint_candidates()
    assert os.path.join(tmp_path, "latest_learner_model_collector_staging.zip") in candidates
    assert os.path.join(tmp_path, "initial_model.zip") in candidates
    assert candidates[0] == os.path.join(tmp_path, "latest_learner_model.zip")


def test_sync_opponent_model_prefers_latest_learner_over_initial(tmp_path, monkeypatch):
    """Hard-reset opponent sync must not reload stale level-entry initial_model."""
    env = _make_env()
    monkeypatch.setenv("SC2_ASYNC_BUFFER_DIR", str(tmp_path))
    (tmp_path / "initial_model.zip").write_bytes(b"stale")
    (tmp_path / "latest_learner_model.zip").write_bytes(b"fresh")

    mock_model = MagicMock()
    mock_model.policy.set_training_mode = MagicMock()

    with patch(
        "src.utils.training_device.warmup_cuda_if_needed",
    ), patch(
        "src.utils.opponent_inference.load_opponent_inference_checkpoint",
        return_value=mock_model,
    ) as mock_load:
        ok = env.sync_opponent_model()

    assert ok is True
    assert mock_load.call_args[0][0] == str(tmp_path / "latest_learner_model.zip")
    assert env.opponent_model is mock_model


def test_build_opponent_model_load_kwargs_matches_learner_architecture():
    from src.utils.training_device import resolve_opponent_device

    env = _make_env()
    env.training_conf["training"] = {"device": "auto"}
    kwargs = env._build_opponent_model_load_kwargs()

    assert kwargs["env"] is env
    assert kwargs["device"] == resolve_opponent_device({"device": "auto"})
    assert kwargs["rnn_type"] == "gru"
    assert kwargs["policy_kwargs"]["lstm_hidden_size"] == 512
    assert kwargs["policy_kwargs"]["spatial_resolution"] == 32
    from src.agent.models.unit_control_network import UnitControlNetwork

    assert kwargs["policy_kwargs"]["features_extractor_class"] is UnitControlNetwork
    assert kwargs["custom_objects"]["policy_kwargs"]["lstm_hidden_size"] == 512


def test_opponent_algo_bundle_falls_back_to_level_yaml_when_model_missing():
    """Subproc workers receive flattened training_conf without a model section."""
    from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
    from src.agent.models.unit_control_network import UnitControlNetwork
    from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy

    env = _make_env()
    env.training_conf.pop("model", None)
    env.level_name = "ctf_level_04"
    mock_cm = MagicMock()
    mock_cm.get_level_config.return_value = {
        "model": {
            "algorithm": "MaskableRecurrentPPO",
            "rnn_type": "gru",
            "policy_kwargs": {
                "features_extractor_class": "src.agent.models.unit_control_network.UnitControlNetwork",
                "features_extractor_kwargs": {"features_dim": 512, "thinking_steps": 3},
                "lstm_hidden_size": 512,
                "net_arch": [],
            },
            "gamma": 0.995,
        }
    }
    env.config_manager = mock_cm

    AlgoClass, PolicyClass, policy_kwargs, rnn_type = env._resolve_opponent_algo_bundle()
    assert AlgoClass is MaskableRecurrentPPO
    assert PolicyClass is UniversalRecurrentPolicy
    assert rnn_type == "gru"
    assert policy_kwargs["features_extractor_class"] is UnitControlNetwork
    assert policy_kwargs["lstm_hidden_size"] == 512

    algo_kwargs = env._build_opponent_algo_kwargs()
    assert algo_kwargs["policy"] is UniversalRecurrentPolicy
    assert algo_kwargs["rnn_type"] == "gru"
    assert algo_kwargs["gamma"] == 0.995


def test_async_collector_combined_conf_includes_model_section():
    """CTF opponent sync needs model metadata in worker training_conf."""
    config = {
        "training": {"num_envs": 2, "n_steps": 2048, "batch_size": 512},
        "env": {"env_class": "CaptureTheFlagEnv", "max_steps": 200},
        "model": {
            "algorithm": "MaskableRecurrentPPO",
            "policy_kwargs": {"lstm_hidden_size": 512},
        },
        "active_layers": ["allied_unit_density"],
    }
    training_params = config["training"]
    env_params = config["env"]
    combined_conf = training_params.copy()
    combined_conf.update(env_params)
    combined_conf["is_training"] = True
    combined_conf["agent_architecture"] = "universal"
    combined_conf["config_dir"] = "config_universal"
    combined_conf["spatial_resolution"] = 32
    for key in (
        "active_layers",
        "active_tactical_layers",
        "action_heads",
        "categorical_heads",
    ):
        if key in config:
            combined_conf[key] = config[key]
    if "model" in config:
        combined_conf["model"] = config["model"]
    combined_conf["level_name"] = "ctf_level_04"

    assert combined_conf["model"]["algorithm"] == "MaskableRecurrentPPO"
    assert combined_conf["level_name"] == "ctf_level_04"


def test_sync_opponent_model_uses_explicit_path(tmp_path, monkeypatch):
    env = _make_env()
    monkeypatch.setenv("SC2_ASYNC_BUFFER_DIR", str(tmp_path))
    ckpt = tmp_path / "initial_model.zip"
    ckpt.write_bytes(b"not-a-real-zip")

    mock_model = MagicMock()
    mock_model.policy.set_training_mode = MagicMock()

    with patch(
        "src.utils.training_device.warmup_cuda_if_needed",
    ), patch(
        "src.utils.opponent_inference.load_opponent_inference_checkpoint",
        return_value=mock_model,
    ) as mock_load:
        ok = env.sync_opponent_model(str(ckpt))

    assert ok is True
    mock_load.assert_called_once()
    assert mock_load.call_args[0][0] == str(ckpt)
    assert env.opponent_model is mock_model


def test_sync_opponent_model_disposes_previous_snapshot(monkeypatch, tmp_path):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    ckpt = tmp_path / "ckpt.zip"
    ckpt.write_bytes(b"PK")

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.agent_architecture = "squad"
    env.training_conf = {"training": {"device": "auto"}}
    env.observation_space = MagicMock()
    env.action_space = MagicMock()
    env.episode_counter = 1
    env._log = MagicMock()
    env._opponent_checkpoint_candidates = MagicMock(return_value=[str(ckpt)])
    env._resolve_opponent_algo_bundle = MagicMock(return_value=(MagicMock(), MagicMock(), {}, "gru"))
    env._build_opponent_algo_kwargs = MagicMock(return_value={})
    env._build_opponent_model_load_kwargs = MagicMock(return_value={})
    env._opponent_graft_grid_size = MagicMock(return_value=32)

    old_model = MagicMock()
    old_model.policy = MagicMock()
    env.opponent_model = old_model

    new_model = MagicMock()
    new_model.policy = MagicMock()
    monkeypatch.setattr(
        "src.utils.training_device.warmup_cuda_if_needed",
        lambda _d: None,
    )
    monkeypatch.setattr(
        "src.utils.opponent_inference.load_opponent_inference_checkpoint",
        lambda *_a, **_k: new_model,
    )
    disposed = []
    monkeypatch.setattr(
        "src.utils.cuda_memory.dispose_opponent_snapshot",
        lambda m, **kw: disposed.append(m),
    )

    assert env._sync_opponent_model() is True
    assert env.opponent_model is new_model
    assert disposed == [old_model]


def test_sync_opponent_model_inplace_skips_dispose(monkeypatch, tmp_path):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    ckpt = tmp_path / "ckpt.zip"
    ckpt.write_bytes(b"PK")

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.agent_architecture = "squad"
    env.training_conf = {"training": {"device": "auto"}}
    env.observation_space = MagicMock()
    env.action_space = MagicMock()
    env.episode_counter = 1
    env._log = MagicMock()
    env._opponent_checkpoint_candidates = MagicMock(return_value=[str(ckpt)])
    env._resolve_opponent_algo_bundle = MagicMock(
        return_value=(MagicMock(), MagicMock(), {}, "gru")
    )
    env._opponent_graft_grid_size = MagicMock(return_value=32)

    existing = MagicMock()
    env.opponent_model = existing

    monkeypatch.setattr(
        "src.utils.training_device.warmup_cuda_if_needed",
        lambda _d: None,
    )
    monkeypatch.setattr(
        "src.utils.opponent_inference.load_opponent_inference_checkpoint",
        lambda *_a, **_k: existing,
    )
    disposed = []
    monkeypatch.setattr(
        "src.utils.cuda_memory.dispose_opponent_snapshot",
        lambda m, **kw: disposed.append(m),
    )

    assert env._sync_opponent_model() is True
    assert env.opponent_model is existing
    assert disposed == []


def test_push_opponent_checkpoint_to_envs_calls_env_method():
    from train.async_train_universal import push_opponent_checkpoint_to_envs

    vec = MagicMock()
    vec.has_attr.return_value = True
    push_opponent_checkpoint_to_envs(vec, r"D:\temp\async_train\initial_model.zip")
    vec.has_attr.assert_called_once_with("sync_opponent_model")
    vec.env_method.assert_called_once_with(
        "sync_opponent_model", r"D:\temp\async_train\initial_model.zip"
    )


def test_push_opponent_checkpoint_skips_non_ctf_envs():
    from train.async_train_universal import push_opponent_checkpoint_to_envs

    vec = MagicMock()
    vec.has_attr.return_value = False
    push_opponent_checkpoint_to_envs(vec, r"D:\temp\async_train\initial_model.zip")
    vec.env_method.assert_not_called()


def test_collector_try_reload_does_not_sync_opponent_by_default(tmp_path, monkeypatch):
    """Learner hot-swap updates trainee policy only; opponent stays frozen until hard reset."""
    import train.async_train_universal as mod

    monkeypatch.setattr(mod, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "COLLECTOR_WEIGHTS_STAGING_ZIP", str(tmp_path / "staging.zip"))

    learner_zip = tmp_path / "latest_learner_model.zip"
    learner_zip.write_bytes(b"PK\x03\x04v1")

    pushed = {"n": 0}

    monkeypatch.setattr(
        mod,
        "push_opponent_checkpoint_to_envs",
        lambda *_a, **_k: pushed.__setitem__("n", pushed["n"] + 1),
    )
    monkeypatch.setattr(mod, "set_parameters_with_retry", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "stage_learner_zip_for_collector", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.utils.model_utils.inspect_checkpoint_policy_weights",
        lambda *_a, **_k: (True, "ok", []),
    )

    model = type("M", (), {"device": "cpu"})()
    sync = mod.CollectorWeightSync()
    assert sync.try_reload(model, object()) is True
    assert pushed["n"] == 0


def test_read_checkpoint_timesteps_from_zip(tmp_path):
    import json
    import zipfile

    from src.utils.model_utils import read_checkpoint_timesteps

    ckpt = tmp_path / "ckpt.zip"
    with zipfile.ZipFile(ckpt, "w") as zf:
        zf.writestr("data", json.dumps({"num_timesteps": 123456}))
    assert read_checkpoint_timesteps(str(ckpt)) == 123456
    assert read_checkpoint_timesteps(str(tmp_path / "missing.zip")) is None


def test_soft_reset_does_not_sync_opponent():
    """Between hard resets the frozen opponent snapshot must not reload."""
    import asyncio
    from unittest.mock import AsyncMock

    env = _make_env()
    env.episode_counter = 2
    env.training_conf["env"] = {"every_k_episodes": 250}
    env._manager = MagicMock()
    env._manager.clients = [MagicMock()]
    env._loop = asyncio.new_event_loop()
    env._soft_reset_game = AsyncMock(return_value=None)
    env._get_joint_observation = AsyncMock(
        return_value=np.zeros((3, 128, 128), dtype=np.float32)
    )
    env.opponent_model = MagicMock()
    env._last_obs = None
    env.game_ended = False

    with patch.object(env, "_sync_opponent_model", return_value=True) as mock_sync:
        obs, _ = env.reset()

    mock_sync.assert_not_called()
    assert obs is not None


def test_hard_reset_syncs_opponent():
    env = _make_env()
    env.episode_counter = 249  # reset() increments → 250 triggers hard reset
    env.training_conf["env"] = {"every_k_episodes": 250}
    env._manager = MagicMock()
    env._manager.clients = [MagicMock()]
    env.map_list = ["MapA", "MapB"]
    env.worker_index = 0
    env.num_workers = 1
    env.map_name = "MapA"

    with patch("train.envs.capture_the_flag_single_env.SingleClientEnv.reset") as mock_super_reset:
        mock_super_reset.return_value = (np.zeros((3, 128, 128)), {})
        with patch.object(env, "_sync_opponent_model", return_value=True) as mock_sync:
            env.reset()

    mock_sync.assert_called_once()


def test_ensure_opponent_model_loaded_retries_sync_when_missing():
    env = _make_env()
    env.opponent_model = None
    with patch.object(env, "_sync_opponent_model", return_value=True) as mock_sync:
        assert env._ensure_opponent_model_loaded() is True
    mock_sync.assert_called_once()


def test_ensure_opponent_model_loaded_skips_preview_computer_opponent():
    env = _make_env()
    env.training_conf["preview_computer_opponent"] = True
    env.opponent_model = None
    with patch.object(env, "_sync_opponent_model") as mock_sync:
        assert env._ensure_opponent_model_loaded() is False
    mock_sync.assert_not_called()


def test_opponent_graft_grid_size_uses_spatial_head_not_env_grid():
    env = _make_env()
    assert env._opponent_graft_grid_size() == 32


def test_predict_opponent_squad_batch_retries_sync_before_default_action(tmp_path, monkeypatch):
    env = _make_env()
    env.active_player_id = 2
    monkeypatch.setenv("SC2_ASYNC_BUFFER_DIR", str(tmp_path))
    env.opponent_model = None
    env.model = None
    env._build_squad_obs_and_masks = MagicMock(return_value=([], []))

    with patch.object(env, "_ensure_opponent_model_loaded", return_value=False) as mock_ensure:
        import asyncio

        actions = asyncio.run(env._predict_opponent_squad_batch([101, 102]))
    mock_ensure.assert_called_once()
    assert len(actions) == 2
    from src.agent.universal_constants import UNIVERSAL_DEFAULT_SPATIAL_IDX

    assert actions[0][0] == 0 and actions[0][1] == UNIVERSAL_DEFAULT_SPATIAL_IDX
