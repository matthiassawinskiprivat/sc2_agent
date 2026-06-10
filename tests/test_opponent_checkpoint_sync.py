"""Opponent self-play must load the same staging zip the async collector writes."""
import os
from unittest.mock import MagicMock, patch

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
    env.episode_counter = 1
    env._log = MagicMock()
    return env


def test_opponent_checkpoint_candidates_include_collector_staging_name(tmp_path, monkeypatch):
    env = _make_env()
    monkeypatch.setenv("SC2_ASYNC_BUFFER_DIR", str(tmp_path))

    candidates = env._opponent_checkpoint_candidates()
    assert os.path.join(tmp_path, "latest_learner_model_collector_staging.zip") in candidates
    assert os.path.join(tmp_path, "initial_model.zip") in candidates


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


def test_sync_opponent_model_uses_explicit_path(tmp_path, monkeypatch):
    env = _make_env()
    monkeypatch.setenv("SC2_ASYNC_BUFFER_DIR", str(tmp_path))
    ckpt = tmp_path / "initial_model.zip"
    ckpt.write_bytes(b"not-a-real-zip")

    mock_model = MagicMock()
    mock_model.policy.set_training_mode = MagicMock()

    with patch(
        "src.agent.maskable_recurrent_ppo.MaskableRecurrentPPO.load",
        return_value=mock_model,
    ) as mock_load:
        ok = env.sync_opponent_model(str(ckpt))

    assert ok is True
    mock_load.assert_called_once()
    assert mock_load.call_args[0][0] == str(ckpt)
    load_kwargs = mock_load.call_args[1]
    assert load_kwargs["env"] is env
    assert load_kwargs["policy_kwargs"]["lstm_hidden_size"] == 512
    assert env.opponent_model is mock_model


def test_push_opponent_checkpoint_to_envs_calls_env_method():
    from train.async_train_universal import push_opponent_checkpoint_to_envs

    vec = MagicMock()
    push_opponent_checkpoint_to_envs(vec, r"D:\temp\async_train\initial_model.zip")
    vec.env_method.assert_called_once_with(
        "sync_opponent_model", r"D:\temp\async_train\initial_model.zip"
    )
