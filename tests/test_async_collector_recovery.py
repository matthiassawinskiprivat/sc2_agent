"""Collector should rebuild vec env when a SubprocVecEnv worker dies."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_recover_collector_vec_env_rebuilds_and_resets_lstm():
    from train.async_train_universal import recover_collector_vec_env

    old_vec = MagicMock()
    model = MagicMock()
    model._last_lstm_states = "old_states"
    env_fns = [MagicMock()]
    fake_obs = {"map": np.zeros((1, 2, 2), dtype=np.float32)}
    new_vec = MagicMock()
    new_vec.reset.return_value = (fake_obs, {})

    with patch("train.async_train_universal.force_kill_sc2"), patch(
        "train.async_train_universal.time.sleep"
    ), patch(
        "train.train_pipeline.MaskableSyncVecEnv", return_value=new_vec
    ), patch(
        "train.env_utils.vec_env_obs_focus_tags",
        return_value=np.array([-1], dtype=np.int64),
    ), patch(
        "train.squad_batch_inference.ensure_collector_lstm_states"
    ) as mock_lstm:
        out_vec, obs, helper, last_dones, focus = recover_collector_vec_env(
            old_vec, model, env_fns, num_envs=1, is_recurrent=True
        )

    old_vec.close.assert_called_once()
    model.set_env.assert_called_once_with(new_vec)
    mock_lstm.assert_called_once_with(model, 1)
    assert out_vec is new_vec
    assert obs is fake_obs
    assert last_dones.shape == (1,)
    assert last_dones.all()
