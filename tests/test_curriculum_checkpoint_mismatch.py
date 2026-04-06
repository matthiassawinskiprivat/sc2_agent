"""Tests for curriculum transfer when observation/action shapes grow (e.g. move -> attack)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gymnasium import spaces

from src.agent.maskable_recurrent_ppo import checkpoint_obs_action_mismatch_env


def _make_dict_obs(map_c: int, tactical_c: int) -> spaces.Dict:
    return spaces.Dict(
        {
            "map": spaces.Box(low=0.0, high=1.0, shape=(map_c, 128, 128), dtype=np.float32),
            "tactical": spaces.Box(low=0.0, high=1.0, shape=(tactical_c, 32, 32), dtype=np.float32),
            "command": spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),
        }
    )


def test_checkpoint_mismatch_true_when_map_channels_grow():
    env = MagicMock()
    env.observation_space = _make_dict_obs(9, 9)
    env.action_space = spaces.MultiDiscrete([3, 1024])

    saved = {
        "observation_space": _make_dict_obs(5, 5),
        "action_space": spaces.MultiDiscrete([2, 1024]),
    }
    with patch(
        "stable_baselines3.common.save_util.load_from_zip_file",
        return_value=(saved, {}, None),
    ):
        assert checkpoint_obs_action_mismatch_env("dummy.zip", env) is True


def test_checkpoint_mismatch_true_when_command_heads_grow():
    env = MagicMock()
    env.observation_space = _make_dict_obs(5, 5)
    env.action_space = spaces.MultiDiscrete([3, 1024])

    saved = {
        "observation_space": _make_dict_obs(5, 5),
        "action_space": spaces.MultiDiscrete([2, 1024]),
    }
    with patch(
        "stable_baselines3.common.save_util.load_from_zip_file",
        return_value=(saved, {}, None),
    ):
        assert checkpoint_obs_action_mismatch_env("dummy.zip", env) is True


def test_checkpoint_mismatch_false_when_identical_shapes():
    env = MagicMock()
    env.observation_space = _make_dict_obs(5, 5)
    env.action_space = spaces.MultiDiscrete([2, 1024])

    saved = {
        "observation_space": _make_dict_obs(5, 5),
        "action_space": spaces.MultiDiscrete([2, 1024]),
    }
    with patch(
        "stable_baselines3.common.save_util.load_from_zip_file",
        return_value=(saved, {}, None),
    ):
        assert checkpoint_obs_action_mismatch_env("dummy.zip", env) is False


def test_checkpoint_mismatch_false_when_peek_fails():
    env = MagicMock()
    env.observation_space = _make_dict_obs(9, 9)
    env.action_space = spaces.MultiDiscrete([3, 1024])

    with patch(
        "stable_baselines3.common.save_util.load_from_zip_file",
        side_effect=OSError("no file"),
    ):
        assert checkpoint_obs_action_mismatch_env("missing.zip", env) is False
