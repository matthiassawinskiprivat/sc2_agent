"""Subproc worker must survive env.reset() failures after terminal steps."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gymnasium import spaces

from train.sb3_subproc_worker_safe import _reset_env_with_retry, _zero_observation


def test_reset_env_with_retry_recovers_after_transient_failure():
    env = MagicMock()
    env.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
    call_count = {"n": 0}

    def flaky_reset(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("SC2 relaunch failed")
        return np.ones(3, dtype=np.float32), {}

    env.reset.side_effect = flaky_reset

    with patch("train.sb3_subproc_worker_safe.time.sleep"):
        obs, info = _reset_env_with_retry(env)

    assert np.allclose(obs, np.ones(3))
    assert call_count["n"] == 2
    assert env.close.call_count == 1


def test_reset_env_with_retry_returns_zero_obs_when_exhausted():
    env = MagicMock()
    env.observation_space = spaces.Dict(
        {
            "map": spaces.Box(low=0, high=1, shape=(2, 4, 4), dtype=np.float32),
            "command": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
        }
    )
    env.reset.side_effect = RuntimeError("SC2 won't start")

    with patch("train.sb3_subproc_worker_safe.time.sleep"):
        obs, info = _reset_env_with_retry(env)

    assert set(obs.keys()) == {"map", "command"}
    assert np.all(obs["map"] == 0)
    assert info.get("sc2_reset_failed") is True


def test_zero_observation_matches_dict_space():
    env = MagicMock()
    env.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
    obs = _zero_observation(env)
    assert obs.shape == (5,)
