"""Tests for policy_active buffer masking and vec_env_requires_policy_action."""
import numpy as np
import torch
from unittest.mock import MagicMock

from train.env_utils import vec_env_requires_policy_action


def test_vec_env_requires_policy_action_from_env_method():
    vec = MagicMock()
    vec.env_method = MagicMock(return_value=[True, False, True])
    arr = vec_env_requires_policy_action(vec, 3)
    assert arr.dtype == bool
    assert arr.tolist() == [True, False, True]


def test_vec_env_requires_policy_action_defaults_when_missing():
    vec = MagicMock()
    vec.env_method = MagicMock(side_effect=AttributeError("no method"))
    arr = vec_env_requires_policy_action(vec, 2)
    assert arr.tolist() == [True, True]


def test_disk_buffer_policy_active_masks_training_samples():
    from gymnasium import spaces
    from src.agent.utils.disk_buffer import DiskDictRolloutBuffer

    obs_space = spaces.Dict({
        "command": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
    })
    action_space = spaces.MultiDiscrete([2, 4])
    buf = DiskDictRolloutBuffer(
        buffer_size=4,
        observation_space=obs_space,
        action_space=action_space,
        n_envs=2,
        temp_dir="C:/temp/sc2_buffer_test_policy_active",
    )
    buf.pos = 0
    buf.add(
        obs={"command": np.zeros((2, 4), dtype=np.float32)},
        action=np.zeros((2, 2), dtype=np.float32),
        reward=np.zeros(2, dtype=np.float32),
        episode_start=np.zeros(2, dtype=np.float32),
        value=torch.zeros(2),
        log_prob=torch.zeros(2),
        policy_active=np.array([1.0, 0.0], dtype=np.float32),
    )
    assert buf.policy_active[0, 0] == 1.0
    assert buf.policy_active[0, 1] == 0.0
