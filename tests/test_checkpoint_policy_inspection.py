"""SB3 checkpoint policy weight validation for async hot-swap."""
import numpy as np
import torch

from src.utils.model_utils import (
    inspect_checkpoint_policy_weights,
    non_finite_state_dict_keys,
    policy_state_dict_finite,
)


def test_non_finite_state_dict_keys_detects_nan():
    sd = {
        "good": torch.ones(2),
        "bad": torch.tensor([1.0, float("nan")]),
    }
    assert non_finite_state_dict_keys(sd) == ["bad"]
    assert not policy_state_dict_finite(sd)


def test_inspect_checkpoint_rejects_corrupt_zip(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not-a-zip")
    ok, reason, keys = inspect_checkpoint_policy_weights(str(bad))
    assert not ok
    assert reason == "not_a_zip"
    assert keys == []


def test_inspect_checkpoint_accepts_finite_sb3_zip(tmp_path):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from gymnasium import spaces

    class _DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
            self.action_space = spaces.Discrete(2)

    model = PPO("MlpPolicy", _DummyEnv(), n_steps=64, batch_size=32)
    path = str(tmp_path / "model.zip")
    model.save(path)
    ok, reason, keys = inspect_checkpoint_policy_weights(path)
    assert ok
    assert reason == "ok"
    assert keys == []
