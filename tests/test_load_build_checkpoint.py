"""Build-agent checkpoints must restore Adam state without weight grafting."""
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from src.agent.build_constants import (
    BUILD_MAP_RESOLUTION,
    BUILD_PLACEMENT_CELLS,
    COMMAND_VECTOR_DIM,
    MACRO_VECTOR_DIM,
    MAX_BUILD_WORKERS,
    WORKER_FEAT_DIM,
)
from src.utils.model_utils import load_build_checkpoint, load_optimizer_state_from_zip


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)


def _build_dummy_env(n_map_ch: int = 3):
    class DummyBuildEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict(
                {
                    "local_map": spaces.Box(
                        0.0,
                        1.0,
                        shape=(n_map_ch, BUILD_MAP_RESOLUTION, BUILD_MAP_RESOLUTION),
                        dtype="float32",
                    ),
                    "worker_features": spaces.Box(
                        0.0,
                        1.0,
                        shape=(MAX_BUILD_WORKERS, WORKER_FEAT_DIM),
                        dtype="float32",
                    ),
                    "macro_vector": spaces.Box(
                        0.0, 1.0, shape=(MACRO_VECTOR_DIM,), dtype="float32"
                    ),
                    "command": spaces.Box(
                        0.0, 1.0, shape=(COMMAND_VECTOR_DIM,), dtype="float32"
                    ),
                }
            )
            self.action_space = spaces.MultiDiscrete(
                [MAX_BUILD_WORKERS, BUILD_PLACEMENT_CELLS]
            )

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    return DummyBuildEnv()


def test_load_build_checkpoint_falls_back_to_policy_and_optimizer(tmp_path):
    policy = _TinyPolicy()
    for p in policy.parameters():
        p.grad = torch.ones_like(p)
    policy.optimizer.step()
    opt_sd = policy.optimizer.state_dict()

    ckpt = tmp_path / "build.zip"
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    policy_bytes = buffer.getvalue()
    buffer = io.BytesIO()
    torch.save(opt_sd, buffer)
    optimizer_bytes = buffer.getvalue()
    with zipfile.ZipFile(ckpt, "w") as zf:
        zf.writestr("data", json.dumps({"num_timesteps": 128}))
        zf.writestr("policy.pth", policy_bytes)
        zf.writestr("policy.optimizer.pth", optimizer_bytes)

    model_kwargs = {"policy": MagicMock(), "env": MagicMock(), "verbose": 0}

    class FakePPO:
        def __init__(self, **kwargs):
            self.policy = _TinyPolicy()

        @classmethod
        def load(cls, path, **kwargs):
            raise RuntimeError("metadata mismatch")

    with patch("src.agent.maskable_ppo.MaskablePPO", FakePPO):
        model = load_build_checkpoint(str(ckpt), model_kwargs, load_kwargs={"device": "cpu"})

    ok, reason = load_optimizer_state_from_zip(model, str(ckpt), "cpu")
    assert ok, reason


@pytest.mark.skipif(
    not Path("models_build_zerg/level_build_zerg_0/final_model.zip").is_file(),
    reason="local build checkpoint required",
)
def test_load_build_checkpoint_from_real_zip():
    from src.agent.maskable_ppo import MaskablePPO
    from src.agent.models.build_policy import BuildActorCriticPolicy
    from src.utils.rollout_buffer_factory import sync_rollout_buffer_config

    env = _build_dummy_env()
    training = {
        "learning_rate": 0.0003,
        "batch_size": 512,
        "n_steps": 2048,
    }
    env_section = {"headless": True}
    rollout_buffer_class, rollout_buffer_kwargs = sync_rollout_buffer_config(
        training, env_section, namespace="build"
    )
    rollout_buffer_kwargs["enable_rnn_sampling"] = False
    model_kwargs = {
        "policy": BuildActorCriticPolicy,
        "env": env,
        "verbose": 0,
        "learning_rate": training["learning_rate"],
        "batch_size": training["batch_size"],
        "n_steps": training["n_steps"],
        "gamma": 0.99,
        "ent_coef": 0.01,
        "n_epochs": 4,
        "policy_kwargs": {
            "features_extractor_kwargs": {
                "features_dim": 128,
                "thinking_steps": 3,
                "map_resolution": BUILD_MAP_RESOLUTION,
            }
        },
        "device": "cpu",
        "rollout_buffer_class": rollout_buffer_class,
        "rollout_buffer_kwargs": rollout_buffer_kwargs,
    }

    ckpt = "models_build_zerg/level_build_zerg_0/final_model.zip"
    model = load_build_checkpoint(
        ckpt,
        model_kwargs,
        load_kwargs={"env": env, "device": "cpu"},
    )
    from stable_baselines3.common.save_util import load_from_zip_file

    _, params, _ = load_from_zip_file(ckpt, device="cpu", load_data=False)
    saved_opt = params["policy.optimizer"]
    restored = model.policy.optimizer.state_dict()
    for key in saved_opt["state"]:
        assert torch.allclose(
            restored["state"][key]["exp_avg"],
            saved_opt["state"][key]["exp_avg"],
        )
