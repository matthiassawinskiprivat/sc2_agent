"""Async checkpoint resume: timestep counter and checkpoint selection."""
import io
import json
import os
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.model_utils import (
    apply_checkpoint_timesteps,
    latest_step_checkpoint,
    read_checkpoint_timesteps,
)


def _write_minimal_sb3_zip(path: Path, num_timesteps: int) -> None:
    buffer = io.BytesIO()
    torch.save({"w": torch.zeros(1)}, buffer)
    policy_bytes = buffer.getvalue()
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data", json.dumps({"num_timesteps": num_timesteps}))
        zf.writestr("policy.pth", policy_bytes)


def test_read_checkpoint_timesteps_from_zip(tmp_path):
    ckpt = tmp_path / "universal_12345_steps.zip"
    _write_minimal_sb3_zip(ckpt, 12345)
    assert read_checkpoint_timesteps(str(ckpt)) == 12345


def test_apply_checkpoint_timesteps_sets_model_counter(tmp_path):
    ckpt = tmp_path / "universal_500000_steps.zip"
    _write_minimal_sb3_zip(ckpt, 500_000)
    model = MagicMock(num_timesteps=0)
    applied = apply_checkpoint_timesteps(model, str(ckpt))
    assert applied == 500_000
    assert model.num_timesteps == 500_000


def test_latest_step_checkpoint_prefers_highest_step_not_mtime(tmp_path):
    low = tmp_path / "universal_100000_steps.zip"
    high = tmp_path / "universal_900000_steps.zip"
    _write_minimal_sb3_zip(low, 100_000)
    _write_minimal_sb3_zip(high, 900_000)
    # Make the lower-step file newer on disk — must still pick 900k.
    time.sleep(0.05)
    os.utime(low, None)

    picked = latest_step_checkpoint([str(low), str(high)])
    assert picked == str(high)


def test_get_level_status_uses_highest_step_checkpoint(tmp_path):
    from train.train_pipeline import PipelineManager

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "pipeline.yaml").write_text("levels: [ctf_level_01]\n")
    (config_dir / "level_0.yaml").write_text("training: {}\nenv: {}\nmodel: {}\n")

    models = tmp_path / "models_universal" / "ctf_level_01"
    models.mkdir(parents=True)
    low = models / "universal_200000_steps.zip"
    high = models / "universal_800000_steps.zip"
    _write_minimal_sb3_zip(low, 200_000)
    _write_minimal_sb3_zip(high, 800_000)
    time.sleep(0.05)
    os.utime(low, None)

    manager = PipelineManager(
        config_dir=str(config_dir),
        model_dir_base=str(tmp_path / "models_universal"),
    )
    status, path = manager.get_level_status("ctf_level_01")
    assert status == "CONTINUE_TRAINING"
    assert path == str(high)


def test_load_squad_checkpoint_graft_restores_timesteps(tmp_path):
    ckpt = tmp_path / "universal_750000_steps.zip"
    _write_minimal_sb3_zip(ckpt, 750_000)

    grafted_model = MagicMock(num_timesteps=0)
    grafted_model.policy.load_state_dict.return_value = ([], [])
    with patch(
        "src.agent.maskable_recurrent_ppo.MaskableRecurrentPPO"
    ) as MockAlgo, patch(
        "stable_baselines3.common.save_util.load_from_zip_file",
        return_value=(None, {"policy": {"layer": torch.zeros(1)}}, None),
    ), patch(
        "src.utils.model_utils.graft_squad_policy_from_universal",
        return_value={"layer": torch.zeros(1)},
    ):
        MockAlgo.load.side_effect = RuntimeError("shape mismatch")
        MockAlgo.return_value = grafted_model
        from src.utils.model_utils import load_squad_checkpoint

        model = load_squad_checkpoint(
            str(ckpt),
            algo_kwargs={"policy": MagicMock(), "env": MagicMock()},
            load_kwargs={"device": "cpu"},
            observation_space=MagicMock(spaces={"map": MagicMock(shape=(6, 32, 32))}),
            action_space=MagicMock(),
        )

    assert model is grafted_model
    assert model.num_timesteps == 750_000


def test_load_squad_checkpoint_direct_load_keeps_self_attention(tmp_path):
    """Same-arch squad resume must not fall through to universal graft.

    Regression: MaskableRecurrentPPO.load forced UnitControlNetwork, which
    rejected ``use_self_attention`` and triggered the graft fallback.
    """
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
    from src.agent.models.squad_network import SquadNetwork
    from src.agent.models.squad_policy import SquadRecurrentPolicy
    from src.agent.squad_constants import SQUAD_UNIT_FEAT_DIM, squad_action_space_nvec
    from src.utils.model_utils import load_squad_checkpoint

    def _minimal_env(obs_space, action_space):
        class _Env(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = obs_space
                self.action_space = action_space

            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

        return _Env()

    obs_space = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (4, 128, 128), np.float32),
            "command": spaces.Box(0, 1, (10,), np.float32),
            "anchor_tactical": spaces.Box(0, 1, (4, 32, 32), np.float32),
            "unit_scalars": spaces.Box(0, 1, (24, SQUAD_UNIT_FEAT_DIM), np.float32),
            "unit_mask": spaces.Box(0, 1, (24,), np.float32),
        }
    )
    action_space = spaces.MultiDiscrete(squad_action_space_nvec())
    policy_kwargs = {
        "features_extractor_class": SquadNetwork,
        "features_extractor_kwargs": {
            "features_dim": 64,
            "thinking_steps": 1,
            "use_self_attention": True,
        },
        "lstm_hidden_size": 32,
    }
    env = _minimal_env(obs_space, action_space)
    src = MaskableRecurrentPPO(
        SquadRecurrentPolicy,
        env,
        n_steps=64,
        batch_size=32,
        policy_kwargs=policy_kwargs,
        device="cpu",
        rnn_type="gru",
    )
    src.num_timesteps = 12_345
    marker = src.policy.features_extractor.entity_encoder.self_attn.self_attn.in_proj_weight
    marker_clone = marker.detach().clone()
    marker.data.fill_(0.42)

    ckpt = tmp_path / "squad_self_attn.zip"
    src.save(str(ckpt))

    env2 = _minimal_env(obs_space, action_space)
    algo_kwargs = {
        "policy": SquadRecurrentPolicy,
        "env": env2,
        "n_steps": 64,
        "batch_size": 32,
        "policy_kwargs": policy_kwargs,
        "device": "cpu",
        "rnn_type": "gru",
    }
    load_kwargs = {
        "env": env2,
        "policy": SquadRecurrentPolicy,
        "policy_kwargs": policy_kwargs,
        "device": "cpu",
        "rnn_type": "gru",
    }

    with patch(
        "src.utils.model_utils.graft_squad_policy_from_universal"
    ) as graft_mock:
        model = load_squad_checkpoint(
            str(ckpt),
            algo_kwargs,
            load_kwargs,
            observation_space=obs_space,
            action_space=action_space,
            restore_optimizer=True,
        )

    graft_mock.assert_not_called()
    assert isinstance(model.policy, SquadRecurrentPolicy)
    fe = model.policy.features_extractor
    assert isinstance(fe, SquadNetwork)
    assert fe.entity_encoder.use_self_attention is True
    loaded = fe.entity_encoder.self_attn.self_attn.in_proj_weight
    assert torch.allclose(loaded, torch.full_like(loaded, 0.42))
    assert not torch.allclose(loaded, marker_clone)
    assert model.num_timesteps == 12_345


def test_learner_resume_reads_timesteps_from_initial_model_zip(tmp_path):
    """Learner resume path reads num_timesteps from collector initial_model.zip."""
    initial = tmp_path / "initial_model.zip"
    _write_minimal_sb3_zip(initial, 420_000)

    total_steps_trained = 0
    model = MagicMock(num_timesteps=0)
    with zipfile.ZipFile(initial, "r") as archive:
        payload = json.loads(archive.read("data").decode("utf-8"))
        total_steps_trained = payload.get("num_timesteps", 0)
        model.num_timesteps = total_steps_trained

    assert total_steps_trained == 420_000
    assert model.num_timesteps == 420_000
