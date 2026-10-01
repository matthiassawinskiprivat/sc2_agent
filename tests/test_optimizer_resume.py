"""Optimizer state must survive same-level async/sync checkpoint resume only."""
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.utils.model_utils import load_optimizer_state_from_zip


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)


def _write_sb3_like_zip(path: Path, policy_sd, optimizer_sd, num_timesteps: int = 42):
    buffer = io.BytesIO()
    torch.save(policy_sd, buffer)
    policy_bytes = buffer.getvalue()
    buffer = io.BytesIO()
    torch.save(optimizer_sd, buffer)
    optimizer_bytes = buffer.getvalue()
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data", json.dumps({"num_timesteps": num_timesteps}))
        zf.writestr("policy.pth", policy_bytes)
        zf.writestr("policy.optimizer.pth", optimizer_bytes)


def test_load_optimizer_state_from_zip_restores_adam_moments(tmp_path):
    policy = _TinyPolicy()
    for p in policy.parameters():
        p.grad = torch.ones_like(p)
    policy.optimizer.step()
    before = policy.optimizer.state_dict()

    ckpt = tmp_path / "ckpt.zip"
    _write_sb3_like_zip(ckpt, policy.state_dict(), before, num_timesteps=99)

    fresh = _TinyPolicy()
    fresh.load_state_dict(policy.state_dict())
    ok, reason = load_optimizer_state_from_zip(
        MagicMock(policy=fresh), str(ckpt), "cpu"
    )
    assert ok, reason
    restored = fresh.optimizer.state_dict()
    assert restored["param_groups"] == before["param_groups"]
    for key in before["state"]:
        assert torch.allclose(
            restored["state"][key]["exp_avg"],
            before["state"][key]["exp_avg"],
        )
        assert torch.allclose(
            restored["state"][key]["exp_avg_sq"],
            before["state"][key]["exp_avg_sq"],
        )


def test_load_optimizer_pads_grown_input_columns_so_step_succeeds(tmp_path):
    """Adam moments from a 48-wide encoder must pad to 49 before step()."""
    old = nn.Linear(48, 8)
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    for param in old.parameters():
        param.grad = torch.ones_like(param)
    opt.step()
    old_exp = opt.state[old.weight]["exp_avg"].clone()
    old_sq = opt.state[old.weight]["exp_avg_sq"].clone()
    old_bias_exp = opt.state[old.bias]["exp_avg"].clone()

    ckpt = tmp_path / "wide.zip"
    _write_sb3_like_zip(ckpt, old.state_dict(), opt.state_dict())

    fresh = nn.Linear(49, 8)
    with torch.no_grad():
        fresh.weight[:, :48] = old.weight
        fresh.bias.copy_(old.bias)
    fresh.optimizer = torch.optim.Adam(fresh.parameters(), lr=1e-3)

    ok, reason = load_optimizer_state_from_zip(
        MagicMock(policy=fresh), str(ckpt), "cpu"
    )
    assert ok, reason
    exp = fresh.optimizer.state[fresh.weight]["exp_avg"]
    sq = fresh.optimizer.state[fresh.weight]["exp_avg_sq"]
    assert exp.shape == (8, 49)
    assert torch.allclose(exp[:, :48], old_exp)
    assert torch.allclose(sq[:, :48], old_sq)
    assert torch.count_nonzero(exp[:, 48]) == 0
    assert torch.count_nonzero(sq[:, 48]) == 0
    assert torch.allclose(fresh.optimizer.state[fresh.bias]["exp_avg"], old_bias_exp)

    fresh(torch.randn(2, 49)).sum().backward()
    fresh.optimizer.step()


def test_async_learner_restore_optimizer_only_on_same_level_resume():
    """Curriculum transfer loads weights but must not carry Adam from the prior level."""
    from train.async_train_universal import AsyncLearner

    config = {"training": {"save_freq": 50000}, "env": {}}
    resumed = AsyncLearner(
        "level_attack_2",
        config,
        resume_timesteps_from_save=True,
        model_path="models_universal/level_attack_2/universal_1000_steps.zip",
    )
    transferred = AsyncLearner(
        "level_attack_2",
        config,
        resume_timesteps_from_save=False,
        model_path="models_universal/level_attack_1/final_model.zip",
    )
    fresh = AsyncLearner(
        "level_attack_2",
        config,
        resume_timesteps_from_save=True,
        model_path=None,
    )

    assert resumed.restore_optimizer is True
    assert transferred.restore_optimizer is False
    assert fresh.restore_optimizer is False


def test_load_curriculum_checkpoint_skips_optimizer_by_default(tmp_path):
    """Curriculum callers must opt in to Adam restore; default is fresh optimizer."""
    from unittest.mock import MagicMock, patch

    ckpt = tmp_path / "ckpt.zip"
    _write_sb3_like_zip(ckpt, {"w": torch.zeros(1)}, {"state": {}, "param_groups": []})

    algo_class = MagicMock()
    algo_class.load.return_value = MagicMock()
    with patch(
        "src.utils.model_utils.load_optimizer_state_from_zip",
        return_value=(True, "ok"),
    ) as mock_opt:
        from src.utils.model_utils import load_curriculum_checkpoint

        load_curriculum_checkpoint(
            algo_class,
            str(ckpt),
            {},
            {},
            MagicMock(spaces={"map": MagicMock(shape=(9, 128, 128))}),
            MagicMock(nvec=[64, 1024]),
        )
    mock_opt.assert_not_called()


    ckpt = tmp_path / "empty.zip"
    with zipfile.ZipFile(ckpt, "w") as zf:
        zf.writestr("data", json.dumps({"num_timesteps": 0}))
        buffer = io.BytesIO()
        torch.save({}, buffer)
        zf.writestr("policy.pth", buffer.getvalue())

    policy = _TinyPolicy()
    ok, reason = load_optimizer_state_from_zip(
        MagicMock(policy=policy), str(ckpt), "cpu"
    )
    assert not ok
    assert reason == "no_optimizer_in_checkpoint"


@pytest.mark.skipif(
    not Path("models_universal/level_attack_2/universal_2015232_steps.zip").is_file(),
    reason="local attack_2 checkpoint required",
)
def test_load_optimizer_state_from_real_checkpoint():
    from stable_baselines3.common.save_util import load_from_zip_file

    ckpt = "models_universal/level_attack_2/universal_2015232_steps.zip"
    _, params, _ = load_from_zip_file(ckpt, device="cpu", load_data=False)
    saved_opt = params["policy.optimizer"]

    policy = _TinyPolicy()
    ok, reason = load_optimizer_state_from_zip(
        MagicMock(policy=policy), ckpt, "cpu"
    )
    assert not ok
    assert reason.startswith("load_state_dict_failed")
    assert saved_opt is not None
