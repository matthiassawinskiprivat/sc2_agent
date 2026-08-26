"""PPO train must sanitize NaNs without cloning spatial tensors."""
import torch

from src.agent.maskable_recurrent_ppo import sanitize_obs_tensors_inplace


def test_sanitize_obs_tensors_inplace_dict_keeps_identity():
    maps = torch.tensor([float("nan"), 1.0, float("inf")], dtype=torch.float32)
    cmd = torch.tensor([0.5, float("-inf")], dtype=torch.float32)
    obs = {"map": maps, "command": cmd}

    out = sanitize_obs_tensors_inplace(obs)
    assert out is obs
    assert out["map"] is maps
    assert out["command"] is cmd
    assert torch.equal(maps, torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(cmd, torch.tensor([0.5, 0.0]))
