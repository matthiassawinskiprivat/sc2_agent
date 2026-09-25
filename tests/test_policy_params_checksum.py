"""Policy checksum pairing for async rollout_policy.zip."""
from __future__ import annotations

import zipfile
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.model_utils import (
    policy_params_checksum,
    rollout_policy_zip_checksum,
)


def test_policy_params_checksum_stable_and_sensitive():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2, bias=False)

    a = Tiny()
    b = Tiny()
    b.load_state_dict(a.state_dict())
    assert policy_params_checksum(a) == policy_params_checksum(b)
    with torch.no_grad():
        a.fc.weight.add_(0.1)
    assert policy_params_checksum(a) != policy_params_checksum(b)


def test_policy_params_checksum_device_invariant():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4, bias=False)

    cpu = Tiny()
    if not torch.cuda.is_available():
        return
    gpu = Tiny().cuda()
    gpu.load_state_dict(cpu.state_dict())
    assert policy_params_checksum(cpu) == policy_params_checksum(gpu)


def test_rollout_policy_zip_checksum(tmp_path: Path):
    path = tmp_path / "rollout_policy.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("policy.pth", b"fake-policy-bytes-v1")
        zf.writestr("data", b"{}")
    a = rollout_policy_zip_checksum(str(path))
    b = rollout_policy_zip_checksum(str(path))
    assert a == b
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("policy.pth", b"fake-policy-bytes-v2")
        zf.writestr("data", b"{}")
    assert rollout_policy_zip_checksum(str(path)) != a
