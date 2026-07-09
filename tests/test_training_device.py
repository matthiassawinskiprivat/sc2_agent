"""Tests for src.utils.training_device (SB3 device resolution)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from src.utils.training_device import (
    cuda_init_file_lock,
    cuda_init_lock_path,
    resolve_opponent_device,
    resolve_training_device,
    warmup_cuda_if_needed,
)


def test_resolve_auto_requires_cuda():
    with patch.dict(os.environ, {"SC2_TRAINING_DEVICE": ""}, clear=False):
        with patch("torch.cuda.is_available", return_value=True):
            d = resolve_training_device({"device": "auto"})
            assert d.type == "cuda"
            assert d.index == 0


def test_resolve_cuda_raises_when_unavailable():
    with patch.dict(os.environ, {"SC2_TRAINING_DEVICE": ""}, clear=False):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="requires CUDA"):
                resolve_training_device({"device": "auto"})


def test_resolve_cpu_explicitly_rejected():
    with pytest.raises(RuntimeError, match="not supported"):
        resolve_training_device({"device": "cpu"})


def test_resolve_invalid_raises():
    with pytest.raises(ValueError, match="Invalid training device"):
        resolve_training_device({"device": "mps"})


def test_env_sc2_training_device_overrides_yaml():
    with patch.dict(os.environ, {"SC2_TRAINING_DEVICE": "cuda"}, clear=False):
        with patch("torch.cuda.is_available", return_value=True):
            d = resolve_training_device({"device": "auto"})
            assert d.type == "cuda"


def test_opponent_device_rejects_cpu():
    with pytest.raises(RuntimeError, match="not supported"):
        resolve_opponent_device({"opponent_device": "cpu"})


def test_opponent_device_uses_cuda():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_opponent_device({}).type == "cuda"


def test_warmup_rejects_cpu():
    with pytest.raises(RuntimeError, match="does not support CPU"):
        warmup_cuda_if_needed(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_warmup_cuda_runs():
    warmup_cuda_if_needed(torch.device("cuda:0"))


def test_cuda_init_lock_path_override(tmp_path, monkeypatch):
    lock_file = tmp_path / "custom_cuda.lock"
    monkeypatch.setenv("SC2_CUDA_INIT_LOCK", str(lock_file))
    assert cuda_init_lock_path() == str(lock_file)


def test_cuda_init_file_lock_serializes(tmp_path, monkeypatch):
    lock_file = tmp_path / "cuda.lock"
    monkeypatch.setenv("SC2_CUDA_INIT_LOCK", str(lock_file))
    with cuda_init_file_lock():
        assert lock_file.is_file()
    assert not lock_file.exists()
    with cuda_init_file_lock():
        pass
    assert not lock_file.exists()
