"""Tests for src.utils.training_device (SB3 device resolution)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from src.utils.training_device import (
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
