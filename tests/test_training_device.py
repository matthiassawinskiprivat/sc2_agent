"""Tests for src.utils.training_device (SB3 device resolution)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from src.utils.training_device import resolve_training_device, warmup_cuda_if_needed


def test_resolve_explicit_cpu():
    assert resolve_training_device({"device": "cpu"}).type == "cpu"


def test_resolve_invalid_raises():
    with pytest.raises(ValueError, match="Invalid training device"):
        resolve_training_device({"device": "mps"})


def test_resolve_cuda_raises_when_unavailable():
    with patch.dict(os.environ, {"SC2_TRAINING_DEVICE": ""}, clear=False):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA"):
                resolve_training_device({"device": "cuda"})


def test_resolve_auto_uses_cuda_when_available():
    with patch.dict(os.environ, {"SC2_TRAINING_DEVICE": ""}, clear=False):
        with patch("torch.cuda.is_available", return_value=True):
            d = resolve_training_device({"device": "auto"})
            assert d.type == "cuda"
            assert d.index == 0


def test_env_sc2_training_device_overrides_yaml():
    with patch.dict(os.environ, {"SC2_TRAINING_DEVICE": "cpu"}, clear=False):
        d = resolve_training_device({"device": "auto"})
        assert d.type == "cpu"


def test_warmup_cpu_no_op():
    warmup_cuda_if_needed(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_warmup_cuda_runs():
    warmup_cuda_if_needed(torch.device("cuda:0"))
