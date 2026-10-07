"""Tests for src.utils.training_device (SB3 device resolution)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from src.utils.training_device import (
    clear_stale_training_locks,
    cuda_init_file_lock,
    cuda_init_lock_path,
    opponent_load_file_lock,
    opponent_load_lock_path,
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
        assert lock_file.read_text(encoding="utf-8").strip() == str(os.getpid())
    assert not lock_file.exists()
    with cuda_init_file_lock():
        pass
    assert not lock_file.exists()


def test_clear_stale_training_locks_removes_empty_and_dead_owner(tmp_path, monkeypatch):
    import time as time_mod

    from src.utils import training_device as td

    cuda_lock = tmp_path / "cuda.lock"
    opp_lock = tmp_path / "opp.lock"
    live_lock = tmp_path / "live.lock"
    monkeypatch.setenv("SC2_CUDA_INIT_LOCK", str(cuda_lock))
    monkeypatch.setenv("SC2_OPPONENT_LOAD_LOCK", str(opp_lock))
    cuda_lock.write_text("", encoding="utf-8")
    # Empty locks need age > grace before clear_stale will unlink them.
    old = time_mod.time() - (td._LOCK_EMPTY_GRACE_SEC + 1.0)
    os.utime(cuda_lock, (old, old))
    opp_lock.write_text("99999999\n", encoding="utf-8")  # almost certainly dead
    live_lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    cleared = clear_stale_training_locks()
    assert str(cuda_lock) in cleared
    assert str(opp_lock) in cleared
    assert not cuda_lock.exists()
    assert not opp_lock.exists()
    assert live_lock.exists()


def test_clear_stale_training_locks_keeps_fresh_empty_lock(tmp_path, monkeypatch):
    cuda_lock = tmp_path / "cuda.lock"
    opp_lock = tmp_path / "opp.lock"
    monkeypatch.setenv("SC2_CUDA_INIT_LOCK", str(cuda_lock))
    monkeypatch.setenv("SC2_OPPONENT_LOAD_LOCK", str(opp_lock))
    cuda_lock.write_text("", encoding="utf-8")
    cleared = clear_stale_training_locks()
    assert cleared == []
    assert cuda_lock.exists()


def test_opponent_lock_steals_stale_owner(tmp_path, monkeypatch):
    lock_file = tmp_path / "opp.lock"
    monkeypatch.setenv("SC2_OPPONENT_LOAD_LOCK", str(lock_file))
    lock_file.write_text("99999999\n", encoding="utf-8")
    with opponent_load_file_lock(timeout=2.0):
        assert lock_file.is_file()
        assert lock_file.read_text(encoding="utf-8").strip() == str(os.getpid())
    assert not lock_file.exists()
    assert opponent_load_lock_path() == str(lock_file)
