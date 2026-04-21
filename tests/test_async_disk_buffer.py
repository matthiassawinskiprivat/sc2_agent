"""Tests for AsyncDiskBuffer — cross-process signaling for async training."""

import json
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from gymnasium import spaces

from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM


def _create_async_buffer(tmp_path):
    """Create an AsyncDiskBuffer with minimal config in an isolated temp dir."""
    from src.agent.utils.async_disk_buffer import AsyncDiskBuffer

    obs_space = spaces.Dict({
        "map": spaces.Box(0, 1, (4, 128, 128), dtype=np.float32),
        "command": spaces.Box(0, 1, (10,), dtype=np.float32),
    })
    action_space = spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 1024])

    buf = AsyncDiskBuffer(
        buffer_size=8,
        observation_space=obs_space,
        action_space=action_space,
        device="cpu",
        n_envs=1,
        temp_dir=str(tmp_path),
    )
    return buf


class TestSignalReady:
    def test_creates_signal_file(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        buf.signal_ready({"epoch": 1})

        assert os.path.exists(buf.signal_file)

    def test_signal_file_contains_metadata_keys(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        buf.signal_ready({"epoch": 42, "level": "attack_0"})

        with open(buf.signal_file) as f:
            data = json.load(f)

        assert data["epoch"] == 42
        assert data["level"] == "attack_0"
        assert "buffer_size" in data
        assert "n_envs" in data
        assert "observation_space" in data

    def test_signal_file_records_buffer_size(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        buf.signal_ready({})

        with open(buf.signal_file) as f:
            data = json.load(f)
        assert data["buffer_size"] == 8
        assert data["n_envs"] == 1

    def test_creates_payload_npz(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        buf.signal_ready({})

        payload_path = os.path.join(buf.instance_dir, "payload.npz")
        assert os.path.exists(payload_path)


class TestCheckReady:
    def test_returns_metadata_when_signal_exists(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        buf.signal_ready({"epoch": 5})

        result = buf.check_ready()
        assert result is not None
        assert result["epoch"] == 5

    def test_returns_none_when_no_signal(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        result = buf.check_ready()
        assert result is None


class TestClearSignal:
    def test_removes_signal_file(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        buf.signal_ready({"epoch": 1})
        assert os.path.exists(buf.signal_file)

        buf.clear_signal()
        assert not os.path.exists(buf.signal_file)

    def test_clear_when_no_signal_is_noop(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        buf.clear_signal()


class TestWaitForConsumer:
    def test_returns_true_after_clear_signal(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        result = buf.wait_for_consumer(timeout=1)
        assert result is True

    def test_returns_true_after_signal_then_clear(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        buf.signal_ready({"epoch": 1})
        buf.clear_signal()

        result = buf.wait_for_consumer(timeout=1)
        assert result is True

    def test_returns_false_on_timeout(self, tmp_path):
        buf = _create_async_buffer(tmp_path)
        buf.signal_ready({"epoch": 1})

        result = buf.wait_for_consumer(timeout=0.2)
        assert result is False
