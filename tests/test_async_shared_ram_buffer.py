"""Async shared-memory rollout buffer IPC metadata."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from gymnasium import spaces

from src.agent.utils.async_shared_ram_buffer import (
    AsyncSharedRamBuffer,
    _attach_shm_segment,
    observation_buffer_nbytes,
)


def _obs_space():
    return spaces.Dict(
        {
            "map": spaces.Box(low=0, high=1, shape=(2, 4, 4), dtype=np.float32),
            "tactical": spaces.Box(low=0, high=1, shape=(1, 2, 2), dtype=np.float32),
            "command": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
        }
    )


def test_observation_buffer_nbytes_ctf_scale_no_overflow():
    # CTF-scale shapes overflow np.prod int32; math.prod must yield positive bytes.
    nbytes = observation_buffer_nbytes((2048, 4, 23, 128, 128))
    assert nbytes > 0
    assert nbytes == 2048 * 4 * 23 * 128 * 128 * 4


def test_signal_ready_marks_shared_memory_ipc(tmp_path: Path):
    act = spaces.MultiDiscrete([4, 16])
    buf = AsyncSharedRamBuffer(
        4,
        _obs_space(),
        act,
        n_envs=2,
        temp_dir=str(tmp_path),
        storage_backend="ram",
    )
    buf.instance_id = "buffer_0"
    buf.instance_dir = str(tmp_path / "buffer_0")
    tmp_path.joinpath("buffer_0").mkdir()
    buf.bind_shared_observation_buffers()
    buf.full = True
    buf.pos = 4

    meta = {"level": "test"}
    buf.signal_ready(meta)

    signal = json.loads((tmp_path / "buffer_0" / "buffer_ready.json").read_text())
    assert signal["ipc_backend"] == "shared_memory"
    assert "map" in signal["shared_obs"]
    assert signal["shared_obs"]["map"]["name"] == "sc2async_buffer_0_map"
    assert signal["shared_obs"]["map"]["nbytes"] == observation_buffer_nbytes(
        (4, 2, 2, 4, 4)
    )
    assert "sample_mean" in signal["shared_obs"]["map"]

    with np.load(tmp_path / "buffer_0" / "payload.npz") as payload:
        assert "obs_map" not in payload.files
        assert "obs_tactical" not in payload.files
        assert "obs_command" in payload.files

    buf.close_shared_memory(unlink=True)


def test_ram_buffer_get_splits_on_episode_starts(tmp_path: Path):
    """CTF RAM async buffer uses DiskDictRolloutBuffer.get (focal-unit aware sequences)."""
    from src.agent.utils.disk_buffer import build_recurrent_sequence_spans

    act = spaces.MultiDiscrete([4, 16])
    buf = AsyncSharedRamBuffer(
        16,
        _obs_space(),
        act,
        n_envs=1,
        temp_dir=str(tmp_path),
        storage_backend="ram",
    )
    buf.episode_starts = np.zeros((16, 1), dtype=np.float32)
    buf.episode_starts[5, 0] = 1.0
    buf.episode_starts[10, 0] = 1.0
    buf.policy_active = np.ones((16, 1), dtype=np.float32)
    buf.pos = 16
    buf.full = True

    spans = build_recurrent_sequence_spans(
        1, 16, buf.episode_starts, max_seq_len=32, policy_active=buf.policy_active
    )
    assert spans == [(0, 0, 5), (0, 5, 5), (0, 10, 6)]

    batches = list(buf.get(batch_size=32))
    assert batches
    total_steps = sum(int(batch.mask.sum().item()) for batch in batches)
    assert total_steps == 16

    buf.close_shared_memory(unlink=True)


def test_attach_shared_observations_known_size(tmp_path: Path):
    """Learner attach must work from shape/nbytes without CPython size-probe."""
    act = spaces.MultiDiscrete([4, 16])
    buf = AsyncSharedRamBuffer(
        4,
        _obs_space(),
        act,
        n_envs=2,
        temp_dir=str(tmp_path),
        storage_backend="ram",
    )
    buf.instance_id = "buffer_attach"
    buf.instance_dir = str(tmp_path / "buffer_attach")
    tmp_path.joinpath("buffer_attach").mkdir()
    buf.bind_shared_observation_buffers()
    buf.observations["map"][0, 0, 0, 0, 0] = 7.5

    meta = {"shared_obs": {}}
    buf.signal_ready(meta)
    signal = json.loads((tmp_path / "buffer_attach" / "buffer_ready.json").read_text())

    learner = AsyncSharedRamBuffer(
        4,
        _obs_space(),
        act,
        n_envs=2,
        temp_dir=str(tmp_path / "learner"),
        storage_backend="ram",
    )
    learner.observations = {}
    AsyncSharedRamBuffer.attach_shared_observations(learner, signal)

    assert learner.observations["map"][0, 0, 0, 0, 0] == pytest.approx(7.5)

    buf.close_shared_memory(unlink=True)
    learner.close_shared_memory(unlink=False)


@pytest.mark.skipif(os.name != "nt", reason="Windows named-mmap attach path")
def test_windows_attach_does_not_call_sharedmemory_create_false(tmp_path: Path):
    """Regression: SharedMemory(create=False) double-maps and raises WinError 1450."""
    act = spaces.MultiDiscrete([4, 16])
    buf = AsyncSharedRamBuffer(
        4,
        _obs_space(),
        act,
        n_envs=2,
        temp_dir=str(tmp_path),
        storage_backend="ram",
    )
    buf.instance_id = "buffer_win"
    buf.instance_dir = str(tmp_path / "buffer_win")
    tmp_path.joinpath("buffer_win").mkdir()
    buf.bind_shared_observation_buffers()
    buf.observations["map"][1, 0, 0, 0, 0] = 3.25

    meta = {"level": "test"}
    buf.signal_ready(meta)
    signal = json.loads((tmp_path / "buffer_win" / "buffer_ready.json").read_text())

    learner = AsyncSharedRamBuffer(
        4,
        _obs_space(),
        act,
        n_envs=2,
        temp_dir=str(tmp_path / "learner_win"),
        storage_backend="ram",
    )
    learner.observations = {}
    with patch(
        "src.agent.utils.async_shared_ram_buffer.SharedMemory",
        side_effect=AssertionError("must not SharedMemory(create=False) on Windows attach"),
    ):
        AsyncSharedRamBuffer.attach_shared_observations(learner, signal)

    assert learner.observations["map"][1, 0, 0, 0, 0] == pytest.approx(3.25)
    buf.close_shared_memory(unlink=True)
    learner.close_shared_memory(unlink=False)


@pytest.mark.skipif(os.name != "nt", reason="Windows named mapping")
def test_attach_missing_segment_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        _attach_shm_segment("sc2async_missing_segment_do_not_create", 4096)


def test_attach_sample_mean_mismatch_raises(tmp_path: Path):
    """Refuse to train if the mapped segment is not the collector's tensor."""
    act = spaces.MultiDiscrete([4, 16])
    buf = AsyncSharedRamBuffer(
        4,
        _obs_space(),
        act,
        n_envs=2,
        temp_dir=str(tmp_path),
        storage_backend="ram",
    )
    buf.instance_id = "buffer_fp"
    buf.instance_dir = str(tmp_path / "buffer_fp")
    tmp_path.joinpath("buffer_fp").mkdir()
    buf.bind_shared_observation_buffers()
    buf.observations["map"][0, 0, 0, 0, 0] = 1.0

    meta = {"level": "test"}
    buf.signal_ready(meta)
    signal = json.loads((tmp_path / "buffer_fp" / "buffer_ready.json").read_text())
    signal["shared_obs"]["map"]["sample_mean"] = 99.0

    learner = AsyncSharedRamBuffer(
        4,
        _obs_space(),
        act,
        n_envs=2,
        temp_dir=str(tmp_path / "learner_fp"),
        storage_backend="ram",
    )
    learner.observations = {}
    with pytest.raises(RuntimeError, match="attach mismatch"):
        AsyncSharedRamBuffer.attach_shared_observations(learner, signal)

    buf.close_shared_memory(unlink=True)
    learner.close_shared_memory(unlink=False)
