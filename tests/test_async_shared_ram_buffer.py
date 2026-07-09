"""Async shared-memory rollout buffer IPC metadata."""
import json
from pathlib import Path

import numpy as np
from gymnasium import spaces

from src.agent.utils.async_shared_ram_buffer import (
    AsyncSharedRamBuffer,
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
