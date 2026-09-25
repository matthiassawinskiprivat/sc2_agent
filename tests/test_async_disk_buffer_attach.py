"""Learner must reopen disk observation memmaps every async buffer consume."""
import numpy as np
import pytest
from gymnasium import spaces

from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from src.agent.utils.async_disk_buffer import AsyncDiskBuffer


def _buffer(tmp_path):
    obs_space = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (2, 8, 8), dtype=np.float32),
            "tactical": spaces.Box(0, 1, (1, 4, 4), dtype=np.float32),
            "command": spaces.Box(0, 1, (10,), dtype=np.float32),
        }
    )
    return AsyncDiskBuffer(
        4,
        obs_space,
        spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 16]),
        n_envs=1,
        temp_dir=str(tmp_path),
    )


def test_signal_ready_sets_ipc_backend_disk(tmp_path):
    buf = _buffer(tmp_path)
    buf.signal_ready({})
    meta = buf.check_ready()
    assert meta["ipc_backend"] == "disk"


def test_attach_disk_observations_reopens_after_collector_rewrite(tmp_path):
    instance_dir = tmp_path / "buffer_0"
    instance_dir.mkdir()
    meta = {"buffer_size": 4, "n_envs": 1}

    collector = _buffer(tmp_path)
    collector.instance_dir = str(instance_dir)
    collector.disk_files = {}
    for key in ("map", "tactical"):
        subspace = collector.observation_space.spaces[key]
        shape = (4, 1, *subspace.shape)
        path = instance_dir / f"obs_{key}.bin"
        collector.observations[key] = np.memmap(
            path, dtype=np.float32, mode="w+", shape=shape
        )
        collector.disk_files[key] = str(path)
    collector.observations["map"][0, 0, 0, 0, 0] = 1.0
    collector.observations["map"].flush()

    learner = _buffer(tmp_path / "learner")
    AsyncDiskBuffer.attach_disk_observations(learner, str(instance_dir), meta)
    assert learner.observations["map"][0, 0, 0, 0, 0] == pytest.approx(1.0)
    assert learner._owns_disk_files is False

    collector.observations["map"][0, 0, 0, 0, 0] = 99.0
    collector.observations["map"].flush()
    first_handle = learner.observations["map"]

    AsyncDiskBuffer.attach_disk_observations(learner, str(instance_dir), meta)
    assert learner.observations["map"] is not first_handle
    assert learner.observations["map"][0, 0, 0, 0, 0] == pytest.approx(99.0)


def test_disk_obs_fingerprint_in_signal_and_attach_mismatch(tmp_path):
    """Disk IPC must fingerprint map/tactical like shared-RAM sample_mean checks."""
    from src.agent.utils.disk_buffer import obs_sample_mean

    collector = _buffer(tmp_path)
    instance_dir = tmp_path / "buffer_fp"
    instance_dir.mkdir()
    collector.instance_dir = str(instance_dir)
    collector.disk_files = {}
    collector._owns_disk_files = True
    for key in ("map", "tactical"):
        subspace = collector.observation_space.spaces[key]
        shape = (4, 1, *subspace.shape)
        path = instance_dir / f"obs_{key}.bin"
        collector.observations[key] = np.memmap(
            path, dtype=np.float32, mode="w+", shape=shape
        )
        collector.disk_files[key] = str(path)
    collector.observations["map"].fill(0.25)
    collector.observations["map"].flush()
    collector.observations["tactical"].fill(0.5)
    collector.observations["tactical"].flush()

    collector.signal_ready({})
    meta = collector.check_ready()
    assert meta["ipc_backend"] == "disk"
    assert "disk_obs" in meta
    assert meta["disk_obs"]["map"]["sample_mean"] == pytest.approx(
        obs_sample_mean(collector.observations["map"])
    )

    learner = _buffer(tmp_path / "learner_fp")
    AsyncDiskBuffer.attach_disk_observations(learner, str(instance_dir), meta)
    assert learner.observations["map"][0, 0, 0, 0, 0] == pytest.approx(0.25)

    bad = dict(meta)
    bad["disk_obs"] = {
        "map": {"sample_mean": 99.0, "shape": list(collector.observations["map"].shape)}
    }
    with pytest.raises(RuntimeError, match="attach mismatch"):
        AsyncDiskBuffer.attach_disk_observations(learner, str(instance_dir), bad)


def test_learner_del_does_not_delete_collector_disk_files(tmp_path):
    instance_dir = tmp_path / "buffer_0"
    instance_dir.mkdir()
    collector = _buffer(tmp_path)
    collector.instance_dir = str(instance_dir)
    path = instance_dir / "obs_map.bin"
    shape = (4, 1, 2, 8, 8)
    collector.observations["map"] = np.memmap(
        path, dtype=np.float32, mode="w+", shape=shape
    )
    collector.disk_files = {"map": str(path)}
    collector._owns_disk_files = True
    collector.observations["map"][0, 0, 0, 0, 0] = 3.0
    collector.observations["map"].flush()

    learner = _buffer(tmp_path / "learner_del")
    AsyncDiskBuffer.attach_disk_observations(
        learner, str(instance_dir), {"buffer_size": 4, "n_envs": 1}
    )
    assert path.exists()
    # Simulate GC of the learner buffer without wiping collector binaries.
    learner.__del__()
    assert path.exists()
    assert np.memmap(path, dtype=np.float32, mode="r", shape=shape)[0, 0, 0, 0, 0] == pytest.approx(3.0)
