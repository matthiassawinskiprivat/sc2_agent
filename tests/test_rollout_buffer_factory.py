"""Rollout buffer backend selection (RAM vs disk memmap)."""
import numpy as np
import pytest
from gymnasium import spaces

from src.agent.utils.disk_buffer import DiskDictRolloutBuffer
from src.agent.utils.async_disk_buffer import AsyncDiskBuffer
from src.agent.utils.async_shared_ram_buffer import AsyncSharedRamBuffer
from src.utils.rollout_buffer_factory import (
    async_rollout_buffer_class,
    resolve_async_rollout_buffer_backend,
    resolve_rollout_buffer_backend,
    sync_rollout_buffer_config,
)


def test_resolve_rollout_buffer_backend_defaults_to_disk():
    assert resolve_rollout_buffer_backend({}) == "disk"
    assert resolve_rollout_buffer_backend({"rollout_buffer": "disk"}) == "disk"


def test_ctf_env_defaults_to_ram_without_explicit_rollout_buffer():
    ctf_env = {"env_class": "CaptureTheFlagEnv"}
    assert resolve_rollout_buffer_backend({}, ctf_env) == "ram"
    assert async_rollout_buffer_class({}, ctf_env) is AsyncSharedRamBuffer
    assert (
        resolve_async_rollout_buffer_backend({}, ctf_env) == "shared_memory"
    )


def test_ctf_env_explicit_disk_override():
    ctf_env = {"env_class": "CaptureTheFlagEnv"}
    assert (
        resolve_rollout_buffer_backend({"rollout_buffer": "disk"}, ctf_env)
        == "disk"
    )


def test_resolve_rollout_buffer_backend_ram_aliases():
    assert resolve_rollout_buffer_backend({"rollout_buffer": "ram"}) == "ram"
    assert resolve_rollout_buffer_backend({"rollout_buffer_backend": "memory"}) == "ram"


def test_sync_rollout_buffer_config_disk():
    cls, kwargs = sync_rollout_buffer_config({})
    assert cls is DiskDictRolloutBuffer
    assert kwargs["storage_backend"] == "disk"
    assert "temp_dir" in kwargs


def test_sync_rollout_buffer_config_build_namespace():
    cls, kwargs = sync_rollout_buffer_config({}, namespace="build")
    assert cls is DiskDictRolloutBuffer
    assert kwargs["storage_backend"] == "disk"
    assert kwargs["temp_dir"].endswith("sc2_buffer_cache_build")


def test_disk_dict_rollout_buffer_is_maskable():
    from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

    assert issubclass(DiskDictRolloutBuffer, MaskableDictRolloutBuffer)


def test_sync_rollout_buffer_config_ram():
    cls, kwargs = sync_rollout_buffer_config({"rollout_buffer": "ram"})
    assert cls is DiskDictRolloutBuffer
    assert kwargs == {"storage_backend": "ram"}


def test_ram_buffer_keeps_observations_in_memory(tmp_path):
    obs_space = spaces.Dict(
        {
            "map": spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.float32),
            "tactical": spaces.Box(low=0, high=1, shape=(2, 4, 4), dtype=np.float32),
            "command": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
        }
    )
    act_space = spaces.MultiDiscrete([4, 16])
    buf = DiskDictRolloutBuffer(
        8,
        obs_space,
        act_space,
        n_envs=2,
        storage_backend="ram",
        temp_dir=str(tmp_path),
    )
    assert buf.disk_files == {}
    assert not isinstance(buf.observations["map"], np.memmap)
    assert buf.observations["map"].shape == (8, 2, 4, 8, 8)


def test_disk_buffer_uses_memmap(tmp_path):
    obs_space = spaces.Dict(
        {
            "map": spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.float32),
            "tactical": spaces.Box(low=0, high=1, shape=(2, 4, 4), dtype=np.float32),
            "command": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
        }
    )
    act_space = spaces.MultiDiscrete([4, 16])
    buf = DiskDictRolloutBuffer(
        8,
        obs_space,
        act_space,
        n_envs=2,
        storage_backend="disk",
        temp_dir=str(tmp_path),
    )
    assert "map" in buf.disk_files
    assert isinstance(buf.observations["map"], np.memmap)


def test_async_ram_uses_shared_memory_buffer_class():
    assert async_rollout_buffer_class({"rollout_buffer": "ram"}) is AsyncSharedRamBuffer
    assert (
        resolve_async_rollout_buffer_backend({"rollout_buffer": "ram"})
        == "shared_memory"
    )


def test_async_disk_backend_unchanged():
    assert async_rollout_buffer_class({}) is AsyncDiskBuffer
    assert resolve_async_rollout_buffer_backend({}) == "disk"
    assert resolve_async_rollout_buffer_backend({"rollout_buffer": "disk"}) == "disk"
