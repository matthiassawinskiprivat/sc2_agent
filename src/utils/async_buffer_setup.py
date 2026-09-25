"""Wire static async double-buffers (disk memmap vs shared RAM)."""
from __future__ import annotations

import os
from typing import Any, Type

import numpy as np
from gymnasium import spaces

from src.agent.utils.disk_buffer import is_disk_offloaded_obs_key
from src.agent.utils.async_disk_buffer import AsyncDiskBuffer
from src.agent.utils.async_shared_ram_buffer import AsyncSharedRamBuffer


def configure_collector_static_buffer(
    buf: AsyncDiskBuffer,
    observation_space: spaces.Dict,
    static_path: str,
) -> None:
    """
    Point buffer at ``buffer_{i}`` and set up observation storage for collector writes.
    """
    if os.path.exists(buf.instance_dir) and buf.instance_dir != static_path:
        try:
            os.rmdir(buf.instance_dir)
        except OSError:
            pass

    os.makedirs(static_path, exist_ok=True)
    buf.instance_dir = static_path

    if isinstance(buf, AsyncSharedRamBuffer):
        buf.disk_files = {}
        for stale in ("obs_map.bin", "obs_tactical.bin"):
            stale_path = os.path.join(static_path, stale)
            if os.path.exists(stale_path):
                try:
                    os.remove(stale_path)
                except OSError:
                    pass
        buf.bind_shared_observation_buffers()
        return

    buf.disk_files = {}
    for key, subspace in observation_space.spaces.items():
        if is_disk_offloaded_obs_key(key):
            filename = os.path.join(buf.instance_dir, f"obs_{key}.bin")
            shape = (buf.buffer_size, buf.n_envs, *subspace.shape)
            if key in buf.observations and isinstance(buf.observations[key], np.memmap):
                buf.observations[key]._mmap.close()
            buf.observations[key] = np.memmap(
                filename, dtype=np.float32, mode="w+", shape=shape
            )
            buf.disk_files[key] = filename
    buf._owns_disk_files = True


def prepare_learner_placeholder_buffer(buf: AsyncDiskBuffer) -> None:
    """Drop collector-only storage; learner binds on each consume."""
    if isinstance(buf, AsyncSharedRamBuffer):
        buf.disk_files = {}
        buf.close_shared_memory(unlink=False)
        for key in list(buf.observations.keys()):
            if key in buf.shared_observation_keys():
                del buf.observations[key]
        return

    buf.disk_files = {}
    buf._owns_disk_files = False
    for key in list(buf.observations.keys()):
        if isinstance(buf.observations[key], np.memmap):
            try:
                buf.observations[key]._mmap.close()
            except Exception:
                pass
            del buf.observations[key]
