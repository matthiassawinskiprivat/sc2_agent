"""
Resolve sync/async rollout buffer backend from training YAML.

``rollout_buffer`` / ``rollout_buffer_backend``:
  - ``disk`` (default): memmap map/tactical tensors — low RAM, SSD I/O.
  - ``ram``: observations in RAM. Sync: in-process arrays. Async: shared_memory IPC.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Type

from loguru import logger

from src.agent.utils.async_disk_buffer import AsyncDiskBuffer
from src.agent.utils.async_shared_ram_buffer import AsyncSharedRamBuffer
from src.agent.utils.disk_buffer import DiskDictRolloutBuffer


_CTF_ENV_CLASS_MARKERS = (
    "CaptureTheFlagEnv",
    "CaptureTheFlagSingleEnv",
    "CaptureTheFlagSelfPlayEnv",
)


def _is_ctf_env(env_params: Dict[str, Any] | None) -> bool:
    if not env_params:
        return False
    env_class = str(env_params.get("env_class", ""))
    return any(marker in env_class for marker in _CTF_ENV_CLASS_MARKERS)


def resolve_rollout_buffer_backend(
    training_params: Dict[str, Any],
    env_params: Dict[str, Any] | None = None,
) -> str:
    explicit = (
        training_params.get("rollout_buffer_backend")
        or training_params.get("rollout_buffer")
    )
    if explicit is not None:
        raw = explicit
    elif _is_ctf_env(env_params):
        raw = "ram"
    else:
        raw = "disk"
    normalized = str(raw).strip().lower()
    if normalized in ("ram", "memory", "in_memory"):
        return "ram"
    if normalized in ("disk", "memmap", "mmap"):
        return "disk"
    raise ValueError(
        f"Invalid rollout_buffer {raw!r}. Use 'disk' (default) or 'ram'."
    )


def sync_rollout_buffer_config(
    training_params: Dict[str, Any],
    env_params: Dict[str, Any] | None = None,
) -> Tuple[Type[DiskDictRolloutBuffer], Dict[str, Any]]:
    """Buffer class + kwargs for MaskableRecurrentPPO (sync universal training)."""
    backend = resolve_rollout_buffer_backend(training_params, env_params)
    if backend == "ram":
        logger.info("Rollout buffer: RAM (in-memory observations, no memmap).")
        return DiskDictRolloutBuffer, {"storage_backend": "ram"}

    temp_dir = training_params.get(
        "rollout_buffer_temp_dir",
        os.environ.get("SC2_SYNC_BUFFER_DIR", "C:/temp/sc2_buffer_cache"),
    )
    logger.info(f"Rollout buffer: disk (memmap) at {temp_dir}")
    return DiskDictRolloutBuffer, {
        "storage_backend": "disk",
        "temp_dir": os.path.normpath(str(temp_dir)),
    }


def async_rollout_buffer_class(
    training_params: Dict[str, Any],
    env_params: Dict[str, Any] | None = None,
) -> Type[AsyncDiskBuffer]:
    """Async collector/learner buffer: disk memmap or cross-process shared RAM."""
    backend = resolve_rollout_buffer_backend(training_params, env_params)
    if backend == "ram":
        logger.info(
            "Async rollout buffer: shared RAM (multiprocessing.shared_memory) — "
            "no observation memmap on SSD."
        )
        return AsyncSharedRamBuffer
    logger.info("Async rollout buffer: disk memmap for observation IPC.")
    return AsyncDiskBuffer


def resolve_async_rollout_buffer_backend(
    training_params: Dict[str, Any],
    env_params: Dict[str, Any] | None = None,
) -> str:
    """Backward-compatible name: returns ``disk`` or ``shared_memory``."""
    backend = resolve_rollout_buffer_backend(training_params, env_params)
    return "shared_memory" if backend == "ram" else "disk"
