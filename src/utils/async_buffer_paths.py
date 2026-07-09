"""Shared async rollout IPC directory (collector, learner, CTF opponent sync)."""
from __future__ import annotations

import os

# Single source of truth for async rollout IPC directory (collector, learner, CTF).
DEFAULT_ASYNC_BUFFER_DIR = "C:/temp/sc2_async_buffer"


def resolve_async_buffer_dir() -> str:
    """Absolute/normalized path to the async buffer root."""
    return os.path.normpath(
        os.environ.get("SC2_ASYNC_BUFFER_DIR", DEFAULT_ASYNC_BUFFER_DIR)
    )


def ensure_async_buffer_dir_env() -> str:
    """Publish buffer dir to os.environ so SubprocVecEnv workers resolve the same path."""
    path = resolve_async_buffer_dir()
    os.environ["SC2_ASYNC_BUFFER_DIR"] = path
    return path
