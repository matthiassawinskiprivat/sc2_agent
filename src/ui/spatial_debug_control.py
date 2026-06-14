"""Debug UI helpers: spatial grid indices → policy-equivalent world targets."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.agent.universal_constants import build_action_head_manifest

SPATIAL_GRID_SIZE = 32


def spatial_idx_from_grid(gx: int, gy: int, grid_size: int = SPATIAL_GRID_SIZE) -> int:
    """Flat spatial index matching ``UniversalUnitControlAgent`` (row-major gy, gx)."""
    gx_i = max(0, min(int(gx), int(grid_size) - 1))
    gy_i = max(0, min(int(gy), int(grid_size) - 1))
    return gy_i * int(grid_size) + gx_i


def grid_from_spatial_idx(spatial_idx: int, grid_size: int = SPATIAL_GRID_SIZE) -> Tuple[int, int]:
    """Inverse of :func:`spatial_idx_from_grid` → ``(gx, gy)``."""
    gy, gx = divmod(int(spatial_idx), int(grid_size))
    return int(gx), int(gy)


def command_head_index(training_conf: Dict[str, Any], command_name: str) -> int:
    """Manifest index for a configured spatial command head (e.g. ``move_global``)."""
    heads = list(training_conf.get("action_heads") or [])
    manifest = training_conf.get("action_head_manifest") or build_action_head_manifest(
        training_conf, heads
    )
    if command_name in manifest:
        return int(manifest.index(command_name))
    if command_name in heads:
        return int(heads.index(command_name))
    raise ValueError(f"Command head {command_name!r} not in action_heads/manifest")


def build_universal_spatial_action(
    command_name: str,
    gx: int,
    gy: int,
    training_conf: Dict[str, Any],
    grid_size: int = SPATIAL_GRID_SIZE,
) -> np.ndarray:
    """``MultiDiscrete`` action vector ``[command_idx, spatial_idx]`` for debug injection."""
    return np.array(
        [command_head_index(training_conf, command_name), spatial_idx_from_grid(gx, gy, grid_size)],
        dtype=np.int64,
    )


def parse_grid_field(text: str, grid_size: int = SPATIAL_GRID_SIZE) -> int:
    """Parse a UI text field into a clamped grid coordinate."""
    value = int(str(text).strip())
    return max(0, min(value, int(grid_size) - 1))
