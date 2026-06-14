"""Shared game-world ↔ observation pixel geometry (global 128×128 view)."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

TACTICAL_OBS_SIZE = 32
TACTICAL_HALF = TACTICAL_OBS_SIZE // 2

_SPARSE_FRINGE_FRAC = 0.05
_SPARSE_FRINGE_MIN_CELLS = 8


def tactical_crop_origin(focal_x: float, focal_y: float) -> Tuple[float, float]:
    """Top-left game coordinate of the 32×32 tactical fovea (matches ObservationTransformer)."""
    return math.floor(float(focal_x) - float(TACTICAL_HALF)), math.floor(
        float(focal_y) - float(TACTICAL_HALF)
    )


def world_to_tactical_obs_pixel(
    world_x: float,
    world_y: float,
    focal_x: float,
    focal_y: float,
) -> Tuple[int, int]:
    """Map a game-world point to tactical observation pixel indices (32×32)."""
    crop_x, crop_y = tactical_crop_origin(focal_x, focal_y)
    content_x = float(world_x) - crop_x
    content_y = float(world_y) - crop_y
    px = int(math.floor(content_x))
    # Match terrain row assignment: pathing samples floor(world_y), not floor(crop + 31 - py).
    py = int(TACTICAL_OBS_SIZE - 1 - math.floor(content_y))
    px = max(0, min(px, TACTICAL_OBS_SIZE - 1))
    py = max(0, min(py, TACTICAL_OBS_SIZE - 1))
    return px, py


def obs_pixel_center_to_display(
    px: int,
    py: int,
    display_size: Tuple[int, int],
    obs_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Map an obs cell index to the center of that cell on the upscaled UI frame."""
    display_w, display_h = display_size
    obs_w, obs_h = obs_size
    cx = int((px + 0.5) * display_w / float(obs_w))
    cy = int((py + 0.5) * display_h / float(obs_h))
    return cx, cy


def _as_path_array(path_data: Any) -> Optional[np.ndarray]:
    if path_data is None:
        return None
    if hasattr(path_data, "data_numpy"):
        return np.asarray(path_data.data_numpy)
    try:
        return np.asarray(path_data)
    except (TypeError, ValueError):
        return None


def _trim_sparse_fringe(
    arr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> Tuple[int, int, int, int]:
    """Drop walkable-bbox edges that only contain a handful of ramp pixels."""
    min_col_fill = max(_SPARSE_FRINGE_MIN_CELLS, int((x1 - x0 + 1) * _SPARSE_FRINGE_FRAC))
    min_row_fill = max(_SPARSE_FRINGE_MIN_CELLS, int((y1 - y0 + 1) * _SPARSE_FRINGE_FRAC))

    while y0 <= y1 and int(np.count_nonzero(arr[y0, x0 : x1 + 1])) < min_row_fill:
        y0 += 1
    while y1 >= y0 and int(np.count_nonzero(arr[y1, x0 : x1 + 1])) < min_row_fill:
        y1 -= 1
    while x0 <= x1 and int(np.count_nonzero(arr[y0 : y1 + 1, x0])) < min_col_fill:
        x0 += 1
    while x1 >= x0 and int(np.count_nonzero(arr[y0 : y1 + 1, x1])) < min_col_fill:
        x1 -= 1

    return x0, y0, x1, y1


def walkable_area_bounds(
    path_data: Any,
    game_info: Any = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Tight axis-aligned bbox of walkable pathing cells (crop_x, crop_y, width, height)."""
    arr = _as_path_array(path_data)
    if arr is None or arr.size == 0:
        return None

    walkable = arr > 0
    if not np.any(walkable):
        return None

    ys, xs = np.where(walkable)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    if game_info is not None and hasattr(game_info, "map_size"):
        ms_w = int(game_info.map_size.width)
        ms_h = int(game_info.map_size.height)
        x0 = max(0, min(x0, ms_w - 1))
        x1 = max(0, min(x1, ms_w - 1))
        y0 = max(0, min(y0, ms_h - 1))
        y1 = max(0, min(y1, ms_h - 1))

    x0, y0, x1, y1 = _trim_sparse_fringe(arr, x0, y0, x1, y1)
    if x1 < x0 or y1 < y0:
        return None

    return float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)


def observation_bounds(
    game_info: Any,
    path_data: Any = None,
) -> Tuple[float, float, float, float]:
    """
    Return (crop_x, crop_y, span_w, span_h) for observation ↔ world mapping.

    Maps SC2 **world coordinates** (game units on the map) into the fixed observation
    canvases (128×128 global, 32×32 tactical fovea). Those tensor sizes never change.

    Prefer SC2 ``playable_area`` so map edges and bases are not clipped. Fall back to
    trimmed walkable bbox, then full ``map_size``.
    """
    if game_info is None:
        return 0.0, 0.0, 200.0, 200.0

    area = getattr(game_info, "playable_area", None)
    if area is not None:
        return (
            float(area.x),
            float(area.y),
            float(area.width),
            float(area.height),
        )

    walkable = walkable_area_bounds(path_data, game_info)
    if walkable is not None:
        return walkable

    if hasattr(game_info, "map_size"):
        return (
            0.0,
            0.0,
            float(game_info.map_size.width),
            float(game_info.map_size.height),
        )

    return 0.0, 0.0, 200.0, 200.0


def playable_bounds(game_info: Any, path_data: Any = None) -> Tuple[float, float, float, float]:
    """Alias for :func:`observation_bounds` (kept for older call sites)."""
    return observation_bounds(game_info, path_data)


def global_spatial_step(obs_size: int, grid_size: int) -> int:
    """Obs pixels per spatial-policy cell along one axis (e.g. 128 // 32 → 4)."""
    return max(int(obs_size) // max(int(grid_size), 1), 1)


def spatial_cell_obs_bounds(
    gx: int,
    gy: int,
    obs_width: int,
    obs_height: int,
    grid_size: int,
) -> Tuple[int, int, int, int]:
    """Inclusive obs-pixel bounds for one spatial-policy cell (north-west corner indexing)."""
    step = global_spatial_step(obs_width, grid_size)
    px_lo = int(gx) * step
    py_lo = int(gy) * step
    px_hi = min(px_lo + step - 1, int(obs_width) - 1)
    py_hi = min(py_lo + step - 1, int(obs_height) - 1)
    return px_lo, px_hi, py_lo, py_hi


def spatial_idx_to_obs_pixel(
    gx: int,
    gy: int,
    obs_width: int,
    obs_height: int,
    grid_size: int,
) -> Tuple[int, int]:
    """
    Map a spatial-policy cell to the obs pixel at its north-west corner.

    Prefer :func:`spatial_cell_world_center` for actuation; this helper is kept
    for overlays and tests that need a representative pixel index.
    """
    px_lo, _px_hi, py_lo, _py_hi = spatial_cell_obs_bounds(
        gx, gy, obs_width, obs_height, grid_size
    )
    return px_lo, py_lo


def obs_pixel_to_global_world(
    px: int,
    py: int,
    game_info: Any,
    obs_width: int,
    obs_height: int,
    path_data: Any = None,
) -> Tuple[float, float]:
    """Game-world center of an obs pixel (inverse of global ``round`` sampling)."""
    crop_x, crop_y, mw, mh, scale_x, scale_y, offset_x, offset_y = (
        compute_global_obs_geometry(game_info, obs_width, obs_height, path_data)
    )
    wx = (float(px) + 0.5 - offset_x) / scale_x + crop_x
    content_y = (float(obs_height) - 1.0 - float(py)) + 0.5 - offset_y
    wy = content_y / scale_y + crop_y

    min_x, min_y, span_w, span_h = observation_bounds(game_info, path_data)
    eps = 1e-3
    max_x = min_x + span_w - eps
    max_y = min_y + span_h - eps
    wx = max(min_x, min(float(wx), max_x))
    wy = max(min_y, min(float(wy), max_y))
    return wx, wy


def spatial_cell_actuation_pixel(
    gx: int,
    gy: int,
    obs_width: int,
    obs_height: int,
    grid_size: int,
) -> Tuple[int, int]:
    """
    Representative obs pixel for actuation inside one spatial cell.

    Edge cells bias toward the obs row/column that touches the map boundary so
    ladder-sized maps (512 playable in 128 obs) do not decode 8+ gu away from
    edge beacons. Interior cells use the north-west pixel (matches encode owner).
    """
    px_lo, px_hi, py_lo, py_hi = spatial_cell_obs_bounds(
        gx, gy, obs_width, obs_height, grid_size
    )
    last = int(grid_size) - 1
    if gx == 0:
        px = px_lo
    elif gx == last:
        px = px_hi
    else:
        px = px_lo

    if gy == 0:
        py = py_lo
    elif gy == last:
        py = py_hi
    else:
        py = py_lo
    return px, py


def spatial_cell_world_center(
    gx: int,
    gy: int,
    game_info: Any,
    obs_width: int,
    obs_height: int,
    grid_size: int,
    path_data: Any = None,
) -> Tuple[float, float]:
    """
    World target for a spatial-policy cell (actuation decode).

    Average the four obs-pixel corners of the cell — must match
    ``ObservationTransformer._game_to_obs_pixel`` (round sampling) on all map
    sizes including letterboxed ladder maps (512 gu → 128 obs).
    """
    px_lo, px_hi, py_lo, py_hi = spatial_cell_obs_bounds(
        gx, gy, obs_width, obs_height, grid_size
    )
    corners = (
        obs_pixel_to_global_world(px_lo, py_lo, game_info, obs_width, obs_height, path_data),
        obs_pixel_to_global_world(px_hi, py_lo, game_info, obs_width, obs_height, path_data),
        obs_pixel_to_global_world(px_lo, py_hi, game_info, obs_width, obs_height, path_data),
        obs_pixel_to_global_world(px_hi, py_hi, game_info, obs_width, obs_height, path_data),
    )
    wx = sum(c[0] for c in corners) / 4.0
    wy = sum(c[1] for c in corners) / 4.0

    min_x, min_y, span_w, span_h = observation_bounds(game_info, path_data)
    eps = 1e-3
    max_x = min_x + span_w - eps
    max_y = min_y + span_h - eps
    wx = max(min_x, min(float(wx), max_x))
    wy = max(min_y, min(float(wy), max_y))
    return wx, wy


def spatial_idx_to_global_world(
    spatial_idx: int,
    grid_size: int,
    game_info: Any,
    obs_width: int,
    obs_height: int,
    path_data: Any = None,
) -> Tuple[float, float]:
    """Decode a flat spatial index to a global game-world target."""
    gy, gx = divmod(int(spatial_idx), int(grid_size))
    return spatial_cell_world_center(
        gx, gy, game_info, obs_width, obs_height, grid_size, path_data=path_data
    )


def global_world_to_spatial_idx(
    world_x: float,
    world_y: float,
    grid_size: int,
    game_info: Any,
    obs_width: int,
    obs_height: int,
    path_data: Any = None,
) -> int:
    """Encode a world position into the spatial cell that owns its obs pixel."""
    crop_x, crop_y, _mw, _mh, scale_x, scale_y, offset_x, offset_y = (
        compute_global_obs_geometry(game_info, obs_width, obs_height, path_data)
    )
    content_x = (float(world_x) - crop_x) * scale_x + offset_x
    content_y = (float(world_y) - crop_y) * scale_y + offset_y
    px = int(round(content_x))
    py = int(round(float(obs_height) - 1.0 - content_y))
    step = global_spatial_step(obs_width, grid_size)
    gx = max(0, min(int(px) // step, int(grid_size) - 1))
    gy = max(0, min(int(py) // step, int(grid_size) - 1))
    return gy * int(grid_size) + gx


def compute_global_obs_geometry(
    game_info: Any,
    feature_map_width: int,
    feature_map_height: int,
    path_data: Any = None,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Uniform scale + letterbox offsets (matches ObservationTransformer global view).

    Returns (crop_x, crop_y, mw, mh, scale_x, scale_y, offset_x, offset_y).
    """
    crop_x, crop_y, mw, mh = observation_bounds(game_info, path_data)
    mw_i = max(int(mw), 1)
    mh_i = max(int(mh), 1)
    fw = float(feature_map_width)
    fh = float(feature_map_height)
    scale = min(fw / mw_i, fh / mh_i)
    offset_x = float(max(0, int(fw - int(mw_i * scale)) // 2))
    offset_y = float(max(0, int(fh - int(mh_i * scale)) // 2))
    return crop_x, crop_y, float(mw_i), float(mh_i), scale, scale, offset_x, offset_y
