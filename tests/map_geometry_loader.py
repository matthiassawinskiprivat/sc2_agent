"""Load SC2Map playable geometry for coordinate integration tests."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

try:
    import mpyq  # type: ignore

    _MPYQ_AVAILABLE = True
except ImportError:
    _MPYQ_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MapGeometry:
    """Playable geometry extracted from an on-disk SC2Map archive."""

    path: Path
    map_name: str
    map_width: int
    map_height: int
    playable_x: float
    playable_y: float
    playable_width: float
    playable_height: float

    def playable_integer_bounds(self) -> Tuple[range, range]:
        """Inclusive integer world coordinate ranges inside playable_area."""
        x0 = int(self.playable_x)
        y0 = int(self.playable_y)
        x1 = int(self.playable_x + self.playable_width) - 1
        y1 = int(self.playable_y + self.playable_height) - 1
        return range(x0, x1 + 1), range(y0, y1 + 1)

    def playable_integer_count(self) -> int:
        xs, ys = self.playable_integer_bounds()
        return len(xs) * len(ys)

    def iter_playable_integers(self):
        xs, ys = self.playable_integer_bounds()
        for x in xs:
            for y in ys:
                yield x, y


# Small maps: exhaustive integer scan. Large ladder maps: cell centers + edges.
_ROUNDTRIP_EXHAUSTIVE_MAX = 16_384


def roundtrip_world_sample_points(
    geometry: MapGeometry,
    game_info,
    path_data=None,
    *,
    exhaustive_max: int = _ROUNDTRIP_EXHAUSTIVE_MAX,
) -> List[Tuple[float, float]]:
    """
    World (x, y) samples for global spatial encode/decode roundtrip checks.

    Maps with at most ``exhaustive_max`` playable integers use every integer
    coordinate; larger maps sample spatial-cell centers plus corner/edge anchors.
    """
    if geometry.playable_integer_count() <= exhaustive_max:
        return [(float(x), float(y)) for x, y in geometry.iter_playable_integers()]

    from src.utils.obs_geometry import spatial_cell_world_center

    x0 = int(geometry.playable_x)
    y0 = int(geometry.playable_y)
    x1 = int(geometry.playable_x + geometry.playable_width) - 1
    y1 = int(geometry.playable_y + geometry.playable_height) - 1
    mx = (x0 + x1) // 2
    my = (y0 + y1) // 2
    anchors = {
        (x0, y0),
        (x1, y0),
        (x0, y1),
        (x1, y1),
        (mx, y0),
        (mx, y1),
        (x0, my),
        (x1, my),
    }
    points: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in anchors]
    for gy in range(32):
        for gx in range(32):
            wx, wy = spatial_cell_world_center(
                gx, gy, game_info, 128, 128, 32, path_data=path_data
            )
            points.append((wx, wy))
    return points


def iter_fovea_playable_integers(
    geometry: MapGeometry,
    focal_x: float,
    focal_y: float,
):
    """
    Integer playable coords inside the 32×32 tactical fovea (≤1024 points).

    Avoids scanning full playable_area on ladder-sized maps.
    """
    import math

    from src.utils.obs_geometry import tactical_crop_origin

    crop_x, crop_y = tactical_crop_origin(focal_x, focal_y)
    fx0 = int(math.floor(crop_x))
    fy0 = int(math.floor(crop_y))
    fx1 = fx0 + 31
    fy1 = fy0 + 31

    px0 = int(geometry.playable_x)
    py0 = int(geometry.playable_y)
    px1 = int(geometry.playable_x + geometry.playable_width) - 1
    py1 = int(geometry.playable_y + geometry.playable_height) - 1

    x_lo = max(px0, fx0)
    x_hi = min(px1, fx1)
    y_lo = max(py0, fy0)
    y_hi = min(py1, fy1)
    if x_lo > x_hi or y_lo > y_hi:
        return

    for x in range(x_lo, x_hi + 1):
        for y in range(y_lo, y_hi + 1):
            yield x, y


def _terrain_dims(map_archive: "mpyq.MPQArchive") -> Tuple[int, int]:
    xml = map_archive.read_file(b"t3Terrain.xml").decode("utf-8", "replace")
    match = re.search(r'dim="(\d+)\s+(\d+)\s*"', xml)
    if not match:
        raise ValueError("t3Terrain.xml missing dim attribute")
    return int(match.group(1)), int(match.group(2))


def _playable_origin(
    mapinfo: bytes,
    playable_w: int,
    playable_h: int,
    map_w: int,
    map_h: int,
) -> Tuple[float, float]:
    """Best-effort playable_area origin from MapInfo (matches SC2 when inset)."""
    ox = struct.unpack_from("<H", mapinfo, 0x38)[0]
    oy = struct.unpack_from("<H", mapinfo, 0x3C)[0]
    if ox <= 64 and oy <= 64:
        return float(ox), float(oy)
    if playable_w == map_w and playable_h == map_h:
        return 0.0, 0.0
    return (max(map_w - playable_w, 0) / 2.0, max(map_h - playable_h, 0) / 2.0)


def load_sc2map_geometry(map_path: Path) -> MapGeometry:
    """Parse playable bounds from MapInfo + terrain metadata in a .SC2Map file."""
    if not _MPYQ_AVAILABLE:
        raise RuntimeError("mpyq is required to read SC2 map archives")

    path = Path(map_path)
    archive = mpyq.MPQArchive(BytesIO(path.read_bytes()), listfile=False)
    mapinfo = archive.read_file(b"MapInfo")
    if mapinfo is None or len(mapinfo) < 0x18:
        raise ValueError(f"{path.name}: missing or invalid MapInfo")

    playable_w = int(struct.unpack_from("<H", mapinfo, 0x10)[0])
    playable_h = int(struct.unpack_from("<H", mapinfo, 0x14)[0])
    if playable_w <= 0 or playable_h <= 0:
        raise ValueError(f"{path.name}: invalid playable size {playable_w}x{playable_h}")

    terrain_w, terrain_h = _terrain_dims(archive)
    map_w = max(terrain_w - 1, playable_w)
    map_h = max(terrain_h - 1, playable_h)
    playable_x, playable_y = _playable_origin(
        mapinfo, playable_w, playable_h, map_w, map_h
    )

    return MapGeometry(
        path=path.resolve(),
        map_name=path.stem,
        map_width=map_w,
        map_height=map_h,
        playable_x=playable_x,
        playable_y=playable_y,
        playable_width=float(playable_w),
        playable_height=float(playable_h),
    )


def build_mock_game_info(geometry: MapGeometry):
    """MagicMock game_info matching MapGeometry for agent / transformer tests."""
    from unittest.mock import MagicMock

    path = np.ones((geometry.map_height, geometry.map_width), dtype=np.uint8)
    gi = MagicMock()
    gi.map_size.width = geometry.map_width
    gi.map_size.height = geometry.map_height
    gi.playable_area = MagicMock(
        x=geometry.playable_x,
        y=geometry.playable_y,
        width=geometry.playable_width,
        height=geometry.playable_height,
    )
    gi.pathing_grid = MagicMock(data_numpy=path)
    gi.map_name = geometry.map_name
    return gi, path


def sync_transformer_global_geometry(transformer, game_info, path_data=None) -> None:
    """Mirror ObservationTransformer global crop/scale without running transform()."""
    from src.utils.obs_geometry import compute_global_obs_geometry

    cx, cy, mw, mh, sx, sy, ox, oy = compute_global_obs_geometry(
        game_info,
        transformer.feature_map_width,
        transformer.feature_map_height,
        path_data,
    )
    transformer.crop_x = cx
    transformer.crop_y = cy
    transformer.scale_x = sx
    transformer.scale_y = sy
    transformer.scale = min(sx, sy)
    transformer.offset_x = ox
    transformer.offset_y = oy
    transformer._obs_span_w = float(mw)
    transformer._obs_span_h = float(mh)


def collect_pipeline_map_geometries(
    config_dir: str = "config_universal",
) -> List[Tuple[str, MapGeometry]]:
    """
    Unique maps referenced by levels in ``pipeline.yaml``.

    Returns (level_name, geometry) pairs; the same map file may appear once
    (first pipeline level that references its directory).
    """
    from src.ui.level_preview import (
        build_preview_training_conf,
        discover_level_maps,
        load_pipeline_levels,
    )

    seen_paths: set[Path] = set()
    results: List[Tuple[str, MapGeometry]] = []

    for level_name in load_pipeline_levels(config_dir):
        training_conf = build_preview_training_conf(level_name, config_dir=config_dir)
        for map_stem in discover_level_maps(level_name, training_conf):
            maps_dir = training_conf.get("maps_dir") or training_conf.get("env", {}).get(
                "maps_dir"
            )
            if maps_dir:
                base = Path(maps_dir)
                if not base.is_absolute():
                    base = PROJECT_ROOT / maps_dir
            else:
                base = PROJECT_ROOT / "train" / "maps" / level_name

            map_path = (base / f"{map_stem}.SC2Map").resolve()
            if not map_path.exists() or map_path in seen_paths:
                continue
            seen_paths.add(map_path)
            results.append((level_name, load_sc2map_geometry(map_path)))

    return results


def iter_pipeline_map_geometries(
    config_dir: str = "config_universal",
) -> Iterator[Tuple[str, MapGeometry]]:
    yield from collect_pipeline_map_geometries(config_dir)
