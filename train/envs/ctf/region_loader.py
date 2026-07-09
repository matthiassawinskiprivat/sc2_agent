"""
RegionLoader — MPQ archive reader + XML parser for SC2 map regions.

Region names are stable across all training maps (global naming
convention).  The loader caches parsed results keyed by map path so
that repeated ``reset()`` calls within a training run incur only a
single file-read.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import mpyq  # type: ignore
    _MPYQ_AVAILABLE = True
except ImportError:
    _MPYQ_AVAILABLE = False


@dataclass(frozen=True)
class Point2:
    """2-D point in SC2 world coordinates."""

    x: float
    y: float

    def distance_to(self, other: "Point2") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass(frozen=True)
class CircleRegion:
    name: str
    center: Point2
    radius: float

    @property
    def shape_type(self) -> str:
        return "circle"

    def contains(self, point: Point2) -> bool:
        return self.center.distance_to(point) <= self.radius


@dataclass(frozen=True)
class RectRegion:
    name: str
    min_corner: Point2
    max_corner: Point2

    @property
    def shape_type(self) -> str:
        return "rect"

    @property
    def center(self) -> Point2:
        return Point2(
            (self.min_corner.x + self.max_corner.x) / 2.0,
            (self.min_corner.y + self.max_corner.y) / 2.0,
        )

    def contains(self, point: Point2) -> bool:
        return (
            self.min_corner.x <= point.x <= self.max_corner.x
            and self.min_corner.y <= point.y <= self.max_corner.y
        )


RegionDef = CircleRegion | RectRegion


def _parse_point2(value_str: str) -> Point2:
    parts = [p.strip() for p in value_str.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'x,y' format, got {value_str!r}")
    return Point2(float(parts[0]), float(parts[1]))


def _parse_regions_xml(xml_bytes: bytes) -> Dict[str, RegionDef]:
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))

    region_elements = root.findall("region")
    if not region_elements:
        container = root.find("regions")
        if container is not None:
            region_elements = container.findall("region")

    result: Dict[str, RegionDef] = {}
    for region_el in region_elements:
        name_el = region_el.find("name")
        if name_el is None:
            continue
        name = name_el.get("value", "").strip()
        if not name:
            continue

        shape_el = region_el.find("shape")
        if shape_el is None:
            raise ValueError(f"Region '{name}' has no <shape> element.")

        shape_type = (shape_el.get("type") or "").lower()

        if shape_type == "circle":
            center_el = shape_el.find("center")
            radius_el = shape_el.find("radius")
            if center_el is None or radius_el is None:
                raise ValueError(f"Circle region '{name}' missing <center> or <radius>.")
            center = _parse_point2(center_el.get("value", ""))
            radius = float(radius_el.get("value", "0"))
            result[name] = CircleRegion(name=name, center=center, radius=radius)

        elif shape_type == "rect":
            min_el = shape_el.find("min")
            max_el = shape_el.find("max")
            if min_el is None or max_el is None:
                raise ValueError(f"Rect region '{name}' missing <min> or <max>.")
            min_corner = _parse_point2(min_el.get("value", ""))
            max_corner = _parse_point2(max_el.get("value", ""))
            result[name] = RectRegion(name=name, min_corner=min_corner, max_corner=max_corner)

        else:
            raise ValueError(
                f"Region '{name}' has unsupported shape type {shape_type!r}. "
                "Supported: 'circle', 'rect'."
            )

    return result


class RegionLoader:
    """
    Loads and caches SC2 map region definitions.

    Thread-safety: the cache is written exactly once per map path (on
    first load) and thereafter read-only, so concurrent reads are safe.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, RegionDef]] = {}

    def load(self, map_path: str) -> Dict[str, RegionDef]:
        key = str(Path(map_path).resolve())
        if key in self._cache:
            return self._cache[key]
        regions = self._read_from_mpq(map_path)
        self._cache[key] = regions
        return regions

    def get_cached(self, map_path: str) -> Dict[str, RegionDef]:
        key = str(Path(map_path).resolve())
        if key not in self._cache:
            raise RuntimeError(
                f"Regions for '{map_path}' not loaded yet. Call load() first."
            )
        return self._cache[key]

    def load_from_xml_bytes(
        self, xml_bytes: bytes, cache_key: str = "__test__"
    ) -> Dict[str, RegionDef]:
        """
        Load regions from raw XML bytes (bypasses MPQ).  Useful for unit
        tests that don't have an actual .sc2map file.

        If *cache_key* is already cached, returns the cached dict object
        (identity preserved) without re-parsing.
        """
        if cache_key in self._cache:
            return self._cache[cache_key]
        regions = _parse_regions_xml(xml_bytes)
        self._cache[cache_key] = regions
        return regions

    def _read_from_mpq(self, map_path: str) -> Dict[str, RegionDef]:
        path = Path(map_path)
        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")

        if not _MPYQ_AVAILABLE:
            raise ImportError(
                "mpyq is required to read SC2 map archives. "
                "Install it with: pip install mpyq"
            )

        archive = mpyq.MPQArchive(BytesIO(path.read_bytes()), listfile=False)

        raw: Optional[bytes] = archive.read_file("Regions")
        if not raw:
            raw = archive.read_file("(regions)")
        if not raw:
            raise RuntimeError(
                f"MPQ archive '{map_path}' has no 'Regions' entry. "
                "Ensure map regions are defined in the SC2 editor."
            )

        return _parse_regions_xml(raw)

    def get_center(self, regions: Dict[str, RegionDef], name: str) -> Point2:
        if name not in regions:
            raise KeyError(
                f"Region '{name}' not found.  Available: {sorted(regions)}"
            )
        return regions[name].center

    def find_units_in_region(
        self,
        region: RegionDef,
        units: List[Tuple[int, Point2]],
    ) -> List[int]:
        return [tag for tag, pos in units if region.contains(pos)]
