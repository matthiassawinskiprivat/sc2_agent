"""Validate SC2 map files used by training envs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from loguru import logger

# Truncated downloads are often exactly 1 MiB; real AIE ladder maps are ~4–8 MiB.
_MIN_SC2MAP_BYTES = 2_000_000


def is_usable_sc2map(path: Path, *, min_bytes: int = _MIN_SC2MAP_BYTES) -> bool:
    """True if ``path`` looks like a complete MPQ ``.SC2Map`` (not a truncated stub)."""
    try:
        if not path.is_file():
            return False
        size = path.stat().st_size
    except OSError:
        return False
    if size < int(min_bytes):
        return False
    try:
        with open(path, "rb") as handle:
            magic = handle.read(3)
    except OSError:
        return False
    return magic == b"MPQ"


def list_usable_sc2map_stems(maps_dir: Path | str) -> List[str]:
    """Sorted map stems under ``maps_dir``, skipping truncated/invalid files."""
    root = Path(maps_dir)
    if not root.is_dir():
        return []
    stems: List[str] = []
    for path in sorted(root.glob("*.SC2Map")):
        if is_usable_sc2map(path):
            stems.append(path.stem)
        else:
            logger.warning(
                f"Skipping unusable SC2 map {path.name} "
                f"(size={path.stat().st_size if path.exists() else '?'} bytes)"
            )
    return stems


def filter_usable_sc2map_paths(paths: Iterable[Path]) -> List[Path]:
    return [p for p in paths if is_usable_sc2map(Path(p))]
