"""
Region consistency test for train/maps/ladder/with_region/*.SC2Map.

Checks the per-map invariant required for region-based CTF spawning:
  - Every attacker region '{n}_a' must have a matching defender region '{n}_d'.

Defender-only regions ('{n}_d' without a corresponding '{n}_a') are allowed —
they are silently skipped during spawning.  Different maps may also have
different total numbers of usable pairs.

Requires mpyq (pip install mpyq) to read SC2 map archives.
"""
import re
import pytest
from pathlib import Path

from train.envs.ctf.region_loader import RegionLoader, _MPYQ_AVAILABLE

MAPS_DIR = Path("train/maps/ladder/with_region")
PAIR_RE = re.compile(r"^(\d+)_(a|d)$")


def _parse_pairs(regions: dict) -> dict:
    """Return {pair_index: set_of_sides} for all regions matching the naming scheme."""
    pairs: dict[int, set] = {}
    for name in regions:
        m = PAIR_RE.match(name)
        if m:
            idx = int(m.group(1))
            pairs.setdefault(idx, set()).add(m.group(2))
    return pairs


@pytest.mark.skipif(
    not MAPS_DIR.exists(),
    reason=f"Map directory '{MAPS_DIR}' not present — skipping region tests.",
)
@pytest.mark.skipif(
    not _MPYQ_AVAILABLE,
    reason="mpyq not installed — install with 'pip install mpyq' to run region tests.",
)
def test_all_maps_have_consistent_region_pairs():
    """
    For every SC2Map in the with_region pool:
    - at least one usable (complete) region pair must exist
    - every attacker region '{n}_a' must have a matching defender '{n}_d'
      (the converse is not required — defender-only markers are allowed)
    """
    loader = RegionLoader()
    map_files = sorted(MAPS_DIR.glob("*.SC2Map"))
    assert map_files, f"No .SC2Map files found in {MAPS_DIR}"

    for map_path in map_files:
        regions = loader.load(str(map_path))
        pairs = _parse_pairs(regions)

        # Identify attacker indices that lack a defender
        attackers_without_defender = {
            idx for idx, sides in pairs.items()
            if "a" in sides and "d" not in sides
        }
        assert not attackers_without_defender, (
            f"{map_path.name}: attacker regions {sorted(attackers_without_defender)} "
            "have no matching '_d' defender region."
        )

        # At least one complete pair (both _a and _d) must exist per map
        complete_pairs = {idx for idx, sides in pairs.items() if sides == {"a", "d"}}
        assert complete_pairs, (
            f"{map_path.name}: no complete region pairs found. "
            f"Available regions: {sorted(regions.keys())}"
        )
