"""
ObjectiveResolver — maps region names to live SC2 unit/structure tags.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from train.envs.ctf.config import CTFConfig
from train.envs.ctf.region_loader import Point2, RegionDef

try:
    from train.envs.ctf.game_data import uid as _uid
    _PYLON_TYPE_ID = _uid("PYLON")
except Exception:
    _PYLON_TYPE_ID = 60


class ObjectiveResolver:
    def __init__(
        self,
        config: CTFConfig,
        regions: Dict[str, RegionDef],
        sc2_client: Optional[Any],
    ) -> None:
        self.config = config
        self.regions = regions
        self._sc2_client = sc2_client
        self._pylon_tags: Dict[str, int] = {}
        self._region_to_attacker_tags: Dict[str, Set[int]] = {}
        self._resolved = False

    def resolve(self, units: Optional[List[Tuple[int, int, Point2]]] = None) -> None:
        if units is None:
            units = self._fetch_units()
        self._pylon_tags = {}
        for sq in self.config.squads:
            if sq.role != "defend":
                continue
            for anchor in sq.defend_anchors:
                self._pylon_tags[anchor] = self._find_pylon_in_region(
                    anchor, units, owner_side=sq.side
                )
        self._resolved = True

    def resolve_attacker_unit_tags(
        self,
        attacker_units: Dict[str, List[int]],
    ) -> None:
        self._region_to_attacker_tags = {}
        for sq in self.config.squads:
            if sq.role != "attack":
                continue
            tags = attacker_units.get(sq.id, [])
            for target_region in sq.attack_targets:
                existing = self._region_to_attacker_tags.setdefault(target_region, set())
                existing.update(tags)

    def pylon_tag_for_region(self, region_name: str) -> int:
        if not self._resolved:
            raise RuntimeError("Call resolve() before querying pylon tags.")
        if region_name not in self._pylon_tags:
            raise KeyError(f"No pylon tag resolved for region '{region_name}'.")
        return self._pylon_tags[region_name]

    def all_pylon_tags(self) -> Dict[str, int]:
        return dict(self._pylon_tags)

    def attacker_tags_for_region(self, region_name: str) -> FrozenSet[int]:
        return frozenset(self._region_to_attacker_tags.get(region_name, set()))

    def is_pylon_alive(self, region_name: str, alive_tags: Set[int]) -> bool:
        tag = self._pylon_tags.get(region_name)
        return tag is not None and tag in alive_tags

    def _fetch_units(self) -> List[Tuple[int, int, Point2]]:
        if self._sc2_client is None:
            return []
        return []

    def _find_pylon_in_region(
        self,
        region_name: str,
        units: List[Tuple[int, int, Point2]],
        owner_side: str,
    ) -> int:
        if region_name not in self.regions:
            raise KeyError(
                f"Defend anchor region '{region_name}' not in map regions. "
                f"Available: {sorted(self.regions)}"
            )
        region = self.regions[region_name]
        pylons_in_region = [
            tag for tag, type_id, pos in units
            if type_id == _PYLON_TYPE_ID and region.contains(pos)
        ]
        assert len(pylons_in_region) == 1, (
            f"Expected exactly 1 pylon in region '{region_name}', "
            f"found {len(pylons_in_region)}."
        )
        return pylons_in_region[0]
