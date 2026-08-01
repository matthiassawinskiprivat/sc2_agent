"""Combat weapon/armor/shield upgrade tracks for CTF ``env_mode: upgrades``.

Spell/caster upgrades are intentionally excluded. Costs follow standard ladder
tiers (L1 100/100, L2 175/175, L3 250/250).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

try:
    from sc2.ids.upgrade_id import UpgradeId

    _SC2 = True
except ImportError:  # pragma: no cover
    UpgradeId = object  # type: ignore
    _SC2 = False


# Per-level step cost (minerals, vespene). Cumulative for tier k = sum of steps 1..k.
UPGRADE_STEP_COST: Tuple[Tuple[int, int], ...] = (
    (0, 0),  # tier 0 placeholder
    (100, 100),
    (175, 175),
    (250, 250),
)


def cumulative_upgrade_cost(tier: int) -> Tuple[int, int]:
    """Total minerals/vespene to reach ``tier`` (0–3)."""
    t = max(0, min(3, int(tier)))
    m = v = 0
    for step in range(1, t + 1):
        dm, dv = UPGRADE_STEP_COST[step]
        m += dm
        v += dv
    return m, v


@dataclass(frozen=True)
class CombatTrack:
    """One research line (e.g. Terran infantry weapons)."""

    name: str
    race: str
    # UpgradeId names for levels 1..3 (resolved at runtime if sc2 available)
    upgrade_names: Tuple[str, str, str]
    # Structures that can research this track (first available wins)
    building_names: Tuple[str, ...]


# Weapons + armor + shields only (no Stim, Combat Shield, Charge, …).
_COMBAT_TRACKS: Tuple[CombatTrack, ...] = (
    # Terran
    CombatTrack(
        "terran_infantry_weapons",
        "terran",
        (
            "TERRANINFANTRYWEAPONSLEVEL1",
            "TERRANINFANTRYWEAPONSLEVEL2",
            "TERRANINFANTRYWEAPONSLEVEL3",
        ),
        ("ENGINEERINGBAY",),
    ),
    CombatTrack(
        "terran_infantry_armors",
        "terran",
        (
            "TERRANINFANTRYARMORSLEVEL1",
            "TERRANINFANTRYARMORSLEVEL2",
            "TERRANINFANTRYARMORSLEVEL3",
        ),
        ("ENGINEERINGBAY",),
    ),
    CombatTrack(
        "terran_vehicle_weapons",
        "terran",
        (
            "TERRANVEHICLEWEAPONSLEVEL1",
            "TERRANVEHICLEWEAPONSLEVEL2",
            "TERRANVEHICLEWEAPONSLEVEL3",
        ),
        ("ARMORY",),
    ),
    CombatTrack(
        "terran_vehicle_armors",
        "terran",
        (
            "TERRANVEHICLEARMORSLEVEL1",
            "TERRANVEHICLEARMORSLEVEL2",
            "TERRANVEHICLEARMORSLEVEL3",
        ),
        ("ARMORY",),
    ),
    CombatTrack(
        "terran_ship_weapons",
        "terran",
        (
            "TERRANSHIPWEAPONSLEVEL1",
            "TERRANSHIPWEAPONSLEVEL2",
            "TERRANSHIPWEAPONSLEVEL3",
        ),
        ("ARMORY",),
    ),
    CombatTrack(
        "terran_ship_armors",
        "terran",
        (
            "TERRANSHIPARMORSLEVEL1",
            "TERRANSHIPARMORSLEVEL2",
            "TERRANSHIPARMORSLEVEL3",
        ),
        ("ARMORY",),
    ),
    # Protoss
    CombatTrack(
        "protoss_ground_weapons",
        "protoss",
        (
            "PROTOSSGROUNDWEAPONSLEVEL1",
            "PROTOSSGROUNDWEAPONSLEVEL2",
            "PROTOSSGROUNDWEAPONSLEVEL3",
        ),
        ("FORGE",),
    ),
    CombatTrack(
        "protoss_ground_armors",
        "protoss",
        (
            "PROTOSSGROUNDARMORSLEVEL1",
            "PROTOSSGROUNDARMORSLEVEL2",
            "PROTOSSGROUNDARMORSLEVEL3",
        ),
        ("FORGE",),
    ),
    CombatTrack(
        "protoss_air_weapons",
        "protoss",
        (
            "PROTOSSAIRWEAPONSLEVEL1",
            "PROTOSSAIRWEAPONSLEVEL2",
            "PROTOSSAIRWEAPONSLEVEL3",
        ),
        ("CYBERNETICSCORE",),
    ),
    CombatTrack(
        "protoss_air_armors",
        "protoss",
        (
            "PROTOSSAIRARMORSLEVEL1",
            "PROTOSSAIRARMORSLEVEL2",
            "PROTOSSAIRARMORSLEVEL3",
        ),
        ("CYBERNETICSCORE",),
    ),
    CombatTrack(
        "protoss_shields",
        "protoss",
        (
            "PROTOSSSHIELDSLEVEL1",
            "PROTOSSSHIELDSLEVEL2",
            "PROTOSSSHIELDSLEVEL3",
        ),
        ("FORGE",),
    ),
    # Zerg
    CombatTrack(
        "zerg_melee_weapons",
        "zerg",
        (
            "ZERGMELEEWEAPONSLEVEL1",
            "ZERGMELEEWEAPONSLEVEL2",
            "ZERGMELEEWEAPONSLEVEL3",
        ),
        ("EVOLUTIONCHAMBER",),
    ),
    CombatTrack(
        "zerg_missile_weapons",
        "zerg",
        (
            "ZERGMISSILEWEAPONSLEVEL1",
            "ZERGMISSILEWEAPONSLEVEL2",
            "ZERGMISSILEWEAPONSLEVEL3",
        ),
        ("EVOLUTIONCHAMBER",),
    ),
    CombatTrack(
        "zerg_ground_armors",
        "zerg",
        (
            "ZERGGROUNDARMORSLEVEL1",
            "ZERGGROUNDARMORSLEVEL2",
            "ZERGGROUNDARMORSLEVEL3",
        ),
        ("EVOLUTIONCHAMBER",),
    ),
    CombatTrack(
        "zerg_flyer_weapons",
        "zerg",
        (
            "ZERGFLYERWEAPONSLEVEL1",
            "ZERGFLYERWEAPONSLEVEL2",
            "ZERGFLYERWEAPONSLEVEL3",
        ),
        ("SPIRE", "GREATERSPIRE"),
    ),
    CombatTrack(
        "zerg_flyer_armors",
        "zerg",
        (
            "ZERGFLYERARMORSLEVEL1",
            "ZERGFLYERARMORSLEVEL2",
            "ZERGFLYERARMORSLEVEL3",
        ),
        ("SPIRE", "GREATERSPIRE"),
    ),
)


def combat_tracks_for_race(race: str) -> List[CombatTrack]:
    r = str(race or "").lower().strip()
    return [t for t in _COMBAT_TRACKS if t.race == r]


def all_combat_upgrade_names() -> frozenset:
    names: set[str] = set()
    for track in _COMBAT_TRACKS:
        names.update(track.upgrade_names)
    return frozenset(names)


def _resolve_upgrade(name: str):
    if not _SC2:
        return name
    return getattr(UpgradeId, name)


def upgrade_ids_for_tiers(chosen: Mapping[str, int]) -> List:
    """Expand track→tier map to ordered UpgradeId list (L1 then L2 then L3)."""
    by_name = {t.name: t for t in _COMBAT_TRACKS}
    out: List = []
    for track_name, tier in chosen.items():
        track = by_name.get(track_name)
        if track is None:
            continue
        t = max(0, min(3, int(tier)))
        for level in range(t):
            out.append(_resolve_upgrade(track.upgrade_names[level]))
    return out


def buildings_needed_for_tracks(chosen: Mapping[str, int]) -> List[str]:
    """Unique research building type names required for non-zero tiers."""
    by_name = {t.name: t for t in _COMBAT_TRACKS}
    needed: List[str] = []
    seen: set[str] = set()
    for track_name, tier in chosen.items():
        if int(tier) <= 0:
            continue
        track = by_name.get(track_name)
        if track is None:
            continue
        bname = track.building_names[0]
        if bname not in seen:
            seen.add(bname)
            needed.append(bname)
    return needed


def sample_pool_upgrades(
    race: str,
    mineral_budget: int,
    vespene_budget: int,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, int], int, int]:
    """Sample independent tiers 0–3 per combat track; charge cumulative cost.

    Returns ``(chosen_tiers, remaining_minerals, remaining_vespene)``.
    Tracks that cannot fit are lowered until affordable (possibly 0).
    """
    rng = rng or random.Random()
    m = int(mineral_budget)
    v = int(vespene_budget)
    chosen: Dict[str, int] = {}
    tracks = list(combat_tracks_for_race(race))
    rng.shuffle(tracks)
    for track in tracks:
        want = int(rng.randint(0, 3))
        tier = want
        while tier > 0:
            cm, cv = cumulative_upgrade_cost(tier)
            if cm <= m and cv <= v:
                break
            tier -= 1
        if tier > 0:
            cm, cv = cumulative_upgrade_cost(tier)
            m -= cm
            v -= cv
        chosen[track.name] = tier
    # Stable key order for determinism in tests / logging
    ordered = {t.name: chosen.get(t.name, 0) for t in combat_tracks_for_race(race)}
    return ordered, m, v
