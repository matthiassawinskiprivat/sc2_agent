"""Inclusion rules for generated ``data/unit_data.json``.

Combat units with weapons are kept automatically. Weaponless units (workers,
buildings, casters, spawners) must appear in ``WHITELIST`` or
``WEAPON_UPGRADE_BONUS_DB`` or they are dropped on regenerate.
"""

from __future__ import annotations

from typing import FrozenSet

# Upgrade bonus table also acts as an inclusion allowlist for armed units.
WEAPON_UPGRADE_BONUS_DB = {
    "MARINE": 1, "MARAUDER": 1, "SIEGETANK": 3, "SIEGETANKSIEGED": 3, "THOR": 3, "THORAP": 3,
    "VIKINGFIGHTER": 1, "VIKINGASSAULT": 1, "BANSHEE": 1, "BATTLECRUISER": 1,
    "HELLION": 1, "HELLIONTANK": 2, "CYCLONE": 1, "LIBERATOR": 5, "LIBERATORAG": 5,
    "GHOST": 1, "REAPER": 1, "AUTOTURRET": 0,
    "ZERGLING": 1, "ROACH": 2, "RAVAGER": 2, "HYDRALISK": 1, "LURKERMP": 2,
    "MUTALISK": 1, "CORRUPTOR": 1, "ULTRALISK": 3, "BROODLORD": 2, "QUEEN": 1,
    "BANELING": 2, "INFESTORTERRAN": 1, "BROODLING": 1, "LOCUSTMP": 1, "LOCUSTMPFLYING": 1,
    "ZEALOT": 1, "STALKER": 1, "IMMORTAL": 2, "COLOSSUS": 1, "ARCHON": 3,
    "VOIDRAY": 1, "CARRIER": 1, "TEMPEST": 3, "ADEPT": 1, "PHOENIX": 1,
    "DARKTEMPLAR": 3, "ORACLE": 0, "INTERCEPTOR": 1, "MOTHERSHIP": 4,
    "HIGHTEMPLAR": 0,
    "SPINECRAWLER": 0, "SPORECRAWLER": 0, "MISSILETURRET": 0,
    "PHOTONCANNON": 0, "PLANETARYFORTRESS": 0, "BUNKER": 0,
}

# Units kept even with empty weapons (workers, buildings, casters, spawners).
WHITELIST: FrozenSet[str] = frozenset({
    "SCV", "PROBE", "DRONE", "OVERLORD", "OBSERVER", "WARPPRISM", "MEDIVAC",
    "COMMANDCENTER", "NEXUS", "HATCHERY", "LAIR", "HIVE", "PLANETARYFORTRESS", "ORBITALCOMMAND",
    "SUPPLYDEPOT", "PYLON", "GATEWAY", "WARPGATE", "BARRACKS", "FACTORY", "STARPORT",
    "ROBOTICSFACILITY", "STARGATE", "SPAWNINGPOOL", "ROACHWARREN", "HYDRALISKDEN", "SPIRE",
    "CREEPTUMOR", "CREEPTUMORBURROWED", "QUEEN", "LARVA", "EGG", "MULE",
    "PHOTONCANNON", "MISSILETURRET", "BUNKER", "SPINECRAWLER", "SPORECRAWLER", "SHIELDBATTERY",
    "INTERCEPTOR", "LOCUSTMP", "LOCUSTMPFLYING", "BROODLING", "AUTOTURRET", "HIGHTEMPLAR",
    "INFESTOR", "VIPER", "SENTRY", "SWARMHOSTMP", "DISRUPTOR", "DISRUPTORPHASED",
    # Unburrowed Widow Mine has no weapons; combat is on WIDOWMINEBURROWED.
    "WIDOWMINE",
    # Cloak curriculum free detectors (Observer already extracted with detect>0).
    "RAVEN", "OVERSEER",
})

# Applied after live extraction so regenerate keeps LotV-correct fields when the
# API/default path is wrong (e.g. missing live radius spawn → 0.0, sight default 9).
UNIT_FIELD_OVERRIDES = {
    "SWARMHOSTMP": {
        "cost": {"minerals": 100, "vespene": 75, "supply": 3.0},
        "radius": 0.8125,
        "sight": 10.0,
        "detect": 0.0,
        "weapons": [],
    },
    "DISRUPTOR": {
        "cost": {"minerals": 150, "vespene": 150, "supply": 3.0},
        "radius": 0.75,
        "sight": 9.0,
        "detect": 0.0,
        "weapons": [],
    },
    "WIDOWMINE": {
        "cost": {"minerals": 75, "vespene": 25, "supply": 2.0},
        "radius": 0.5,
        "sight": 7.0,
        "detect": 0.0,
        "weapons": [],
    },
    "RAVEN": {
        "cost": {"minerals": 100, "vespene": 200, "supply": 2.0},
        "radius": 0.625,
        "sight": 11.0,
        "detect": 11.0,
        "weapons": [],
    },
    "OVERSEER": {
        "cost": {"minerals": 150, "vespene": 50, "supply": 0.0},
        "radius": 0.875,
        "sight": 11.0,
        "detect": 11.0,
        "weapons": [],
    },
    # Baneling explode radius is splash-only; DPS uses melee contact range.
    "BANELING": {
        "weapons": [
            {
                "type": 1,
                "damage": 20.0,
                "attacks": 1,
                "period": 1.0,
                "range": 0.25,
                "upgrade_bonus": 2,
                "bonuses": [{"type": 1, "bonus": 15.0}],
            }
        ],
    },
}


def should_include_unit(name: str, has_weapons: bool) -> bool:
    """Return True if *name* should be written into unit_data.json."""
    return has_weapons or name in WHITELIST or name in WEAPON_UPGRADE_BONUS_DB


def apply_unit_field_overrides(unit_db: dict) -> None:
    """Merge ``UNIT_FIELD_OVERRIDES`` into *unit_db* in place."""
    for name, patch in UNIT_FIELD_OVERRIDES.items():
        if name not in unit_db:
            continue
        entry = unit_db[name]
        for key, value in patch.items():
            if key == "cost" and isinstance(value, dict):
                entry.setdefault("cost", {}).update(value)
            else:
                entry[key] = value
