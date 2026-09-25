"""
Static SC2 game data: unit costs, unit attributes, and weapon bonuses.

Integer IDs correspond to python-sc2 ``UnitTypeId`` enum values.
If python-sc2 is available the constants are read from the library;
otherwise well-known fallback integers are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set


def _load_ids() -> Dict[str, int]:
    try:
        from sc2.ids.unit_typeid import UnitTypeId
        return {name: uid.value for name, uid in UnitTypeId.__members__.items()}
    except ImportError:
        pass
    return {
        "MARINE": 48, "MARAUDER": 51, "REAPER": 49, "GHOST": 50,
        "HELLION": 53, "HELLIONTANK": 484, "SIEGETANK": 33, "SIEGETANKSIEGED": 32,
        "WIDOWMINE": 498, "WIDOWMINEBURROWED": 500, "CYCLONE": 692,
        "THOR": 52, "THORAP": 516, "VIKINGFIGHTER": 35, "VIKINGASSAULT": 34,
        "MEDIVAC": 54, "BANSHEE": 55, "BATTLECRUISER": 57, "RAVEN": 56,
        "LIBERATOR": 689, "LIBERATORAG": 734,
        "SCV": 45, "MULE": 268, "COMMANDCENTER": 18, "BARRACKS": 21,
        "FACTORY": 27, "STARPORT": 28,
        "PROBE": 84, "ZEALOT": 73, "STALKER": 74, "SENTRY": 77, "ADEPT": 311,
        "HIGHTEMPLAR": 75, "DARKTEMPLAR": 76, "IMMORTAL": 83, "COLOSSUS": 4,
        "DISRUPTOR": 694, "ARCHON": 141, "OBSERVER": 82, "WARPPRISM": 81,
        "VOIDRAY": 80, "PHOENIX": 78, "ORACLE": 495, "CARRIER": 79,
        "TEMPEST": 396, "MOTHERSHIP": 10,
        "PYLON": 60, "NEXUS": 59, "GATEWAY": 62, "ROBOTICSFACILITY": 71, "STARGATE": 67,
        "DRONE": 104, "ZERGLING": 105, "BANELING": 9, "ROACH": 110, "RAVAGER": 688,
        "HYDRALISK": 107, "LURKERMP": 502, "LURKERMPBURROWED": 503,
        "INFESTOR": 111, "SWARMHOSTMP": 494, "LOCUSTMP": 489,
        "ULTRALISK": 109, "MUTALISK": 106, "CORRUPTOR": 112, "BROODLORD": 114,
        "VIPER": 499, "OVERLORD": 106, "OVERLORDTRANSPORT": 893, "OVERSEER": 129, "QUEEN": 126,
        "HATCHERY": 86, "LAIR": 100, "HIVE": 101,
        "SPINECRAWLER": 98, "SPORECRAWLER": 99,
        "SHIELDBATTERY": 1910, "WARPPRISMPHASING": 136,
        "INTERCEPTOR": 85, "LOCUSTMPFLYING": 693, "BROODLING": 289,
        "OBSERVERSIEGEMODE": 1911, "OVERSEERSIEGEMODE": 1912,
        "ZERGLINGBURROWED": 119, "BANELINGBURROWED": 115, "ROACHBURROWED": 118,
        "RAVAGERBURROWED": 690, "HYDRALISKBURROWED": 117, "INFESTORBURROWED": 127,
        "SWARMHOSTBURROWEDMP": 493, "ULTRALISKBURROWED": 131, "QUEENBURROWED": 125,
        "DRONEBURROWED": 116, "LURKERBURROWED": 912,
    }


_ID_TABLE: Dict[str, int] = _load_ids()


def uid(name: str) -> int:
    """Return integer UnitTypeId for *name*.  Raises ``KeyError`` if unknown."""
    return _ID_TABLE[name.upper()]


@dataclass(frozen=True)
class UnitInfo:
    type_id: int
    name: str
    mineral_cost: int
    vespene_cost: int
    attributes: FrozenSet[str]
    footprint_radius: float = 0.5


@dataclass(frozen=True)
class WeaponBonus:
    attacker_type_id: int
    target_attr: str
    bonus: float


_UNIT_LIST: list[UnitInfo] = [
    UnitInfo(uid("MARINE"),      "Marine",      50,  0,  frozenset({"biological", "light"}),            0.375),
    UnitInfo(uid("MARAUDER"),    "Marauder",    100, 25, frozenset({"biological", "armored"}),           0.5),
    UnitInfo(uid("REAPER"),      "Reaper",      50,  50, frozenset({"biological", "light"}),             0.375),
    UnitInfo(uid("GHOST"),       "Ghost",       200, 100, frozenset({"biological", "psionic", "light"}), 0.375),
    UnitInfo(uid("HELLION"),     "Hellion",     100, 0,  frozenset({"mechanical", "light"}),             1.0),
    UnitInfo(uid("HELLIONTANK"), "Hellbat",     100, 0,  frozenset({"mechanical", "biological", "light"}), 1.0),
    UnitInfo(uid("SIEGETANK"),   "Siege Tank",  150, 125, frozenset({"mechanical", "armored"}),          0.875),
    UnitInfo(uid("CYCLONE"),     "Cyclone",     150, 100, frozenset({"mechanical", "armored"}),          1.0),
    UnitInfo(uid("WIDOWMINE"),   "Widow Mine",  75,  25, frozenset({"mechanical", "light"}),             0.5),
    UnitInfo(uid("THOR"),        "Thor",        300, 200, frozenset({"mechanical", "armored", "massive"}), 1.75),
    UnitInfo(uid("THORAP"),      "Thor AP",     300, 200, frozenset({"mechanical", "armored", "massive"}), 1.75),
    UnitInfo(uid("VIKINGFIGHTER"), "Viking",    150, 75, frozenset({"mechanical", "armored"}),           1.0),
    UnitInfo(uid("MEDIVAC"),     "Medivac",     100, 100, frozenset({"mechanical", "armored"}),          1.0),
    UnitInfo(uid("BANSHEE"),     "Banshee",     150, 100, frozenset({"mechanical", "light"}),            1.0),
    UnitInfo(uid("RAVEN"),       "Raven",       100, 200, frozenset({"mechanical", "armored"}),          1.0),
    UnitInfo(uid("BATTLECRUISER"), "Battlecruiser", 400, 300, frozenset({"mechanical", "armored", "massive"}), 1.25),
    UnitInfo(uid("ZEALOT"),      "Zealot",      100, 0,  frozenset({"light"}),                          0.5),
    UnitInfo(uid("STALKER"),     "Stalker",     125, 50, frozenset({"armored"}),                        0.625),
    UnitInfo(uid("SENTRY"),      "Sentry",      50,  100, frozenset({"mechanical", "light", "psionic"}), 0.5),
    UnitInfo(uid("ADEPT"),       "Adept",       100, 25, frozenset({"light"}),                          0.5),
    UnitInfo(uid("HIGHTEMPLAR"), "High Templar", 50, 150, frozenset({"psionic", "light"}),              0.5),
    UnitInfo(uid("DARKTEMPLAR"), "Dark Templar", 125, 125, frozenset({"psionic", "light"}),             0.5),
    UnitInfo(uid("IMMORTAL"),    "Immortal",    275, 100, frozenset({"mechanical", "armored"}),          1.0),
    UnitInfo(uid("COLOSSUS"),    "Colossus",    300, 200, frozenset({"mechanical", "armored", "massive", "light"}), 1.5),
    UnitInfo(uid("DISRUPTOR"),   "Disruptor",   150, 150, frozenset({"mechanical", "armored"}),          0.75),
    # Composite cost: an Archon is merged from two templars and can never be
    # bought directly, so its own build cost in the game data is 0/0. Priced here
    # as High Templar + Dark Templar (175/275) — a 0/0 army unit would be free to
    # lose and worthless to kill under the economic reward.
    UnitInfo(uid("ARCHON"),      "Archon",      175, 275, frozenset({"psionic", "massive"}),             1.0),
    UnitInfo(uid("VOIDRAY"),     "Void Ray",    250, 150, frozenset({"mechanical", "armored"}),          1.0),
    UnitInfo(uid("PHOENIX"),     "Phoenix",     150, 100, frozenset({"mechanical", "light"}),            1.0),
    UnitInfo(uid("ORACLE"),      "Oracle",      150, 150, frozenset({"mechanical", "light"}),            1.0),
    UnitInfo(uid("CARRIER"),     "Carrier",     350, 250, frozenset({"mechanical", "armored", "massive"}), 1.75),
    UnitInfo(uid("TEMPEST"),     "Tempest",     300, 200, frozenset({"mechanical", "armored", "massive"}), 1.75),
    UnitInfo(uid("ZERGLING"),    "Zergling",    25,  0,  frozenset({"biological", "light"}),             0.375),
    # Composite cost: Zergling (25/0) plus the 25/25 morph.
    UnitInfo(uid("BANELING"),    "Baneling",    50,  25, frozenset({"biological", "light"}),             0.375),
    UnitInfo(uid("ROACH"),       "Roach",       75,  25, frozenset({"biological", "armored"}),           0.75),
    UnitInfo(uid("RAVAGER"),     "Ravager",     100, 100, frozenset({"biological", "armored"}),          0.75),
    UnitInfo(uid("HYDRALISK"),   "Hydralisk",   100, 50, frozenset({"biological", "light"}),             0.625),
    UnitInfo(uid("LURKERMP"),    "Lurker",      150, 150, frozenset({"biological", "armored"}),          0.75),
    UnitInfo(uid("INFESTOR"),    "Infestor",    100, 150, frozenset({"biological", "armored"}),          0.75),
    UnitInfo(uid("SWARMHOSTMP"), "Swarm Host",  100, 75,  frozenset({"biological", "armored"}),          0.8125),
    UnitInfo(uid("ULTRALISK"),   "Ultralisk",   300, 200, frozenset({"biological", "armored", "massive"}), 1.0),
    UnitInfo(uid("MUTALISK"),    "Mutalisk",    100, 100, frozenset({"biological", "light"}),            1.0),
    UnitInfo(uid("CORRUPTOR"),   "Corruptor",   150, 100, frozenset({"biological", "armored"}),          1.0),
    UnitInfo(uid("BROODLORD"),   "Brood Lord",  300, 250, frozenset({"biological", "armored", "massive"}), 1.75),
    UnitInfo(uid("QUEEN"),       "Queen",       150, 0,  frozenset({"biological", "psionic"}),           0.75),
    UnitInfo(uid("OVERLORD"),    "Overlord",    100, 0,  frozenset({"biological", "armored"}),            1.0),
    UnitInfo(uid("OVERLORDTRANSPORT"), "Overlord Transport", 125, 25, frozenset({"biological", "armored"}), 1.0),
    UnitInfo(uid("WARPPRISM"),   "Warp Prism",  200, 0,  frozenset({"mechanical", "armored", "psionic"}), 0.875),
    UnitInfo(uid("VIPER"),       "Viper",       100, 200, frozenset({"biological", "armored", "psionic"}), 0.75),
    UnitInfo(uid("LIBERATOR"),   "Liberator",   150, 150, frozenset({"mechanical", "armored"}),          0.75),
    UnitInfo(uid("MOTHERSHIP"),  "Mothership",  400, 400, frozenset({"mechanical", "armored", "massive", "psionic"}), 1.5),
    # Composite cost: Overlord (100/0) plus the 50/50 Overseer morph.
    UnitInfo(uid("OVERSEER"),    "Overseer",    150, 50, frozenset({"biological", "armored"}),            1.0),
    UnitInfo(uid("OBSERVER"),    "Observer",    25,  75, frozenset({"mechanical", "light", "detector"}),  0.5),
    # Worker units: heal/repair actors in the CTF healing curriculum, so they need
    # a real price even though they never fight.
    UnitInfo(uid("SCV"),         "SCV",         50,  0,  frozenset({"mechanical", "biological", "light"}), 0.375),
    UnitInfo(uid("PROBE"),       "Probe",       50,  0,  frozenset({"mechanical", "light"}),              0.375),
    UnitInfo(uid("DRONE"),       "Drone",       50,  0,  frozenset({"biological", "light"}),              0.375),
]

UNIT_DATA: Dict[int, UnitInfo] = {u.type_id: u for u in _UNIT_LIST}

# Transformed / morphed / burrowed bodies that are the *same* purchased unit.
# Economic reward accounting must price a Sieged Tank exactly like a Siege Tank,
# otherwise sieging would register as the whole army value vanishing and
# reappearing. Keys resolve to the canonical type before any table lookup.
_TRANSFORM_ALIAS_NAMES: Dict[str, str] = {
    "SIEGETANKSIEGED": "SIEGETANK",
    "WIDOWMINEBURROWED": "WIDOWMINE",
    "VIKINGASSAULT": "VIKINGFIGHTER",
    "LIBERATORAG": "LIBERATOR",
    "WARPPRISMPHASING": "WARPPRISM",
    "OBSERVERSIEGEMODE": "OBSERVER",
    "OVERSEERSIEGEMODE": "OVERSEER",
    "LURKERMPBURROWED": "LURKERMP",
    "LURKERBURROWED": "LURKERMP",
    "ZERGLINGBURROWED": "ZERGLING",
    "BANELINGBURROWED": "BANELING",
    "ROACHBURROWED": "ROACH",
    "RAVAGERBURROWED": "RAVAGER",
    "HYDRALISKBURROWED": "HYDRALISK",
    "INFESTORBURROWED": "INFESTOR",
    "SWARMHOSTBURROWEDMP": "SWARMHOSTMP",
    "ULTRALISKBURROWED": "ULTRALISK",
    "QUEENBURROWED": "QUEEN",
    "DRONEBURROWED": "DRONE",
}

TRANSFORM_ALIASES: Dict[int, int] = {}
for _alias, _canonical in _TRANSFORM_ALIAS_NAMES.items():
    try:
        TRANSFORM_ALIASES[uid(_alias)] = uid(_canonical)
    except KeyError:  # pragma: no cover - alias absent from this SC2 build
        continue

# Temporary bodies: summoned, spawned, or hallucinated units that were never
# purchased. Economic reward must ignore them or a Carrier's Interceptor churn
# reads as army value appearing and dying every few steps.
_TEMPORARY_UNIT_NAMES = (
    "INTERCEPTOR", "LOCUSTMP", "LOCUSTMPFLYING", "BROODLING",
    "ADEPTPHASESHIFT", "DISRUPTORPHASED", "CHANGELING",
    "INFESTEDTERRAN", "INFESTORTERRAN", "MULE",
)

TEMPORARY_UNIT_IDS: FrozenSet[int] = frozenset(
    _ID_TABLE[name] for name in _TEMPORARY_UNIT_NAMES if name in _ID_TABLE
)


def resolve_type_id(type_id: int) -> int:
    """Canonical purchased type for a transformed/burrowed body."""
    try:
        key = int(type_id)
    except (TypeError, ValueError):
        return -1
    return TRANSFORM_ALIASES.get(key, key)


def is_temporary_unit_type(type_id: int) -> bool:
    """True for summons/hallucinations that carry no purchased economic value."""
    try:
        return int(type_id) in TEMPORARY_UNIT_IDS
    except (TypeError, ValueError):
        return False

_WEAPON_BONUS_LIST: list[WeaponBonus] = [
    WeaponBonus(uid("HELLION"),      "light",      5.0),
    WeaponBonus(uid("HELLIONTANK"),  "light",      12.0),
    WeaponBonus(uid("MARAUDER"),     "armored",    10.0),
    WeaponBonus(uid("GHOST"),        "biological", 15.0),
    WeaponBonus(uid("SIEGETANK"),    "armored",    30.0),
    WeaponBonus(uid("CYCLONE"),      "armored",    6.0),
    WeaponBonus(uid("THOR"),         "light",      12.0),
    WeaponBonus(uid("BATTLECRUISER"), "mechanical", 8.0),
    WeaponBonus(uid("STALKER"),      "armored",    4.0),
    WeaponBonus(uid("ADEPT"),        "light",      12.0),
    WeaponBonus(uid("HIGHTEMPLAR"),  "light",      3.0),
    WeaponBonus(uid("IMMORTAL"),     "armored",    30.0),
    WeaponBonus(uid("COLOSSUS"),     "light",      5.0),
    WeaponBonus(uid("VOIDRAY"),      "armored",    6.0),
    WeaponBonus(uid("TEMPEST"),      "massive",    40.0),
    WeaponBonus(uid("BANELING"),     "light",      15.0),
    WeaponBonus(uid("ROACH"),        "armored",    0.0),
    WeaponBonus(uid("HYDRALISK"),    "armored",    0.0),
    WeaponBonus(uid("LURKERMP"),     "armored",    10.0),
    WeaponBonus(uid("ULTRALISK"),    "armored",    15.0),
    WeaponBonus(uid("CORRUPTOR"),    "massive",    4.0),
]

WEAPON_BONUS: Dict[tuple[int, str], float] = {
    (wb.attacker_type_id, wb.target_attr): wb.bonus
    for wb in _WEAPON_BONUS_LIST
    if wb.bonus > 0.0
}


def get_unit_info(type_id: int) -> Optional[UnitInfo]:
    return UNIT_DATA.get(resolve_type_id(type_id))


def get_weapon_bonus(attacker_type_id: int, target_type_id: int) -> float:
    attacker_id = resolve_type_id(attacker_type_id)
    attacker_info = UNIT_DATA.get(attacker_id)
    target_info = get_unit_info(target_type_id)
    if attacker_info is None or target_info is None:
        return 0.0
    total = 0.0
    for attr in target_info.attributes:
        total += WEAPON_BONUS.get((attacker_id, attr), 0.0)
    return total


def get_unit_cost(type_id: int) -> tuple[int, int]:
    info = get_unit_info(type_id)
    if info is None:
        return (0, 0)
    return (info.mineral_cost, info.vespene_cost)


def get_footprint_radius(type_id: int) -> float:
    info = get_unit_info(type_id)
    return info.footprint_radius if info is not None else 0.5
