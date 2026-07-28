"""Splash-damage catalog for observation layers and unit scalars.

Geometry is approximate: circle (or donut via min_range) centered on the unit.
Intensity is catalog splash damage. Idle DISRUPTOR is not splash / not onetime;
only DISRUPTORPHASED (Purification Nova ball) is among Disruptor forms.

``onetime_unit`` = Baneling + Nova (single-blast entities).
``is_temporary`` = timed lifetime summons / shades / drones / hallucinations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, FrozenSet, Optional, Tuple

if TYPE_CHECKING:
    from sc2.unit import Unit

# type name -> (radius_gu, min_range_gu, splash_damage)
SPLASH_BY_UNIT: Dict[str, Tuple[float, float, float]] = {
    "BANELING": (2.2, 0.0, 20.0),
    "DISRUPTORPHASED": (1.5, 0.0, 100.0),
    "SIEGETANKSIEGED": (13.0, 2.0, 40.0),
    "ARCHON": (3.0, 0.0, 25.0),
    "HELLION": (5.0, 0.0, 8.0),
    "HELLIONTANK": (2.0, 0.0, 18.0),
    "WIDOWMINEBURROWED": (5.0, 0.0, 40.0),
    "COLOSSUS": (7.0, 0.0, 10.0),
    "ULTRALISK": (2.0, 0.0, 12.0),
    "LURKERMP": (8.0, 0.0, 20.0),
    "PLANETARYFORTRESS": (7.0, 0.0, 40.0),
}

# Single-blast / suicide entities (not general timed summons).
ONETIME_UNIT_TYPES: FrozenSet[str] = frozenset({"BANELING", "DISRUPTORPHASED"})

# Unit types whose attack/ability splash can hurt allies (curriculum flag).
FRIENDLY_FIRE_UNIT_TYPES: FrozenSet[str] = frozenset(
    {
        "SIEGETANKSIEGED",
        "WIDOWMINEBURROWED",
        "DISRUPTORPHASED",
        "RAVAGER",
    }
)

# Units that despawn on a timer (or are hallucinations). Interceptors are excluded —
# they return to the Carrier rather than expiring on a lifetime buff.
TEMPORARY_LIFETIME_UNIT_TYPES: FrozenSet[str] = frozenset(
    {
        "DISRUPTORPHASED",
        "BROODLING",
        "LOCUSTMP",
        "LOCUSTMPFLYING",
        "ADEPTPHASESHIFT",
        "AUTOTURRET",
        "POINTDEFENSEDRONE",
        "INFESTORTERRAN",
        "INFESTORTERRANBURROWED",
        "CHANGELING",
        "CHANGELINGZEALOT",
        "CHANGELINGMARINE",
        "CHANGELINGMARINESHIELD",
        "CHANGELINGZERGLING",
        "CHANGELINGZERGLINGWINGS",
        "MULE",
    }
)


def _unit_type_name(unit: "Unit") -> str:
    type_id = getattr(unit, "type_id", None)
    name = getattr(type_id, "name", None)
    if name:
        return str(name).upper()
    return str(type_id or "").upper()


def splash_spec(unit: "Unit") -> Optional[Tuple[float, float, float]]:
    """Return (radius_gu, min_range_gu, splash_damage) or None if not splash."""
    return SPLASH_BY_UNIT.get(_unit_type_name(unit))


def unit_has_splash(unit: "Unit") -> bool:
    """True when the unit type deals catalogued splash damage."""
    return _unit_type_name(unit) in SPLASH_BY_UNIT


def unit_is_onetime(unit: "Unit") -> bool:
    """True for Baneling and Purification Nova ball (DISRUPTORPHASED)."""
    return _unit_type_name(unit) in ONETIME_UNIT_TYPES


def unit_has_friendly_fire(unit: "Unit") -> bool:
    """True when the unit type can deal friendly-fire splash / bile / nova."""
    return _unit_type_name(unit) in FRIENDLY_FIRE_UNIT_TYPES


def unit_is_temporary(unit: "Unit") -> bool:
    """True for timed-lifetime summons/shades/drones, or any hallucination."""
    if bool(getattr(unit, "is_hallucination", False)):
        return True
    return _unit_type_name(unit) in TEMPORARY_LIFETIME_UNIT_TYPES
