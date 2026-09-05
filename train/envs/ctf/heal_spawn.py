"""Heal-curriculum spawn helpers (guaranteed active healer, pre-damaged army)."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Prefer the longer-range caster when the pool lists more than one.
HEALER_PREFERENCE = ("QUEEN", "MEDIVAC", "SCV")

# debug_set_unit_value unit_value: 2 = current HP, 3 = current shields.
_HEALTH_UNIT_VALUE = 2
_SHIELD_UNIT_VALUE = 3
DAMAGED_UNITS_HEALTH_FRACTION = 0.5
_SKIP_TYPE_NAMES = frozenset(
    {
        "PYLON",
        "CREEPTUMOR",
        "CREEPTUMORBURROWED",
        "CREEPTUMORQUEEN",
    }
)


def guaranteed_healer_name(
    unit_pool: Iterable[str] | None,
    spawned_names: Iterable[str] | None,
) -> Optional[str]:
    """Healer type to append when the pool lists one but the budget roll missed.

    Preference is Queen, then Medivac, then SCV among types present in
    ``unit_pool``. Returns ``None`` when the pool has no active healer, or
    when the roll already includes one.
    """
    pool_upper = {str(name).upper() for name in (unit_pool or [])}
    spawned_upper = {str(name).upper() for name in (spawned_names or [])}
    available = [name for name in HEALER_PREFERENCE if name in pool_upper]
    if not available:
        return None
    if any(name in spawned_upper for name in HEALER_PREFERENCE):
        return None
    return available[0]


def damaged_units_enabled(training_conf: dict | None) -> bool:
    """True when ``env.damaged_units`` is set; default false."""
    conf = training_conf or {}
    env = conf.get("env") or {}
    val = conf.get("damaged_units", env.get("damaged_units", False))
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _iter_units(bot: Any) -> list:
    units = getattr(bot, "all_units", None)
    if units is None:
        return []
    try:
        return list(units)
    except TypeError:
        return []


def _type_name(unit: Any) -> str:
    tid = getattr(unit, "type_id", None)
    name = getattr(tid, "name", None)
    if name:
        return str(name).upper()
    return str(tid).upper() if tid is not None else ""


def damaged_unit_targets(
    bot: Any, fraction: float = DAMAGED_UNITS_HEALTH_FRACTION
) -> tuple[dict[float, list[int]], dict[float, list[int]]]:
    """Group tags by target HP and shield for ``debug_set_unit_value``.

    Skips structures, pylons, and creep tumors. Returns ``(health_groups, shield_groups)``.
    """
    health_groups: dict[float, list[int]] = defaultdict(list)
    shield_groups: dict[float, list[int]] = defaultdict(list)
    frac = min(max(float(fraction), 0.0), 1.0)
    for unit in _iter_units(bot):
        if bool(getattr(unit, "is_structure", False)):
            continue
        if _type_name(unit) in _SKIP_TYPE_NAMES:
            continue
        tag = int(getattr(unit, "tag", 0) or 0)
        if tag <= 0:
            continue
        try:
            health_max = float(getattr(unit, "health_max", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if health_max <= 0.0:
            continue
        health_groups[max(1.0, health_max * frac)].append(tag)
        try:
            shield_max = float(getattr(unit, "shield_max", 0.0) or 0.0)
        except (TypeError, ValueError):
            shield_max = 0.0
        if shield_max > 0.0:
            shield_groups[max(0.0, shield_max * frac)].append(tag)
    return health_groups, shield_groups


async def apply_damaged_units(
    bot: Any, fraction: float = DAMAGED_UNITS_HEALTH_FRACTION
) -> int:
    """Set mobile army HP (and shields) to ``fraction`` of max. Returns units touched."""
    if bot is None:
        return 0
    client = getattr(bot, "client", None)
    setter = getattr(client, "debug_set_unit_value", None) if client is not None else None
    if setter is None:
        return 0
    health_groups, shield_groups = damaged_unit_targets(bot, fraction=fraction)
    touched: set[int] = set()
    for value, tags in health_groups.items():
        try:
            await setter(tags, _HEALTH_UNIT_VALUE, float(value))
            touched.update(tags)
        except Exception as exc:
            logger.warning("damaged_units health set failed for %d tag(s): %s", len(tags), exc)
    for value, tags in shield_groups.items():
        try:
            await setter(tags, _SHIELD_UNIT_VALUE, float(value))
            touched.update(tags)
        except Exception as exc:
            logger.warning("damaged_units shield set failed for %d tag(s): %s", len(tags), exc)
    return len(touched)
