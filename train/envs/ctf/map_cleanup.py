"""Reveal concealed CTF leftovers so soft-reset ``debug_kill`` can remove them."""
from __future__ import annotations

from typing import Any, Optional

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

# Keep aligned with CaptureTheFlagBot._BURROW_UP_BY_TYPE.
_BURROW_UP_BY_TYPE = {
    UnitTypeId.ZERGLINGBURROWED: AbilityId.BURROWUP_ZERGLING,
    UnitTypeId.BANELINGBURROWED: AbilityId.BURROWUP_BANELING,
    UnitTypeId.ROACHBURROWED: AbilityId.BURROWUP_ROACH,
    UnitTypeId.RAVAGERBURROWED: AbilityId.BURROWUP_RAVAGER,
    UnitTypeId.HYDRALISKBURROWED: AbilityId.BURROWUP_HYDRALISK,
    UnitTypeId.INFESTORBURROWED: AbilityId.BURROWUP_INFESTOR,
    UnitTypeId.SWARMHOSTBURROWEDMP: AbilityId.BURROWUP_SWARMHOST,
    UnitTypeId.ULTRALISKBURROWED: AbilityId.BURROWUP_ULTRALISK,
    UnitTypeId.QUEENBURROWED: AbilityId.BURROWUP_QUEEN,
    UnitTypeId.DRONEBURROWED: AbilityId.BURROWUP_DRONE,
    UnitTypeId.WIDOWMINEBURROWED: AbilityId.BURROWUP_WIDOWMINE,
    UnitTypeId.LURKERMPBURROWED: AbilityId.BURROWUP_LURKER,
    UnitTypeId.LURKERBURROWED: AbilityId.BURROWUP_LURKER,
}

# Surface type while ``is_burrowed`` is already True (morph not updated yet).
_BURROW_UP_FROM_SURFACE = {
    UnitTypeId.ZERGLING: AbilityId.BURROWUP_ZERGLING,
    UnitTypeId.BANELING: AbilityId.BURROWUP_BANELING,
    UnitTypeId.ROACH: AbilityId.BURROWUP_ROACH,
    UnitTypeId.RAVAGER: AbilityId.BURROWUP_RAVAGER,
    UnitTypeId.HYDRALISK: AbilityId.BURROWUP_HYDRALISK,
    UnitTypeId.INFESTOR: AbilityId.BURROWUP_INFESTOR,
    UnitTypeId.SWARMHOSTMP: AbilityId.BURROWUP_SWARMHOST,
    UnitTypeId.ULTRALISK: AbilityId.BURROWUP_ULTRALISK,
    UnitTypeId.QUEEN: AbilityId.BURROWUP_QUEEN,
    UnitTypeId.DRONE: AbilityId.BURROWUP_DRONE,
    UnitTypeId.WIDOWMINE: AbilityId.BURROWUP_WIDOWMINE,
    UnitTypeId.LURKERMP: AbilityId.BURROWUP_LURKER,
}

_CLOAK_OFF_BY_TYPE = {
    UnitTypeId.GHOST: AbilityId.BEHAVIOR_CLOAKOFF_GHOST,
    UnitTypeId.BANSHEE: AbilityId.BEHAVIOR_CLOAKOFF_BANSHEE,
}

# Cloaking Field is channeled (Mothership / Oracle). LotV has no cloak-off id;
# STOP cancels the channel so nearby units uncloak before debug_kill.
_FIELD_CLOAK_STOP_TYPES = frozenset(
    {
        UnitTypeId.MOTHERSHIP,
        UnitTypeId.ORACLE,
    }
)


def reveal_ability_for_cleanup(unit: Any) -> Optional[AbilityId]:
    """Ability that surfaces a cloaked/burrowed unit, or None if already visible.

    Dark Templar stay permanently cloaked; ``debug_kill`` has to handle them.
    Mothership / Oracle cloaking is an area field — cancel it with STOP.
    """
    tid = getattr(unit, "type_id", None)
    if tid is None or tid == UnitTypeId.DARKTEMPLAR:
        return None
    if tid in _FIELD_CLOAK_STOP_TYPES:
        return AbilityId.STOP
    burrowed_up = _BURROW_UP_BY_TYPE.get(tid)
    if burrowed_up is not None:
        return burrowed_up
    if getattr(unit, "is_burrowed", False) is True:
        return _BURROW_UP_FROM_SURFACE.get(tid, AbilityId.BURROWUP)
    cloak_off = _CLOAK_OFF_BY_TYPE.get(tid)
    if cloak_off is not None:
        return cloak_off
    if getattr(unit, "is_cloaked", False) is True:
        return AbilityId.BEHAVIOR_CLOAKOFF
    return None


def queued_unit_command(result: Any) -> Optional[Any]:
    """python-sc2 ``unit(ability)`` returns a bool after ``bot.do()``, or a UnitCommand.

    ``client.actions`` only accepts objects with ``combining_tuple``. ``False``
    means the order was rejected; ``True`` means it is already on ``bot.actions``.
    """
    if result is False or result is None:
        return None
    if hasattr(result, "combining_tuple"):
        return result
    return None
