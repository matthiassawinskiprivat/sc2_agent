"""CTF ``env_mode: spellcaster`` / heal — starting energy and spell-tech research.

``apply_caster_start_energy`` gives each caster two uses of its main energy
spell (one use when two would fill the bar). Nothing is set to ``energy_max``.
A full bar stalls the fight and never forces a choice about when to spend it.

``debug_upgrade`` grants researched abilities such as Psi Storm without a
Templar Archives. The cheat reaches only the player whose client sends it, and
the god-mode match joins player 1. Protoss must be the first race in the level
so High Templars are that player. Ghost EMP and Infestor Fungal Growth need no
research.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

RefreshState = Callable[[], Awaitable[None]]

# Same ladder length as cloak/drop: enough ``debug_upgrade`` calls to unlock
# Psi Storm and other ability techs (SC2 has no selective upgrade cheat).
SPELLCASTER_UPGRADE_CALLS = 20
# debug_set_unit_value unit_value=1 is energy.
_ENERGY_UNIT_VALUE = 1
# Transfusion costs 50. Two casts, not the 200 cap (four casts stalls a fight).
QUEEN_HEAL_START_ENERGY = 100.0

# AoE spell these casters use, and the upgrade that gates it.
# ``None`` means the spell is available without research.
# Ghost cloak and Infestor Neural Parasite are separate researched abilities.
# Neural is unlocked by the debug_upgrade ladder (same as Psi Storm).
# Unit-target specials: Ghost Steady Targeting and Feedback need no research.
# Neural Parasite and Yamato (BattlecruiserEnableSpecializations; older
# clients called it YAMATOCANNON) are unlocked by the debug_upgrade ladder.
SPELLCASTER_CAST_UPGRADES: Dict[str, Optional[str]] = {
    "HIGHTEMPLAR": "PSISTORMTECH",
    "GHOST": None,
    "INFESTOR": None,
}

# Race → unit added to post-AoE pools so the AoE command still has a caster.
RACE_SPELLCASTER_UNIT: Dict[str, str] = {
    "protoss": "HIGHTEMPLAR",
    "terran": "GHOST",
    "zerg": "INFESTOR",
}


def energy_units(bot: Any) -> List[Tuple[int, float]]:
    """``(tag, energy_max)`` for live units that have an energy bar."""
    units = getattr(bot, "all_units", None)
    if units is None:
        return []
    try:
        iterable: Iterable = list(units)
    except TypeError:
        return []
    out: List[Tuple[int, float]] = []
    for unit in iterable:
        tag = int(getattr(unit, "tag", 0) or 0)
        emax = float(getattr(unit, "energy_max", 0.0) or 0.0)
        if tag > 0 and emax > 0.0:
            out.append((tag, emax))
    return out


def caster_start_energy_table() -> Dict[Any, float]:
    """Starting energy by type: two casts, or one when two would fill the bar.

    Channel units (Medivac heal, Shield Battery recharge, Banshee cloak drain)
    stay at their natural 50. A full 200 bar is what made every caster feel
    unlimited for the length of a fight.
    """
    from sc2.ids.unit_typeid import UnitTypeId

    table: Dict[Any, float] = {
        UnitTypeId.QUEEN: QUEEN_HEAL_START_ENERGY,  # Transfusion 50
        UnitTypeId.MEDIVAC: 50.0,
        UnitTypeId.SHIELDBATTERY: 50.0,
        UnitTypeId.HIGHTEMPLAR: 150.0,  # Psi Storm 75
        UnitTypeId.GHOST: 150.0,  # EMP 75
        UnitTypeId.INFESTOR: 150.0,  # Fungal Growth 75
        UnitTypeId.VIPER: 125.0,  # Parasitic Bomb 125; two do not fit in 200
        UnitTypeId.BATTLECRUISER: 125.0,  # Yamato 125; two do not fit in 200
        UnitTypeId.SENTRY: 100.0,  # Force Field 50
        UnitTypeId.RAVEN: 150.0,  # Interference Matrix 75
        UnitTypeId.BANSHEE: 50.0,
        UnitTypeId.ORACLE: 100.0,  # Stasis Ward 50
        UnitTypeId.MOTHERSHIP: 100.0,  # Time Warp 100; two fill the 200 bar
        UnitTypeId.OVERSEER: 125.0,  # Contaminate 125; two do not fit in 200
    }
    for name, value in (
        ("QUEENBURROWED", QUEEN_HEAL_START_ENERGY),
        ("INFESTORBURROWED", 150.0),
        ("OVERSEERSIEGEMODE", 125.0),
    ):
        type_id = getattr(UnitTypeId, name, None)
        if type_id is not None:
            table[type_id] = value
    return table


def caster_start_energy_groups(bot: Any) -> Dict[float, List[int]]:
    """``{energy: [tags]}`` for casters in ``caster_start_energy_table``.

    The value is capped at that unit's ``energy_max``. Units with no energy
    bar, a bad tag, or a type that is not in the table are left alone.
    """
    table = caster_start_energy_table()
    units = getattr(bot, "all_units", None)
    if units is None:
        return {}
    try:
        iterable: Iterable = list(units)
    except TypeError:
        return {}
    grouped: Dict[float, List[int]] = defaultdict(list)
    for unit in iterable:
        budget = table.get(getattr(unit, "type_id", None))
        if budget is None:
            continue
        tag = int(getattr(unit, "tag", 0) or 0)
        emax = float(getattr(unit, "energy_max", 0.0) or 0.0)
        if tag <= 0 or emax <= 0.0:
            continue
        grouped[min(float(budget), emax)].append(tag)
    return grouped


async def apply_caster_start_energy(bot: Any) -> int:
    """Set each known caster to its limited starting energy. Returns how many were set."""
    if bot is None:
        return 0
    client = getattr(bot, "client", None)
    if client is None:
        return 0
    setter = getattr(client, "debug_set_unit_value", None)
    if setter is None:
        return 0
    filled = 0
    for value, tags in caster_start_energy_groups(bot).items():
        try:
            await setter(tags, _ENERGY_UNIT_VALUE, float(value))
            filled += len(tags)
        except Exception as exc:
            logger.warning(f"caster start energy failed: {exc}")
    return filled


async def apply_spell_upgrades(
    *,
    bot: Any,
    already_full_ladder: bool,
    refresh_state: RefreshState,
) -> int:
    """Run the debug_upgrade ladder unless cloak already completed it.

    The cheat applies to player 1 only. Returns how many calls completed.
    """
    if already_full_ladder or bot is None:
        return 0
    client = getattr(bot, "client", None)
    if client is None:
        return 0
    applied = 0
    for _ in range(SPELLCASTER_UPGRADE_CALLS):
        try:
            await client.debug_upgrade()
            step = getattr(client, "step", None)
            if step is not None:
                await step()
            await refresh_state()
            applied += 1
        except Exception as exc:
            logger.warning(f"spellcaster debug_upgrade failed: {exc}")
            break
    return applied
