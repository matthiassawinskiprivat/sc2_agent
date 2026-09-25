"""CTF ``env_mode: spellcaster`` / heal — max energy and spell-tech research.

``apply_max_energy`` is also used by heal mode so Queens spawn with enough
energy to Transfuse (50) instead of the default 25.

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

# AoE spell these casters use, and the upgrade that gates it.
# ``None`` means the spell is available without research.
# Ghost cloak and Infestor Neural Parasite are separate researched abilities.
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


async def apply_max_energy(bot: Any) -> int:
    """Set every energy unit to its ``energy_max``. Returns how many units were set."""
    if bot is None:
        return 0
    client = getattr(bot, "client", None)
    if client is None:
        return 0
    grouped: Dict[float, List[int]] = defaultdict(list)
    for tag, emax in energy_units(bot):
        grouped[emax].append(tag)
    filled = 0
    setter = getattr(client, "debug_set_unit_value", None)
    if setter is None:
        return 0
    for emax, tags in grouped.items():
        try:
            await setter(tags, _ENERGY_UNIT_VALUE, float(emax))
            filled += len(tags)
        except Exception as exc:
            logger.warning(f"spellcaster max energy failed: {exc}")
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
