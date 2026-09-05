"""CTF ``env_mode: spellcaster`` / heal — max energy and spell-tech research.

``apply_max_energy`` is also used by heal mode so Queens spawn with enough
energy to Transfuse (50) instead of the default 25.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

RefreshState = Callable[[], Awaitable[None]]

# Same ladder length as cloak/drop: enough ``debug_upgrade`` calls to unlock
# Psi Storm and other ability techs (SC2 has no selective upgrade cheat).
SPELLCASTER_UPGRADE_CALLS = 20
# debug_set_unit_value unit_value=1 is energy.
_ENERGY_UNIT_VALUE = 1


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

    Returns how many ``debug_upgrade`` calls were made this invocation.
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
