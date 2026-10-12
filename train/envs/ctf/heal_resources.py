"""CTF mineral/vespene grant for player 1."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Set

logger = logging.getLogger(__name__)

# Ensure at least this much of each resource for player 1.
# ``debug_all_resources`` adds 5000 minerals + 5000 vespene (``debug_gas`` is a no-op).
CTF_RESOURCE_FLOOR = 100
# Backward-compatible alias (heal-mode Terran repair used the same floor).
HEAL_TERRAN_RESOURCE_FLOOR = CTF_RESOURCE_FLOOR


def terran_player_ids(
    unit_pools_config: Iterable[dict] | None,
    pool_player_id: Callable[[dict], int],
) -> Set[int]:
    """Player ids that have at least one Terran unit pool."""
    return {
        int(pool_player_id(pool))
        for pool in (unit_pools_config or [])
        if str(pool.get("race", "")).lower() == "terran"
    }


async def apply_ctf_resources(
    *,
    bot: Any,
    free_enabled: bool,
) -> bool:
    """Grant minerals and vespene to player 1.

    Returns ``free_enabled`` unchanged. ``debug_free`` is not used.

    That cheat is documented as waiving mineral and gas, but on the current
    client it also waives ability energy. A Queen set to 100 energy then
    Transfuses for free and the bar never falls, so spell timing cannot be
    learned. Player 1 gets a real bank from ``debug_all_resources`` instead.
    There is no per-player resource cheat for player 2, and ``debug_control_enemy``
    cannot spend player 2's bank, so opponent repair and morph costs are not
    waived.

    ``debug_gas`` alone does not work in python-sc2 / SC2. Top up with
    ``debug_all_resources`` when player 1 minerals *or* vespene is below
    ``CTF_RESOURCE_FLOOR`` (100).
    """
    if bot is None:
        return free_enabled

    client = getattr(bot, "client", None)
    if client is None:
        return free_enabled

    try:
        minerals = int(getattr(bot, "minerals", 0) or 0)
        vespene = int(getattr(bot, "vespene", 0) or 0)
        if minerals < CTF_RESOURCE_FLOOR or vespene < CTF_RESOURCE_FLOOR:
            await client.debug_all_resources()
    except Exception as exc:
        logger.warning(f"CTF resource grant failed: {exc}")
        return free_enabled

    step = getattr(client, "step", None)
    if step is not None:
        await step()
    return free_enabled
