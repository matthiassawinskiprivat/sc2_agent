"""Drop-mode free transports (Medivac / Warp Prism / Overlord Transport).

The budgeted army still comes from each squad's ``unit_pool``. When
``env_mode`` contains ``drop``, one race transport is added on top and is
not charged against ``mineral_budget`` / ``vespene_budget``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Sequence

RefreshState = Callable[[], Awaitable[None]]

# Direct create. Overlord→Ventral Sacs morph is unreliable on the debug path
# (same reason cloak mode creates Overseer instead of morphing an Overlord).
DROP_TRANSPORT_BY_RACE = {
    "terran": "MEDIVAC",
    "protoss": "WARPPRISM",
    "zerg": "OVERLORDTRANSPORT",
}


def drop_transport_name(race: str) -> Optional[str]:
    """Unit name of the free transport for ``race``, or None if the race has none."""
    return DROP_TRANSPORT_BY_RACE.get(str(race).lower())


async def spawn_drop_transports(
    *,
    bot: Any,
    transport_cmds: Sequence[list],
    refresh_state: RefreshState,
) -> None:
    """Spawn free drop transports via ``debug_create_unit`` (all races).

    ``transport_cmds`` entries are ``[unit_type, count, pos, player_id]``.
    """
    if bot is None or not transport_cmds:
        return
    # One create per owner — mixed-owner lists can silently drop trailing cmds.
    by_player: dict[int, list] = {}
    for cmd in transport_cmds:
        pid = int(cmd[3]) if len(cmd) > 3 else 1
        by_player.setdefault(pid, []).append(cmd)
    for pid in sorted(by_player):
        await bot.client.debug_create_unit(by_player[pid])
    await bot.client.step()
    await refresh_state()
