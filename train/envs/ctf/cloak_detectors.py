"""Cloak-mode free detectors (Observer / Raven / Overseer)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Sequence

RefreshState = Callable[[], Awaitable[None]]


async def spawn_cloak_detectors(
    *,
    bot: Any,
    detector_cmds: Sequence[list],
    refresh_state: RefreshState,
) -> None:
    """Spawn cloak detectors via ``debug_create_unit`` (all races).

    ``detector_cmds`` entries are ``[unit_type, count, pos, player_id]``.
    Overseer is created directly (same as Raven/Observer); Overlord→morph is
    unreliable in this debug spawn path and is intentionally not used.
    """
    if bot is None or not detector_cmds:
        return
    # One create per owner — mixed-owner lists can silently drop trailing cmds.
    by_player: dict[int, list] = {}
    for cmd in detector_cmds:
        pid = int(cmd[3]) if len(cmd) > 3 else 1
        by_player.setdefault(pid, []).append(cmd)
    for pid in sorted(by_player):
        await bot.client.debug_create_unit(by_player[pid])
    await bot.client.step()
    await refresh_state()
