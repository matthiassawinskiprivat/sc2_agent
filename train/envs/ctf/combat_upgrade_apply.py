"""Progressive ``debug_upgrade`` schedule for CTF ``env_mode: upgrades``.

SC2 has no selective per-upgrade debug grant. ``Client.debug_upgrade()`` unlocks the
next tier of available research (weapons/armor and unit abilities) for the player
whose client sends it. God-mode joins only player 1, so that player receives it.

Within each hard-reset window of ``every_k_episodes``:
  - before 25% → 0 calls
  - at/after 25% → 1 call total
  - at/after 50% → 2 calls total
  - at/after 75% → 3 calls total
Hard reset starts a new game (tier back to 0). Soft resets keep researched upgrades.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

RefreshState = Callable[[], Awaitable[None]]


def desired_debug_upgrade_calls(episode_counter: int, every_k: int) -> int:
    """How many ``debug_upgrade`` calls should have happened in this hard-reset window."""
    k = max(1, int(every_k))
    pos = int(episode_counter) % k
    if pos == 0:
        # Episode every_k, 2*every_k, … → fresh game after hard reset.
        return 0
    t1, t2, t3 = k // 4, k // 2, (3 * k) // 4
    if pos >= t3:
        return 3
    if pos >= t2:
        return 2
    if pos >= t1:
        return 1
    return 0


async def apply_scheduled_debug_upgrades(
    *,
    bot: Any,
    episode_counter: int,
    every_k: int,
    already_applied: int,
    refresh_state: RefreshState,
) -> int:
    """Call ``debug_upgrade`` until ``already_applied`` matches the schedule.

    Returns the new applied count (0–3).
    """
    if bot is None:
        return int(already_applied)
    desired = desired_debug_upgrade_calls(episode_counter, every_k)
    applied = max(0, min(3, int(already_applied)))
    if desired <= applied:
        return applied
    client = getattr(bot, "client", None)
    if client is None:
        return applied
    for i in range(applied, desired):
        try:
            await client.debug_upgrade()
            await client.step()
            await refresh_state()
            applied = i + 1
            logger.info(
                f"upgrades mode: debug_upgrade #{applied}/3 "
                f"(episode={episode_counter}, every_k={every_k})"
            )
        except Exception as exc:
            logger.warning(f"debug_upgrade #{i + 1} failed: {exc}")
            break
    return applied
