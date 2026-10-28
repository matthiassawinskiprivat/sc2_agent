"""Budget-greedy CTF army composition with per-type spawn caps."""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

logger = logging.getLogger(__name__)

# Soft uniqueness caps when spending mineral/vespene budgets.
# One Mothership. At most two of each backline caster (Ghost, High Templar,
# Infestor, Viper) and at most two Medivacs, so a budget cannot fill the squad
# with casters or transports.
DEFAULT_UNIT_SPAWN_CAPS: Dict[str, int] = {
    "MOTHERSHIP": 1,
    "GHOST": 2,
    "HIGHTEMPLAR": 2,
    "INFESTOR": 2,
    "VIPER": 2,
    "MEDIVAC": 2,
}


def calculate_budgeted_spawns(
    unit_pool: Sequence[str],
    unit_data: Mapping[str, dict],
    minerals: float,
    vespene: float,
    *,
    caps: Optional[Mapping[str, int]] = None,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Greedily sample units until the budget is exhausted.

    ``caps`` limits how many of each type may appear (default: one MOTHERSHIP).
    Unknown unit names are skipped with an error log.
    """
    choose = rng.choice if rng is not None else random.choice
    spawn_caps = dict(caps) if caps is not None else dict(DEFAULT_UNIT_SPAWN_CAPS)
    counts: MutableMapping[str, int] = {}
    m = float(minerals)
    v = float(vespene)
    pool = list(unit_pool)
    spawns: List[str] = []

    def can_afford(u: str) -> bool:
        if u not in unit_data:
            logger.error("Unit %s not found in unit_data.json. Skipping.", u)
            return False
        key = str(u).upper()
        cap = int(spawn_caps.get(key, spawn_caps.get(u, 10**9)))
        if counts.get(key, 0) >= cap:
            return False
        cost = unit_data[u]["cost"]
        return cost["minerals"] <= m and cost["vespene"] <= v

    affordable = [u for u in pool if can_afford(u)]
    while affordable:
        u = choose(affordable)
        spawns.append(u)
        key = str(u).upper()
        counts[key] = counts.get(key, 0) + 1
        cost = unit_data[u]["cost"]
        m -= cost["minerals"]
        v -= cost["vespene"]
        affordable = [u for u in pool if can_afford(u)]
    return spawns
