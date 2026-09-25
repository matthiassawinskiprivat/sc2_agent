"""CTF ``env_mode: heal`` — a few burrowed tumors so Queens can Transfuse.

StarCraft II has no debug command to paint source-less creep. Unfinished
``CREEPTUMOR`` also takes ~11s to morph before it spreads.

Burrowed tumors generate creep from their cell outward (~10 radius when
filled). One tumor at spawn therefore still *looks* like a slow spread.
A compact ring around each spawn site puts every nearby cell next to a
source so the army starts on creep without waiting:

- Zerg **defend**: defending spawn cluster + midpoint toward the attacker
- Zerg **attack**: attack spawn cluster + assigned defender cluster + midpoint
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

logger = logging.getLogger(__name__)

# Env-planted tumors, plus the morphing forms a Queen leaves behind.
HEAL_TUMOR_TYPE_IDS = frozenset(
    {
        UnitTypeId.CREEPTUMOR,
        UnitTypeId.CREEPTUMORBURROWED,
        UnitTypeId.CREEPTUMORQUEEN,
    }
)

_CREATE_BATCH = 40
_XY = Tuple[float, float]

# Army spawn ring is dist=3. Cluster radius 5 sits just outside that blob.
SPAWN_CLUSTER_RADIUS = 5.0
# Unit vectors: 4 cardinals + 4 diagonals (distance == radius).
_CLUSTER_DIRS: Tuple[_XY, ...] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.7071067811865476, 0.7071067811865476),
    (0.7071067811865476, -0.7071067811865476),
    (-0.7071067811865476, 0.7071067811865476),
    (-0.7071067811865476, -0.7071067811865476),
)


def spawn_tumor_cluster(
    center: _XY,
    radius: float = SPAWN_CLUSTER_RADIUS,
) -> List[_XY]:
    """Center plus an 8-point ring so spawn-adjacent cells have a nearby source."""
    x, y = center
    pts: List[_XY] = [(x, y)]
    r = float(radius)
    for dx, dy in _CLUSTER_DIRS:
        pts.append((x + r * dx, y + r * dy))
    return pts


def _spawn_xy(spawn_center: Any) -> Optional[_XY]:
    try:
        return (float(spawn_center.x), float(spawn_center.y))
    except (AttributeError, TypeError, ValueError):
        return None


def _centers_by_label(pool_spawn_centers: Sequence[Any]) -> Dict[Any, Dict[str, _XY]]:
    by_label: Dict[Any, Dict[str, _XY]] = {}
    for row in pool_spawn_centers or []:
        if len(row) < 4:
            continue
        _player_id, command, spawn_center, label = row[0], row[1], row[2], row[3]
        xy = _spawn_xy(spawn_center)
        if xy is None or label is None:
            continue
        by_label.setdefault(label, {})[str(command).lower()] = xy
    return by_label


def zerg_roles_by_label(unit_pools: Iterable[dict] | None) -> Dict[Any, Set[str]]:
    """Arena label → Zerg command roles in that arena (``attack`` / ``defend``)."""
    out: Dict[Any, Set[str]] = {}
    for pool in unit_pools or []:
        if str(pool.get("race", "")).lower() != "zerg":
            continue
        label = pool.get("region")
        if label is None:
            continue
        cmd = str(pool.get("high_level_command", "attack")).lower()
        out.setdefault(label, set()).add(cmd)
    return out


def zerg_arena_tumor_positions(
    pool_spawn_centers: Sequence[Any],
    unit_pools: Iterable[dict] | None,
) -> List[Point2]:
    """Burrowed-tumor sites for each Zerg arena (deduped).

    Spawn sites get a center+ring cluster so creep does not have to crawl
    out from a single tumor. The corridor midpoint stays a single source.
    """
    centers = _centers_by_label(pool_spawn_centers)
    roles = zerg_roles_by_label(unit_pools)
    seen: Set[Tuple[float, float]] = set()
    points: List[Point2] = []

    def add(xy: Optional[_XY]) -> None:
        if xy is None:
            return
        key = (round(xy[0], 3), round(xy[1], 3))
        if key in seen:
            return
        seen.add(key)
        points.append(Point2(xy))

    def add_spawn_cluster(xy: Optional[_XY]) -> None:
        if xy is None:
            return
        for pt in spawn_tumor_cluster(xy):
            add(pt)

    for label, zerg_cmds in roles.items():
        sides = centers.get(label, {})
        attack = sides.get("attack")
        defend = sides.get("defend")
        midpoint = None
        if attack is not None and defend is not None:
            midpoint = (
                (attack[0] + defend[0]) / 2.0,
                (attack[1] + defend[1]) / 2.0,
            )
        if "defend" in zerg_cmds:
            add_spawn_cluster(defend)
            add(midpoint)
        if "attack" in zerg_cmds:
            add_spawn_cluster(attack)
            add_spawn_cluster(defend)
            add(midpoint)
    return points


def creep_source_create_cmds(
    positions: Sequence[Point2],
    player_id: int,
) -> List[list]:
    """``debug_create_unit`` rows: finished tumors, not the 11s morphing type."""
    pid = int(player_id)
    return [
        [UnitTypeId.CREEPTUMORBURROWED, 1, pos, pid] for pos in positions
    ]


def _create_batches(cmds: List[list], batch_size: int = _CREATE_BATCH) -> List[List[list]]:
    if batch_size <= 0:
        return [cmds] if cmds else []
    return [cmds[i : i + batch_size] for i in range(0, len(cmds), batch_size)]


async def apply_heal_map_creep(
    *,
    bot: Any,
    player_id: int,
    positions: Sequence[Point2] | None = None,
) -> int:
    """Plant the given burrowed tumors. Returns how many create cmds were issued."""
    if bot is None or not positions:
        return 0
    client = getattr(bot, "client", None)
    if client is None:
        return 0
    cmds = creep_source_create_cmds(positions, player_id)
    if not cmds:
        return 0
    created = 0
    setter = getattr(client, "debug_create_unit", None)
    if setter is None:
        return 0
    try:
        for batch in _create_batches(cmds):
            await setter(batch)
            created += len(batch)
    except Exception as exc:
        logger.warning(f"heal map creep create failed: {exc}")
        return created
    return created
