"""Speed-mining worker control (Ares-style boost, no AresBot dependency).

Ports the competitive move-then-SMART pattern used by Ares-sc2 SpeedMining /
Mining(mineral_boost=True), plus sticky mineral-patch assignment so workers
do not thrash between nearest patches every step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.runtime.mining_controller import (
    MiningController,
    _gas_targets,
    _hatch_anchor,
    _local_minerals,
)

if TYPE_CHECKING:
    from sc2.bot_ai import BotAI
    from sc2.unit import Unit

TOWNHALL_RADIUS = 2.75
TOWNHALL_RETURN_FACTOR = 1.08
# Squared-distance band for issuing the boost move (Ares uses 0.5625..4.0).
BOOST_DIST_SQ_MIN = 0.5625
BOOST_DIST_SQ_MAX = 4.0
WORKER_RADIUS = 0.375
MAX_WORKERS_PER_PATCH = 2


def _towards(origin: Point2, target: Point2, distance: float) -> Point2:
    """Point ``distance`` from ``origin`` toward ``target``."""
    return origin.towards(target, distance)


def _dist_sq(a: Point2, b: Point2) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def _mineral_boost_point(mineral: "Unit", hatch: Point2) -> Point2:
    """Stand-off point in front of a mineral patch (toward the hatch)."""
    radius = float(getattr(mineral, "radius", 1.125) or 1.125) + WORKER_RADIUS
    return _towards(mineral.position, hatch, radius)


def _townhall_boost_point(townhall: "Unit", worker_pos: Point2) -> Point2:
    return _towards(
        townhall.position,
        worker_pos,
        TOWNHALL_RADIUS * TOWNHALL_RETURN_FACTOR,
    )


class SpeedMiningController(MiningController):
    """
    Sticky patch assignment + mineral/return speed boost.

    Compatible with ZergBuildBot ``MINE`` tags and ``_mine_anchor_pos``.
    Does not depend on ares-sc2 / cython-extensions-sc2.
    """

    def __init__(self) -> None:
        # worker_tag -> mineral_tag
        self._worker_to_patch: Dict[int, int] = {}
        # mineral_tag -> set(worker_tag)
        self._patch_to_workers: Dict[int, Set[int]] = {}

    def clear(self) -> None:
        self._worker_to_patch.clear()
        self._patch_to_workers.clear()

    async def step(self, bot: "BotAI", worker_tags: Set[int]) -> None:
        if not worker_tags:
            return
        workers = [u for u in bot.workers if int(u.tag) in worker_tags]
        if not workers:
            return

        minerals = _local_minerals(bot)
        if not minerals and not bot.townhalls:
            return

        self._prune_assignments(workers, minerals)
        self._assign_unassigned(workers, minerals)

        townhall = self._resolve_townhall(bot)
        if townhall is None:
            return

        mineral_by_tag = {int(m.tag): m for m in minerals}

        for worker in workers:
            self._step_worker(bot, worker, townhall, mineral_by_tag)

        # Optional gas fill when extractors exist and minerals are saturated.
        await self._maybe_fill_gas(bot, worker_tags)

    def _resolve_townhall(self, bot: "BotAI") -> Optional["Unit"]:
        anchor = _hatch_anchor(bot)
        if bot.townhalls and bot.townhalls.exists:
            if anchor is not None:
                return bot.townhalls.closest_to(anchor)
            return bot.townhalls.first
        return None

    def _prune_assignments(self, workers: List["Unit"], minerals: List["Unit"]) -> None:
        live_workers = {int(w.tag) for w in workers}
        live_minerals = {int(m.tag) for m in minerals}
        for wtag in list(self._worker_to_patch.keys()):
            mtag = self._worker_to_patch[wtag]
            if wtag not in live_workers or mtag not in live_minerals:
                self._unassign(wtag)

    def _unassign(self, worker_tag: int) -> None:
        mtag = self._worker_to_patch.pop(worker_tag, None)
        if mtag is None:
            return
        bucket = self._patch_to_workers.get(mtag)
        if bucket is not None:
            bucket.discard(worker_tag)
            if not bucket:
                self._patch_to_workers.pop(mtag, None)

    def _assign_unassigned(
        self, workers: List["Unit"], minerals: List["Unit"]
    ) -> None:
        if not minerals:
            return
        # Sort patches closest-to-hatch first for better close-patch saturation.
        # Distance key uses first worker as proxy if no hatch; callers pass local mins.
        for worker in workers:
            wtag = int(worker.tag)
            if wtag in self._worker_to_patch:
                continue
            patch = self._pick_patch_for_worker(worker, minerals)
            if patch is None:
                continue
            mtag = int(patch.tag)
            self._worker_to_patch[wtag] = mtag
            self._patch_to_workers.setdefault(mtag, set()).add(wtag)

    def _pick_patch_for_worker(
        self, worker: "Unit", minerals: List["Unit"]
    ) -> Optional["Unit"]:
        # Prefer under-saturated patches (0 then 1 worker), closest to worker.
        def load(m: "Unit") -> int:
            return len(self._patch_to_workers.get(int(m.tag), ()))

        under = [m for m in minerals if load(m) < MAX_WORKERS_PER_PATCH]
        pool = under if under else minerals
        if not pool:
            return None
        # Among under-saturated, fill empties before singles.
        empties = [m for m in pool if load(m) == 0]
        if empties:
            pool = empties
        return min(pool, key=lambda m: worker.distance_to(m.position))

    def _step_worker(
        self,
        bot: "BotAI",
        worker: "Unit",
        townhall: "Unit",
        mineral_by_tag: Dict[int, "Unit"],
    ) -> None:
        orders = getattr(worker, "orders", None) or []
        n_orders = len(orders)
        # Already speed-mining (move + queued SMART).
        if n_orders >= 2:
            return

        wtag = int(worker.tag)
        mtag = self._worker_to_patch.get(wtag)
        mineral = mineral_by_tag.get(mtag) if mtag is not None else None

        carrying = bool(
            getattr(worker, "is_carrying_minerals", False)
            or getattr(worker, "is_carrying_vespene", False)
        )
        returning = bool(getattr(worker, "is_returning", False)) or carrying

        if returning and n_orders < 2:
            self._boost_return(bot, worker, townhall)
            return

        if mineral is None:
            if worker.is_idle and bot.mineral_field:
                nearest = min(bot.mineral_field, key=lambda m: worker.distance_to(m))
                bot.do(worker.gather(nearest))
            return

        if n_orders < 2 and not returning:
            self._boost_gather(bot, worker, mineral, townhall)
            return

        if worker.is_idle:
            bot.do(worker.gather(mineral))

    def _boost_return(self, bot: "BotAI", worker: "Unit", townhall: "Unit") -> None:
        target = _townhall_boost_point(townhall, worker.position)
        dist_sq = _dist_sq(worker.position, target)
        if BOOST_DIST_SQ_MIN < dist_sq < BOOST_DIST_SQ_MAX:
            bot.do(worker.move(target))
            bot.do(worker(AbilityId.SMART, townhall, queue=True))
        elif not getattr(worker, "is_returning", False):
            bot.do(worker(AbilityId.SMART, townhall))

    def _boost_gather(
        self,
        bot: "BotAI",
        worker: "Unit",
        mineral: "Unit",
        townhall: "Unit",
    ) -> None:
        boost_pos = _mineral_boost_point(mineral, townhall.position)
        dist_sq = _dist_sq(worker.position, boost_pos)
        if worker.is_idle or (BOOST_DIST_SQ_MIN < dist_sq < BOOST_DIST_SQ_MAX):
            bot.do(worker.move(boost_pos))
            bot.do(worker(AbilityId.SMART, mineral, queue=True))
        elif dist_sq > BOOST_DIST_SQ_MAX and getattr(worker, "is_gathering", False):
            bot.do(worker.move(boost_pos))
        elif worker.order_target != mineral.tag and not getattr(
            worker, "is_carrying_resource", False
        ):
            bot.do(worker.gather(mineral))

    async def _maybe_fill_gas(self, bot: "BotAI", worker_tags: Set[int]) -> None:
        """Send surplus miners to gas only when extractors exist (Zerg)."""
        ready_extractors = bot.structures(UnitTypeId.EXTRACTOR).ready
        if not ready_extractors.exists:
            return
        gas_targets = list(ready_extractors)
        # Keep it simple: at most 3 per extractor from mining-role idle workers.
        for extractor in gas_targets:
            assigned = sum(
                1
                for w in bot.workers
                if w.order_target == extractor.tag
                or (
                    getattr(w, "is_carrying_vespene", False)
                    and w.order_target == getattr(bot.townhalls.first, "tag", None)
                )
            )
            deficit = max(0, 3 - assigned)
            if deficit <= 0:
                continue
            candidates = [
                w
                for w in bot.workers
                if int(w.tag) in worker_tags
                and w.is_idle
                and int(w.tag) in self._worker_to_patch
            ]
            # Prefer workers on oversaturated patches (3rd worker) — we only assign 2,
            # so use furthest mineral workers as surplus.
            candidates.sort(
                key=lambda w: w.distance_to(extractor),
                reverse=True,
            )
            for worker in candidates[:deficit]:
                self._unassign(int(worker.tag))
                bot.do(worker.gather(extractor))
