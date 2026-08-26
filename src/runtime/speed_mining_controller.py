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
    _gas_worker_quota,
    _hatch_anchor,
    _local_minerals,
    _order_target_tag,
    _worker_is_building,
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

    WORKERS_PER_EXTRACTOR = 3

    def __init__(self) -> None:
        # worker_tag -> mineral_tag
        self._worker_to_patch: Dict[int, int] = {}
        # mineral_tag -> set(worker_tag)
        self._patch_to_workers: Dict[int, Set[int]] = {}
        # Workers reserved for ready extractors (must not be reassigned to minerals).
        self._gas_workers: Set[int] = set()

    def clear(self) -> None:
        self._worker_to_patch.clear()
        self._patch_to_workers.clear()
        self._gas_workers.clear()

    async def step(self, bot: "BotAI", worker_tags: Set[int]) -> None:
        if not worker_tags:
            return
        workers = [u for u in bot.workers if int(u.tag) in worker_tags]
        if not workers:
            return

        minerals = _local_minerals(bot)
        ready_extractors = self._ready_extractors(bot)
        live_tags = {int(w.tag) for w in workers}
        self._gas_workers &= live_tags
        # Drop gas reservation if extractors are gone.
        if not ready_extractors:
            self._gas_workers.clear()

        if not minerals and not bot.townhalls and not ready_extractors:
            return

        townhall = self._resolve_townhall(bot)
        if townhall is None:
            return

        # Fill / trim gas FIRST so pulled drones can receive mineral patches below.
        self._fill_gas(bot, worker_tags, ready_extractors)

        mineral_workers = [w for w in workers if int(w.tag) not in self._gas_workers]
        self._prune_assignments(mineral_workers, minerals)
        self._assign_unassigned(mineral_workers, minerals)

        mineral_by_tag = {int(m.tag): m for m in minerals}

        for worker in workers:
            if int(worker.tag) in self._gas_workers:
                self._step_gas_worker(bot, worker, townhall, ready_extractors)
            else:
                self._step_worker(
                    bot, worker, townhall, mineral_by_tag, ready_extractors
                )

    @staticmethod
    def _ready_extractors(bot: "BotAI") -> List["Unit"]:
        extractors = bot.structures(UnitTypeId.EXTRACTOR)
        ready = getattr(extractors, "ready", extractors)
        if ready is None:
            return []
        # Explicit False only — MagicMock.exists must not count as ready.
        if getattr(ready, "exists", True) is False:
            return []
        try:
            units = list(ready)
        except TypeError:
            return []
        # Only extractors at the current episode hatch — ignore leftovers across the map.
        anchor = _hatch_anchor(bot) or getattr(bot, "_spawn_hatchery_pos", None)
        if anchor is None:
            return units
        local = []
        for ext in units:
            try:
                if float(ext.distance_to(anchor)) <= 20.0:
                    local.append(ext)
            except (TypeError, ValueError):
                local.append(ext)
        return local

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
        for worker in workers:
            wtag = int(worker.tag)
            if wtag in self._worker_to_patch or wtag in self._gas_workers:
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
        extractors: Optional[List["Unit"]] = None,
    ) -> None:
        # Never cancel an in-flight structure build with gather/boost orders.
        if _worker_is_building(worker):
            return

        # Steal non-reserved drones off extractors even if they still show a
        # multi-order queue from a previous frame (obs not updated yet).
        extractors = extractors or []
        if extractors and self._worker_targets_extractor(worker, extractors):
            minerals = list(mineral_by_tag.values()) or _local_minerals(bot)
            if minerals:
                nearest = min(minerals, key=lambda m: worker.distance_to(m))
                bot.do(worker.gather(nearest))
            return

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

    @staticmethod
    def _extractor_tags(extractors: List["Unit"]) -> Set[int]:
        return {int(e.tag) for e in extractors}

    @classmethod
    def _worker_targets_extractor(
        cls, worker: "Unit", extractors: List["Unit"]
    ) -> bool:
        if not extractors:
            return False
        tags = cls._extractor_tags(extractors)
        ot = _order_target_tag(worker)
        if ot is not None and ot in tags:
            return True
        # Fallback: gathering while standing on the extractor footprint.
        # Covers frames where order_target is a Point2 / lagging after boost.
        if getattr(worker, "is_gathering", False) is True:
            for ext in extractors:
                try:
                    if float(worker.distance_to(ext)) <= 2.5:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    def _step_gas_worker(
        self,
        bot: "BotAI",
        worker: "Unit",
        townhall: "Unit",
        extractors: List["Unit"],
    ) -> None:
        """Keep reserved gas workers on extractors / returning vespene."""
        if _worker_is_building(worker):
            return
        if not extractors:
            self._gas_workers.discard(int(worker.tag))
            return

        carrying = bool(getattr(worker, "is_carrying_vespene", False))
        orders = getattr(worker, "orders", None) or []
        if carrying or getattr(worker, "is_returning", False):
            if len(orders) < 2:
                self._boost_return(bot, worker, townhall)
            return

        if self._worker_targets_extractor(worker, extractors):
            return

        # Never issue another gather onto a full extractor (stops a 4th drone).
        quota = _gas_worker_quota(bot)
        cap = min(quota, len(extractors) * self.WORKERS_PER_EXTRACTOR) if quota > 0 else 0
        on_ext = sum(
            1 for w in bot.workers if self._worker_targets_extractor(w, extractors)
        )
        if cap <= 0 or on_ext >= cap:
            return

        nearest = min(extractors, key=lambda e: worker.distance_to(e))
        bot.do(worker.gather(nearest))

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

    def _fill_gas(
        self, bot: "BotAI", worker_tags: Set[int], extractors: List["Unit"]
    ) -> None:
        """Match gas workers to the standing MINE_GAS quota (0 => pull off)."""
        quota = _gas_worker_quota(bot)

        def on_gas(worker: "Unit") -> bool:
            if _worker_is_building(worker):
                return False
            return self._worker_targets_extractor(worker, extractors)

        def in_gas_cycle(worker: "Unit") -> bool:
            if on_gas(worker):
                return True
            return bool(getattr(worker, "is_carrying_vespene", False))

        if quota <= 0 or not extractors:
            minerals = _local_minerals(bot)
            # Pull every drone off local extractors (any role), not only mining tags.
            for w in bot.workers:
                if _worker_is_building(w) or not in_gas_cycle(w):
                    continue
                if minerals:
                    nearest = min(minerals, key=lambda m: w.distance_to(m))
                    bot.do(w.gather(nearest))
            self._gas_workers.clear()
            return

        max_slots = len(extractors) * self.WORKERS_PER_EXTRACTOR
        target = min(quota, max_slots)
        prior_reserved = set(self._gas_workers)

        def gas_rank(w: "Unit") -> tuple:
            wtag = int(w.tag)
            if on_gas(w):
                tier = 0
            elif getattr(w, "is_carrying_vespene", False):
                tier = 1
            elif wtag in prior_reserved and not _worker_is_building(w):
                tier = 2
            else:
                tier = 3
            return (
                tier,
                min(w.distance_to(e) for e in extractors),
                wtag,
            )

        # Prefer mining-role workers for seats; still detect ALL drones on gas.
        mining_pool = [
            w
            for w in bot.workers
            if int(w.tag) in worker_tags and not _worker_is_building(w)
        ]
        preferred = [
            w
            for w in mining_pool
            if on_gas(w)
            or getattr(w, "is_carrying_vespene", False)
            or int(w.tag) in prior_reserved
        ]
        preferred.sort(key=gas_rank)

        if len(preferred) < target:
            extras = [
                w
                for w in mining_pool
                if int(w.tag) not in {int(x.tag) for x in preferred}
                and not getattr(w, "is_carrying_minerals", False)
                and not on_gas(w)
            ]

            def surplus_key(w: "Unit") -> tuple:
                mtag = self._worker_to_patch.get(int(w.tag))
                load = (
                    len(self._patch_to_workers.get(mtag, ()))
                    if mtag is not None
                    else 0
                )
                return (-load, min(w.distance_to(e) for e in extractors))

            extras.sort(key=surplus_key)
            preferred = preferred + extras[: target - len(preferred)]

        chosen = preferred[:target]
        chosen_tags = {int(w.tag) for w in chosen}
        self._gas_workers = set(chosen_tags)
        for wtag in self._gas_workers:
            self._unassign(wtag)

        minerals = _local_minerals(bot)
        # Pull surplus off extractors regardless of role / mining tags.
        for w in bot.workers:
            wtag = int(w.tag)
            if wtag in chosen_tags or _worker_is_building(w):
                continue
            if not in_gas_cycle(w) and wtag not in prior_reserved:
                continue
            if minerals:
                nearest = min(minerals, key=lambda m: w.distance_to(m))
                bot.do(w.gather(nearest))

        for w in chosen:
            if getattr(w, "is_carrying_vespene", False):
                continue
            if on_gas(w):
                continue
            nearest = min(extractors, key=lambda e: w.distance_to(e))
            bot.do(w.gather(nearest))
