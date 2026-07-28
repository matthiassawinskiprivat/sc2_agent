from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from sc2.bot_ai import BotAI
    from sc2.position import Point2


class MiningController(ABC):
    """Pluggable worker mining — scripted today, learned policy later."""

    @abstractmethod
    async def step(self, bot: "BotAI", worker_tags: Set[int]) -> None:
        """Issue gather/mineral orders for workers in the mining role."""


class NoOpMiningController(MiningController):
    """Workers stay idle except when pulled by build or micro agents."""

    async def step(self, bot: "BotAI", worker_tags: Set[int]) -> None:
        return


def _mineral_targets(bot: "BotAI") -> List:
    if bot.mineral_field:
        return list(bot.mineral_field)
    return [u for u in bot.all_units if getattr(u, "is_mineral_field", False)]


def _gas_targets(bot: "BotAI") -> List:
    if bot.vespene_geyser:
        return list(bot.vespene_geyser)
    return [u for u in bot.all_units if getattr(u, "is_vespene_geyser", False)]


def _hatch_anchor(bot: "BotAI") -> Optional["Point2"]:
    """Prefer manager MINE region hatch, then spawn hatch, then first townhall."""
    mine_anchor = getattr(bot, "_mine_anchor_pos", None)
    if mine_anchor is not None:
        return mine_anchor
    if bot.townhalls and bot.townhalls.exists:
        return bot.townhalls.first.position
    return getattr(bot, "_spawn_hatchery_pos", None)


def _local_minerals(bot: "BotAI", radius: float = 15.0) -> List:
    """Prefer mineral patches near the active hatch so drones stay on the home base."""
    minerals = _mineral_targets(bot)
    anchor = _hatch_anchor(bot)
    if anchor is None or not minerals:
        return minerals
    local = [m for m in minerals if m.distance_to(anchor) <= radius]
    return local if local else minerals


def _worker_needs_mining_order(worker) -> bool:
    if worker.is_carrying_minerals or worker.is_carrying_vespene:
        return False
    if worker.is_idle:
        return True
    orders = getattr(worker, "orders", None)
    if orders is not None and len(orders) == 0:
        return True
    if not worker.is_gathering:
        return True
    return False


class ScriptedMiningController(MiningController):
    """
    Simple mineral/gas assignment using burnysc2 (python-sc2) gather orders.

    Orders go through ``bot.do`` so they flush with the build env action path.
    Raw geysers are skipped until a ready extractor exists.
    """

    async def step(self, bot: "BotAI", worker_tags: Set[int]) -> None:
        if not worker_tags:
            return
        workers = [u for u in bot.workers if u.tag in worker_tags]
        if not workers:
            return

        mineral_targets = _local_minerals(bot)
        gas_targets = _gas_targets(bot)
        if not mineral_targets and not gas_targets:
            return

        from sc2.ids.unit_typeid import UnitTypeId

        ready_extractors = bot.structures(UnitTypeId.EXTRACTOR).ready
        can_gather_gas = bool(gas_targets) and ready_extractors.exists

        for worker in workers:
            if worker.is_carrying_minerals or worker.is_carrying_vespene:
                if bot.townhalls:
                    bot.do(worker.return_resource())
                continue
            if not _worker_needs_mining_order(worker):
                continue

            if mineral_targets:
                nearest = min(mineral_targets, key=lambda m: worker.distance_to(m))
                bot.do(worker.gather(nearest))
                continue

            if can_gather_gas:
                assigned_gas = sum(
                    1
                    for w in bot.workers
                    if w.is_gathering
                    and w.order_target in {g.tag for g in gas_targets}
                )
                if assigned_gas < len(gas_targets) * 2:
                    nearest = min(gas_targets, key=lambda g: worker.distance_to(g))
                    bot.do(worker.gather(nearest))


class LearnedMiningController(MiningController):
    """Stub for future RL mining policy replacement."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path

    async def step(self, bot: "BotAI", worker_tags: Set[int]) -> None:
        raise NotImplementedError(
            "LearnedMiningController is not implemented yet; use ScriptedMiningController."
        )
