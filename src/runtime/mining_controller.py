from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, List, Set

if TYPE_CHECKING:
    from sc2.bot_ai import BotAI


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


class ScriptedMiningController(MiningController):
    """
    Simple mineral/gas assignment using burnysc2 (python-sc2) gather orders.

    Not a separate speed-mining library: workers are sent with ``unit.gather()``.
    Raw geysers are skipped until a ready extractor exists (drones cannot mine gas otherwise).
    """

    async def step(self, bot: "BotAI", worker_tags: Set[int]) -> None:
        if not worker_tags:
            return
        workers = [u for u in bot.workers if u.tag in worker_tags]
        if not workers:
            return

        mineral_targets = _mineral_targets(bot)
        gas_targets = _gas_targets(bot)
        if not mineral_targets and not gas_targets:
            return

        from sc2.ids.unit_typeid import UnitTypeId

        ready_extractors = bot.structures(UnitTypeId.EXTRACTOR).ready
        can_gather_gas = bool(gas_targets) and ready_extractors.exists

        for worker in workers:
            if worker.is_carrying_minerals or worker.is_carrying_vespene:
                if bot.townhalls:
                    worker.return_resource()
                continue
            if worker.is_gathering:
                continue

            if mineral_targets:
                nearest = min(mineral_targets, key=lambda m: worker.distance_to(m))
                worker.gather(nearest)
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
                    worker.gather(nearest)


class LearnedMiningController(MiningController):
    """Stub for future RL mining policy replacement."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path

    async def step(self, bot: "BotAI", worker_tags: Set[int]) -> None:
        raise NotImplementedError(
            "LearnedMiningController is not implemented yet; use ScriptedMiningController."
        )
