import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.runtime.mining_controller import NoOpMiningController, ScriptedMiningController


def test_noop_mining_controller():
    bot = MagicMock()
    ctrl = NoOpMiningController()
    asyncio.run(ctrl.step(bot, {1, 2, 3}))


def test_scripted_mining_controller_assigns_gather():
    bot = MagicMock()
    worker = MagicMock()
    worker.tag = 1
    worker.is_idle = True
    worker.is_gathering = False
    worker.is_carrying_minerals = False
    worker.is_carrying_vespene = False
    worker.distance_to = lambda p: 1.0
    worker.gather = MagicMock()

    mineral = MagicMock()
    mineral.tag = 99
    bot.workers = [worker]
    bot.mineral_field = [mineral]
    bot.vespene_geyser = []
    bot.townhalls = []

    ctrl = ScriptedMiningController()
    asyncio.run(ctrl.step(bot, {1}))
    worker.gather.assert_called_once_with(mineral)


def test_scripted_mining_skips_raw_geyser_without_extractor():
    bot = MagicMock()
    worker = MagicMock()
    worker.tag = 1
    worker.is_idle = True
    worker.is_gathering = False
    worker.is_carrying_minerals = False
    worker.is_carrying_vespene = False
    worker.distance_to = lambda p: 0.5 if getattr(p, "tag", None) == 99 else 5.0

    mineral = MagicMock()
    mineral.tag = 99
    geyser = MagicMock()
    geyser.tag = 77

    bot.workers = [worker]
    bot.mineral_field = [mineral]
    bot.vespene_geyser = [geyser]
    bot.townhalls = []
    worker.gather = MagicMock()
    bot.do = MagicMock()

    empty_ready = MagicMock()
    empty_ready.exists = False
    structures = MagicMock()
    structures.ready = empty_ready
    bot.structures = MagicMock(return_value=structures)

    ctrl = ScriptedMiningController()
    asyncio.run(ctrl.step(bot, {1}))

    worker.gather.assert_called_once_with(mineral)
    bot.do.assert_not_called()
