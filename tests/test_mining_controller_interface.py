import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.runtime.mining_controller import (
    NoOpMiningController,
    ScriptedMiningController,
    _local_minerals,
    _worker_needs_mining_order,
)
from sc2.position import Point2


def test_noop_mining_controller():
    bot = MagicMock()
    ctrl = NoOpMiningController()
    asyncio.run(ctrl.step(bot, {1, 2, 3}))


def test_scripted_mining_controller_uses_bot_do():
    bot = MagicMock()
    worker = MagicMock()
    worker.tag = 1
    worker.is_idle = True
    worker.is_gathering = False
    worker.is_carrying_minerals = False
    worker.is_carrying_vespene = False
    worker.orders = []
    worker.distance_to = lambda p: 1.0
    worker.gather = MagicMock(return_value="gather_cmd")

    mineral = MagicMock()
    mineral.tag = 99
    mineral.distance_to = lambda p: 1.0
    bot.workers = [worker]
    bot.mineral_field = [mineral]
    bot.vespene_geyser = []
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.first.position = Point2((10.0, 10.0))
    bot.do = MagicMock()
    bot.structures = MagicMock(return_value=MagicMock(ready=MagicMock(exists=False)))

    ctrl = ScriptedMiningController()
    asyncio.run(ctrl.step(bot, {1}))
    worker.gather.assert_called_once_with(mineral)
    bot.do.assert_called_once_with("gather_cmd")


def test_scripted_mining_prefers_local_minerals_near_hatch():
    bot = MagicMock()
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.first.position = Point2((0.0, 0.0))

    near = MagicMock()
    near.distance_to = lambda p: 5.0
    far = MagicMock()
    far.distance_to = lambda p: 40.0
    bot.mineral_field = [near, far]
    bot.all_units = []

    local = _local_minerals(bot, radius=15.0)
    assert local == [near]


def test_scripted_mining_reissues_when_idle():
    bot = MagicMock()
    worker = MagicMock()
    worker.tag = 1
    worker.is_idle = True
    worker.is_gathering = False
    worker.is_carrying_minerals = False
    worker.is_carrying_vespene = False
    worker.orders = []
    worker.distance_to = lambda p: 1.0
    worker.gather = MagicMock(return_value="g")

    mineral = MagicMock()
    mineral.tag = 99
    mineral.distance_to = lambda p: 1.0
    bot.workers = [worker]
    bot.mineral_field = [mineral]
    bot.vespene_geyser = []
    bot.townhalls = MagicMock()
    bot.townhalls.exists = False
    bot._spawn_hatchery_pos = Point2((0.0, 0.0))
    bot.do = MagicMock()
    bot.structures = MagicMock(return_value=MagicMock(ready=MagicMock(exists=False)))

    assert _worker_needs_mining_order(worker) is True
    ctrl = ScriptedMiningController()
    asyncio.run(ctrl.step(bot, {1}))
    bot.do.assert_called_once()


def test_scripted_mining_skips_raw_geyser_without_extractor():
    bot = MagicMock()
    worker = MagicMock()
    worker.tag = 1
    worker.is_idle = True
    worker.is_gathering = False
    worker.is_carrying_minerals = False
    worker.is_carrying_vespene = False
    worker.orders = []
    worker.distance_to = lambda p: 0.5 if getattr(p, "tag", None) == 99 else 5.0

    mineral = MagicMock()
    mineral.tag = 99
    mineral.distance_to = lambda p: 0.5
    geyser = MagicMock()
    geyser.tag = 77

    bot.workers = [worker]
    bot.mineral_field = [mineral]
    bot.vespene_geyser = [geyser]
    bot.townhalls = MagicMock()
    bot.townhalls.exists = False
    bot._spawn_hatchery_pos = Point2((0.0, 0.0))
    worker.gather = MagicMock(return_value="g")
    bot.do = MagicMock()

    empty_ready = MagicMock()
    empty_ready.exists = False
    structures = MagicMock()
    structures.ready = empty_ready
    bot.structures = MagicMock(return_value=structures)

    ctrl = ScriptedMiningController()
    asyncio.run(ctrl.step(bot, {1}))

    worker.gather.assert_called_once_with(mineral)
    bot.do.assert_called_once_with("g")
