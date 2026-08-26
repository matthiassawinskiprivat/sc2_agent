"""Mining controllers must not issue gather orders that cancel builds."""

import asyncio
from unittest.mock import MagicMock

from sc2.position import Point2

from src.runtime.mining_controller import (
    ScriptedMiningController,
    _worker_is_building,
    _worker_needs_mining_order,
)
from src.runtime.speed_mining_controller import SpeedMiningController


def test_worker_is_building_detects_build_ability():
    worker = MagicMock()
    worker.is_constructing = False
    ability = MagicMock()
    ability.id = MagicMock()
    ability.id.name = "BUILD_SPAWNINGPOOL"
    order = MagicMock()
    order.ability = ability
    worker.orders = [order]
    assert _worker_is_building(worker) is True
    assert _worker_needs_mining_order(worker) is False


def _make_bot_with_builder(builder_tag: int = 1):
    hatch_pos = Point2((50.0, 50.0))
    bot = MagicMock()
    bot._mine_anchor_pos = hatch_pos
    bot._spawn_hatchery_pos = hatch_pos
    bot.start_location = hatch_pos

    builder = MagicMock()
    builder.tag = builder_tag
    builder.is_constructing = True
    builder.is_carrying_minerals = False
    builder.is_carrying_vespene = False
    builder.is_idle = False
    builder.is_gathering = False
    builder.is_returning = False
    builder.orders = [MagicMock()]
    builder.position = Point2((48.0, 50.0))
    builder.distance_to = lambda p: 2.0
    bot.workers = [builder]

    mineral = MagicMock()
    mineral.tag = 99
    mineral.position = Point2((55.0, 50.0))
    mineral.distance_to = lambda p: 5.0
    bot.mineral_field = [mineral]

    townhall = MagicMock()
    townhall.position = hatch_pos
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.first = townhall
    bot.townhalls.closest_to = MagicMock(return_value=townhall)
    bot.do = MagicMock()
    bot.structures = MagicMock(return_value=MagicMock(exists=False, ready=MagicMock(exists=False)))
    return bot


def test_scripted_mining_skips_building_worker():
    async def _run():
        bot = _make_bot_with_builder(1)
        ctrl = ScriptedMiningController()
        await ctrl.step(bot, {1})
        bot.do.assert_not_called()

    asyncio.run(_run())


def test_speed_mining_skips_building_worker():
    async def _run():
        bot = _make_bot_with_builder(7)
        ctrl = SpeedMiningController()
        await ctrl.step(bot, {7})
        bot.do.assert_not_called()

    asyncio.run(_run())
