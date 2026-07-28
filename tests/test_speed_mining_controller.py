import asyncio
from unittest.mock import MagicMock

from sc2.position import Point2

from src.runtime.speed_mining_controller import (
    TOWNHALL_RADIUS,
    TOWNHALL_RETURN_FACTOR,
    SpeedMiningController,
    _mineral_boost_point,
    _townhall_boost_point,
)


def _make_worker(tag: int, pos: Point2, **kwargs):
    w = MagicMock()
    w.tag = tag
    w.position = pos
    w.orders = kwargs.get("orders", [])
    w.is_idle = kwargs.get("is_idle", False)
    w.is_gathering = kwargs.get("is_gathering", False)
    w.is_returning = kwargs.get("is_returning", False)
    w.is_carrying_minerals = kwargs.get("is_carrying_minerals", False)
    w.is_carrying_vespene = False
    w.is_carrying_resource = w.is_carrying_minerals
    w.order_target = kwargs.get("order_target", None)
    w.distance_to = lambda p, _pos=pos: (
        (_pos.x - (getattr(p, "position", p).x)) ** 2
        + (_pos.y - (getattr(p, "position", p).y)) ** 2
    ) ** 0.5
    w.move = MagicMock(return_value=f"move-{tag}")
    w.gather = MagicMock(return_value=f"gather-{tag}")
    return w


def _make_mineral(tag: int, pos: Point2):
    m = MagicMock()
    m.tag = tag
    m.position = pos
    m.radius = 1.125
    m.distance_to = lambda p, _pos=pos: ((_pos.x - p.x) ** 2 + (_pos.y - p.y) ** 2) ** 0.5
    return m


def test_mineral_boost_point_lies_between_patch_and_hatch():
    mineral = _make_mineral(1, Point2((20.0, 20.0)))
    hatch = Point2((10.0, 10.0))
    boost = _mineral_boost_point(mineral, hatch)
    assert mineral.position.distance_to(hatch) > boost.distance_to(hatch)


def test_townhall_boost_point_offsets_from_hatch():
    th = MagicMock()
    th.position = Point2((0.0, 0.0))
    worker_pos = Point2((10.0, 0.0))
    p = _townhall_boost_point(th, worker_pos)
    expected = TOWNHALL_RADIUS * TOWNHALL_RETURN_FACTOR
    assert abs(p.distance_to(th.position) - expected) < 0.05
    assert p.x > 0


def test_speed_mining_sticky_patch_assignment():
    async def _run():
        bot = MagicMock()
        hatch = Point2((0.0, 0.0))
        bot._mine_anchor_pos = hatch
        bot.townhalls = MagicMock()
        bot.townhalls.exists = True
        th = MagicMock()
        th.tag = 99
        th.position = hatch
        bot.townhalls.first = th
        bot.townhalls.closest_to = MagicMock(return_value=th)

        m1 = _make_mineral(10, Point2((5.0, 0.0)))
        m2 = _make_mineral(11, Point2((6.0, 0.0)))
        bot.mineral_field = [m1, m2]
        bot.structures = MagicMock(return_value=MagicMock(ready=MagicMock(exists=False)))
        bot.vespene_geyser = []
        bot.do = MagicMock()

        w1 = _make_worker(1, Point2((1.0, 0.0)), is_idle=True)
        w2 = _make_worker(2, Point2((1.5, 0.0)), is_idle=True)
        bot.workers = [w1, w2]

        ctrl = SpeedMiningController()
        await ctrl.step(bot, {1, 2})
        assert ctrl._worker_to_patch[1] in {10, 11}
        assert ctrl._worker_to_patch[2] in {10, 11}
        first = dict(ctrl._worker_to_patch)
        await ctrl.step(bot, {1, 2})
        assert ctrl._worker_to_patch == first

    asyncio.run(_run())


def test_speed_mining_issues_move_when_in_boost_band():
    async def _run():
        bot = MagicMock()
        hatch_pos = Point2((0.0, 0.0))
        bot._mine_anchor_pos = hatch_pos
        bot.townhalls = MagicMock()
        bot.townhalls.exists = True
        th = MagicMock()
        th.tag = 99
        th.position = hatch_pos
        bot.townhalls.first = th
        bot.townhalls.closest_to = MagicMock(return_value=th)
        bot.structures = MagicMock(return_value=MagicMock(ready=MagicMock(exists=False)))
        bot.vespene_geyser = []
        bot.do = MagicMock()

        mineral = _make_mineral(10, Point2((8.0, 0.0)))
        bot.mineral_field = [mineral]
        boost = _mineral_boost_point(mineral, hatch_pos)
        # Offset ~1.0 from boost so dist_sq ~= 1.0 (inside 0.5625..4.0)
        worker = _make_worker(
            1,
            Point2((boost.x - 1.0, boost.y)),
            is_idle=False,
            is_gathering=True,
            orders=[MagicMock()],
        )
        bot.workers = [worker]

        ctrl = SpeedMiningController()
        ctrl._worker_to_patch[1] = 10
        ctrl._patch_to_workers[10] = {1}
        await ctrl.step(bot, {1})

        worker.move.assert_called()
        assert bot.do.call_count >= 1

    asyncio.run(_run())


def test_resolve_speed_mining_controller():
    from src.zerg_build_bot import ZergBuildBot
    from src.runtime.speed_mining_controller import SpeedMiningController

    bot = ZergBuildBot(training_conf={"mining_controller": "speed"})
    assert isinstance(bot.mining_controller, SpeedMiningController)
