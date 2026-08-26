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


def test_speed_mining_fills_gas_from_busy_mineral_workers():
    """Regression: gas fill must not require idle workers (miners are almost never idle)."""

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
        bot.do = MagicMock()
        bot.vespene_geyser = []
        bot._gas_worker_quota = 3

        mineral = _make_mineral(10, Point2((5.0, 0.0)))
        bot.mineral_field = [mineral]

        extractor = MagicMock()
        extractor.tag = 55
        extractor.position = Point2((3.0, 3.0))
        ready = MagicMock()
        ready.exists = True
        ready.__iter__ = lambda self: iter([extractor])
        structures = MagicMock()
        structures.ready = ready
        bot.structures = MagicMock(return_value=structures)

        workers = []
        for i in range(4):
            w = _make_worker(
                i + 1,
                Point2((1.0 + 0.1 * i, 0.0)),
                is_idle=False,
                is_gathering=True,
                orders=[MagicMock()],
                order_target=10,
            )
            workers.append(w)
        bot.workers = workers

        ctrl = SpeedMiningController()
        # Pre-assign all to the mineral patch (saturated).
        for w in workers:
            ctrl._worker_to_patch[int(w.tag)] = 10
            ctrl._patch_to_workers.setdefault(10, set()).add(int(w.tag))

        await ctrl.step(bot, {1, 2, 3, 4})

        assert len(ctrl._gas_workers) == 3
        assert any(w.gather.called for w in workers)
        gas_gathers = [
            w for w in workers if w.gather.called and w.gather.call_args[0][0] is extractor
        ]
        assert len(gas_gathers) == 3

        # Next step must keep gas workers off minerals.
        bot.do.reset_mock()
        for w in workers:
            w.gather.reset_mock()
            if int(w.tag) in ctrl._gas_workers:
                w.order_target = 55
                w.is_gathering = True
                w.is_idle = False
                w.orders = [MagicMock()]
        await ctrl.step(bot, {1, 2, 3, 4})
        assert len(ctrl._gas_workers) == 3
        for wtag in ctrl._gas_workers:
            assert wtag not in ctrl._worker_to_patch

    asyncio.run(_run())


def test_speed_mining_pulls_excess_gas_workers_down_to_quota():
    """Regression: extras must get mineral gather, not only leave the gas set."""

    async def _run():
        bot = MagicMock()
        hatch = Point2((0.0, 0.0))
        bot._mine_anchor_pos = hatch
        bot._gas_worker_quota = 3
        bot.townhalls = MagicMock()
        bot.townhalls.exists = True
        th = MagicMock()
        th.tag = 99
        th.position = hatch
        bot.townhalls.first = th
        bot.townhalls.closest_to = MagicMock(return_value=th)
        bot.do = MagicMock()
        bot.vespene_geyser = []

        mineral = _make_mineral(10, Point2((5.0, 0.0)))
        bot.mineral_field = [mineral]

        extractor = MagicMock()
        extractor.tag = 55
        extractor.position = Point2((3.0, 3.0))
        ready = MagicMock()
        ready.exists = True
        ready.__iter__ = lambda self: iter([extractor])
        structures = MagicMock()
        structures.ready = ready
        bot.structures = MagicMock(return_value=structures)

        workers = []
        for i in range(4):
            w = _make_worker(
                i + 1,
                Point2((3.0 + 0.1 * i, 3.0)),
                is_idle=False,
                is_gathering=True,
                orders=[MagicMock()],
                order_target=55,
            )
            workers.append(w)
        bot.workers = workers

        ctrl = SpeedMiningController()
        for w in workers:
            ctrl._gas_workers.add(int(w.tag))

        await ctrl.step(bot, {1, 2, 3, 4})

        assert len(ctrl._gas_workers) == 3
        mineral_gathers = [
            w
            for w in workers
            if w.gather.called and w.gather.call_args[0][0] is mineral
        ]
        assert len(mineral_gathers) == 1

    asyncio.run(_run())


def test_speed_mining_respects_zero_gas_quota():
    async def _run():
        bot = MagicMock()
        hatch = Point2((0.0, 0.0))
        bot._mine_anchor_pos = hatch
        bot._gas_worker_quota = 0
        bot.townhalls = MagicMock()
        bot.townhalls.exists = True
        th = MagicMock()
        th.tag = 99
        th.position = hatch
        bot.townhalls.first = th
        bot.townhalls.closest_to = MagicMock(return_value=th)
        bot.do = MagicMock()
        bot.vespene_geyser = []
        mineral = _make_mineral(10, Point2((5.0, 0.0)))
        bot.mineral_field = [mineral]

        extractor = MagicMock()
        extractor.tag = 55
        extractor.position = Point2((3.0, 3.0))
        ready = MagicMock()
        ready.exists = True
        ready.__iter__ = lambda self: iter([extractor])
        structures = MagicMock()
        structures.ready = ready
        bot.structures = MagicMock(return_value=structures)

        w = _make_worker(
            1,
            Point2((3.0, 3.0)),
            is_idle=False,
            is_gathering=True,
            orders=[MagicMock()],
            order_target=55,
        )
        bot.workers = [w]
        ctrl = SpeedMiningController()
        ctrl._gas_workers.add(1)
        await ctrl.step(bot, {1})
        assert ctrl._gas_workers == set()
        w.gather.assert_called_with(mineral)

    asyncio.run(_run())


def test_speed_mining_no_fourth_drone_when_gas_worker_idle_after_return():
    """Regression: post-return idle reserved gas worker must not free a 4th slot."""

    async def _run():
        bot = MagicMock()
        hatch = Point2((0.0, 0.0))
        bot._mine_anchor_pos = hatch
        bot._gas_worker_quota = 3
        bot.townhalls = MagicMock()
        bot.townhalls.exists = True
        th = MagicMock()
        th.tag = 99
        th.position = hatch
        bot.townhalls.first = th
        bot.townhalls.closest_to = MagicMock(return_value=th)
        bot.do = MagicMock()
        bot.vespene_geyser = []

        mineral = _make_mineral(10, Point2((5.0, 0.0)))
        bot.mineral_field = [mineral]

        extractor = MagicMock()
        extractor.tag = 55
        extractor.position = Point2((3.0, 3.0))
        ready = MagicMock()
        ready.exists = True
        ready.__iter__ = lambda self: iter([extractor])
        structures = MagicMock()
        structures.ready = ready
        bot.structures = MagicMock(return_value=structures)

        # A briefly idle after return boost; B/C still on extractor; D on minerals.
        a = _make_worker(1, Point2((1.0, 0.0)), is_idle=True, orders=[], order_target=None)
        b = _make_worker(
            2,
            Point2((3.0, 3.0)),
            is_idle=False,
            is_gathering=True,
            orders=[MagicMock()],
            order_target=55,
        )
        c = _make_worker(
            3,
            Point2((3.1, 3.0)),
            is_idle=False,
            is_gathering=True,
            orders=[MagicMock()],
            order_target=55,
        )
        d = _make_worker(
            4,
            Point2((5.0, 0.0)),
            is_idle=False,
            is_gathering=True,
            orders=[MagicMock()],
            order_target=10,
        )
        bot.workers = [a, b, c, d]

        ctrl = SpeedMiningController()
        ctrl._gas_workers = {1, 2, 3}
        ctrl._worker_to_patch[4] = 10
        ctrl._patch_to_workers[10] = {4}

        await ctrl.step(bot, {1, 2, 3, 4})

        assert ctrl._gas_workers == {1, 2, 3}
        assert not d.gather.called or (
            d.gather.called and d.gather.call_args[0][0] is not extractor
        )
        # Idle reserved worker is re-queued onto the extractor (not replaced).
        a.gather.assert_called_with(extractor)

    asyncio.run(_run())


def test_speed_mining_caps_gas_cycle_when_returner_pushes_over_quota():
    """Regression: 3 on extractor + 1 returning must not keep 4 reserved forever."""

    async def _run():
        bot = MagicMock()
        hatch = Point2((0.0, 0.0))
        bot._mine_anchor_pos = hatch
        bot._gas_worker_quota = 3
        bot.townhalls = MagicMock()
        bot.townhalls.exists = True
        th = MagicMock()
        th.tag = 99
        th.position = hatch
        bot.townhalls.first = th
        bot.townhalls.closest_to = MagicMock(return_value=th)
        bot.do = MagicMock()
        bot.vespene_geyser = []

        mineral = _make_mineral(10, Point2((5.0, 0.0)))
        bot.mineral_field = [mineral]

        extractor = MagicMock()
        extractor.tag = 55
        extractor.position = Point2((3.0, 3.0))
        ready = MagicMock()
        ready.exists = True
        ready.__iter__ = lambda self: iter([extractor])
        structures = MagicMock()
        structures.ready = ready
        bot.structures = MagicMock(return_value=structures)

        on_ext = []
        for i in range(3):
            on_ext.append(
                _make_worker(
                    i + 1,
                    Point2((3.0 + 0.1 * i, 3.0)),
                    is_idle=False,
                    is_gathering=True,
                    orders=[MagicMock()],
                    order_target=55,
                )
            )
        # 4th is returning with gas — old logic kept all 4 in _gas_workers.
        ret = _make_worker(
            4,
            Point2((1.0, 0.0)),
            is_idle=False,
            is_returning=True,
            orders=[MagicMock()],
            order_target=99,
        )
        ret.is_carrying_vespene = True
        ret.is_carrying_resource = True
        bot.workers = on_ext + [ret]

        ctrl = SpeedMiningController()
        ctrl._gas_workers = {1, 2, 3, 4}

        await ctrl.step(bot, {1, 2, 3, 4})

        assert len(ctrl._gas_workers) == 3
        # The farthest / lowest-priority excess must be pulled to minerals.
        pulled = [
            w
            for w in bot.workers
            if w.gather.called and w.gather.call_args[0][0] is mineral
        ]
        assert len(pulled) == 1
        assert int(pulled[0].tag) not in ctrl._gas_workers

    asyncio.run(_run())


def test_speed_mining_steals_non_reserved_drone_off_extractor():
    """Regression: a 4th drone on gas (not in _gas_workers) must be forced to minerals."""

    async def _run():
        bot = MagicMock()
        hatch = Point2((0.0, 0.0))
        bot._mine_anchor_pos = hatch
        bot._gas_worker_quota = 3
        bot.townhalls = MagicMock()
        bot.townhalls.exists = True
        th = MagicMock()
        th.tag = 99
        th.position = hatch
        bot.townhalls.first = th
        bot.townhalls.closest_to = MagicMock(return_value=th)
        bot.do = MagicMock()
        bot.vespene_geyser = []

        mineral = _make_mineral(10, Point2((5.0, 0.0)))
        bot.mineral_field = [mineral]

        extractor = MagicMock()
        extractor.tag = 55
        extractor.position = Point2((3.0, 3.0))
        ready = MagicMock()
        ready.exists = True
        ready.__iter__ = lambda self: iter([extractor])
        structures = MagicMock()
        structures.ready = ready
        bot.structures = MagicMock(return_value=structures)

        reserved = [
            _make_worker(
                i + 1,
                Point2((3.0 + 0.1 * i, 3.0)),
                is_idle=False,
                is_gathering=True,
                orders=[MagicMock(), MagicMock()],  # multi-order queue
                order_target=55,
            )
            for i in range(3)
        ]
        # 4th has multi-order gas gather and is NOT reserved — old code early-returned.
        fourth = _make_worker(
            4,
            Point2((3.3, 3.0)),
            is_idle=False,
            is_gathering=True,
            orders=[MagicMock(), MagicMock()],
            order_target=55,
        )
        bot.workers = reserved + [fourth]

        ctrl = SpeedMiningController()
        ctrl._gas_workers = {1, 2, 3}

        await ctrl.step(bot, {1, 2, 3, 4})

        assert ctrl._gas_workers == {1, 2, 3}
        fourth.gather.assert_called_with(mineral)
        assert fourth.gather.call_args[0][0] is mineral

    asyncio.run(_run())

