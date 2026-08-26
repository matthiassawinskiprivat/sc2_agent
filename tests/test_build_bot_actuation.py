import asyncio
import random
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.build_constants import BUILD_PLACEMENT_CELLS, MAX_BUILD_WORKERS
from src.agent.commands import Command, CommandType
from src.runtime.scripted_manager import ScriptedManager
from src.runtime.worker_registry import WorkerRole
from src.zerg_build_bot import ZergBuildBot


class _EmptyKillable:
    def __bool__(self):
        return False

    @property
    def amount(self):
        return 0

    def __iter__(self):
        return iter([])


def test_on_step_zero_runs_reset_game_state():
    bot = ZergBuildBot(training_conf={})
    bot.reset_game_state = AsyncMock()
    asyncio.run(bot.on_step(0))
    bot.reset_game_state.assert_awaited_once_with(skip_cleanup=True)
    assert bot._game_state_initialized is True


def test_bot_initializes_botai_unit_caches():
    bot = ZergBuildBot(training_conf={})
    assert hasattr(bot, "units")
    assert hasattr(bot, "structures")
    assert hasattr(bot, "workers")


def test_decode_placement_centered_on_manager_target():
    bot = ZergBuildBot(training_conf={})
    center = Point2((40.0, 50.0))
    # cell (8, 8) on 16x16 grid -> offset 0,0 from center (+0.5 pixel center)
    pos = bot.decode_placement(8 * 16 + 8, center)
    assert abs(pos.x - 40.5) < 0.01
    assert abs(pos.y - 50.5) < 0.01


def test_action_masks_always_all_true():
    bot = ZergBuildBot(training_conf={})
    drone = MagicMock()
    drone.tag = 7
    bot.workers = [drone]
    mask = bot.action_masks()
    assert mask.dtype == bool
    assert mask.shape == (MAX_BUILD_WORKERS + BUILD_PLACEMENT_CELLS,)
    assert mask.all()


def test_find_buildable_near_falls_back_when_geyser_tile_not_pathable():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.playable_area = MagicMock()
    bot.game_info.playable_area.x = 0
    bot.game_info.playable_area.y = 0
    bot.game_info.playable_area.width = 64
    bot.game_info.playable_area.height = 64
    bot.game_info.playable_area.contains = lambda pos: True
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[10, 10] = 1
    bot.game_info.pathing_grid = MagicMock()
    bot.game_info.pathing_grid.__getitem__ = lambda self, key: grid[key[1], key[0]]
    bot._default_hatch_location = MagicMock(return_value=Point2((10.0, 10.0)))
    geyser_pos = Point2((40.0, 40.0))
    pos = bot._find_buildable_near(geyser_pos, min_dist=6.0, max_dist=8.0)
    bot._default_hatch_location.assert_called_once()
    assert pos == Point2((10.0, 10.0))


def test_find_buildable_near_uses_rect_bounds_without_contains():
    from sc2.position import Rect

    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.playable_area = Rect((32.0, 32.0, 64.0, 64.0))
    grid = np.zeros((128, 128), dtype=np.uint8)
    grid[40, 40] = 1
    bot.game_info.pathing_grid = MagicMock()
    bot.game_info.pathing_grid.__getitem__ = lambda self, key: grid[key[1], key[0]]
    center = Point2((40.0, 40.0))
    pos = bot._find_buildable_near(center, min_dist=0.0, max_dist=2.0)
    assert pos.x == 40.0
    assert pos.y == 40.0


def test_find_buildable_near_returns_pathable_tile():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.playable_area.contains = lambda pos: True
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[32, 32] = 1
    bot.game_info.pathing_grid = MagicMock()
    bot.game_info.pathing_grid.__getitem__ = lambda self, key: grid[key[1], key[0]]
    center = Point2((32.0, 32.0))
    pos = bot._find_buildable_near(center, min_dist=0.0, max_dist=2.0)
    assert pos.x == 32.0
    assert pos.y == 32.0


def test_spawn_reference_point_falls_back_to_map_center():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.map_center = Point2((50.0, 60.0))
    with patch.object(type(bot), "start_location", new_callable=PropertyMock, return_value=None):
        assert bot._spawn_reference_point() == Point2((50.0, 60.0))


def test_spawn_config_reads_flattened_training_conf():
    bot = ZergBuildBot(
        training_conf={
            "spawn": {"drones": 5, "mining_drones": 2, "minerals": 300},
            "mining_controller": "scripted",
        }
    )
    assert bot._spawn_config()["drones"] == 5
    assert bot._spawn_config()["mining_drones"] == 2
    assert bot._mining_controller_name() == "scripted"


def test_scripted_manager_reset_restores_pristine_target_pos():
    """Regression: soft-reset must revive YAML [-1,-1] after episode mutation."""
    mgr = ScriptedManager.from_level_config(
        {
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "EXTRACTOR",
                    "target_pos": [-1, -1],
                }
            ]
        }
    )
    assert mgr.script_steps[0].command.target_pos == Point2((-1.0, -1.0))
    mgr._emit_step_if_available()
    active = mgr.command_queue.active
    assert active is not None
    # Actuation mutates the *queued* command; script template must stay pristine
    # after reset even if someone also mutates script_steps in place.
    active.target_pos = Point2((56.0, 52.0))
    mgr.script_steps[0].command.target_pos = Point2((56.0, 52.0))
    mgr.reset()
    assert mgr.script_steps[0].command.target_pos == Point2((-1.0, -1.0))
    assert mgr.command_queue.active is None


def test_scripted_manager_reads_flattened_manager_script():
    bot = ZergBuildBot(
        training_conf={
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "SPAWNINGPOOL",
                    "target_pos": [-1, -1],
                }
            ],
            "macro_tick_interval": 8,
            "env": {"realtime": False},
        }
    )
    assert len(bot.scripted_manager.script_steps) == 1
    bot.scripted_manager._emit_step_if_available()
    assert bot.command_queue.has_active_macro()
    assert bot.command_queue.active.unit_type_id.name == "SPAWNINGPOOL"


def test_apply_build_action_issues_worker_build_at_decoded_cell():
    async def _run():
        from sc2.ids.unit_typeid import UnitTypeId

        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "BUILD",
                        "unit_type": "SPAWNINGPOOL",
                        "target_pos": [40.0, 40.0],
                    }
                ],
            }
        )
        bot.scripted_manager._emit_step_if_available()

        worker = MagicMock()
        worker.tag = 42
        worker.position = Point2((30.0, 30.0))
        worker.build = MagicMock(return_value=MagicMock())
        bot.workers = [worker]
        bot.do = MagicMock(return_value=True)
        bot.can_afford = MagicMock(return_value=True)
        bot.already_pending = MagicMock(return_value=0)
        bot.structures = MagicMock(return_value=MagicMock(exists=False))
        bot._placement_cell_legal = MagicMock(return_value=True)

        applied = await bot.apply_build_action(np.array([0, 8 * 16 + 8], dtype=np.int64))
        assert applied is True
        bot.do.assert_called_once()
        worker.build.assert_called_once()
        build_args = worker.build.call_args[0]
        assert build_args[0] == UnitTypeId.SPAWNINGPOOL
        target = build_args[1]
        assert abs(target.x - 40.5) < 0.01
        assert abs(target.y - 40.5) < 0.01
        assert bot.worker_registry.get_role(42) == WorkerRole.BUILD
        assert bot._last_build_attempt_failed is False

    asyncio.run(_run())


def test_apply_build_extractor_targets_geyser_unit_not_point():
    """Burnysc2 asserts gas builds receive the geyser Unit, not a Point2."""

    async def _run():
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.unit import Unit

        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "BUILD",
                        "unit_type": "EXTRACTOR",
                        "target_pos": [40.0, 41.0],
                    }
                ],
            }
        )
        bot.scripted_manager._emit_step_if_available()

        worker = MagicMock()
        worker.tag = 7
        worker.position = Point2((30.0, 30.0))
        worker.build = MagicMock(return_value=MagicMock())
        bot.workers = [worker]
        bot.do = MagicMock(return_value=True)
        bot.can_afford = MagicMock(return_value=True)
        bot.already_pending = MagicMock(return_value=0)
        bot.structures = MagicMock(return_value=MagicMock(exists=False))
        bot._placement_cell_legal = MagicMock(return_value=True)

        geyser = MagicMock(spec=Unit)
        geyser.position = Point2((40.0, 41.0))
        geyser.distance_to = lambda p: 0.0
        bot.vespene_geyser = MagicMock()
        bot.vespene_geyser.exists = True
        bot.vespene_geyser.closest_to.return_value = geyser
        bot._target_geyser_pos = Point2((40.0, 41.0))
        bot._spawn_hatchery_pos = Point2((40.0, 40.0))

        await bot.apply_build_action(np.array([0, 8 * 16 + 8], dtype=np.int64))
        worker.build.assert_called_once()
        build_args = worker.build.call_args[0]
        assert build_args[0] == UnitTypeId.EXTRACTOR
        assert build_args[1] is geyser
        bot.vespene_geyser.closest_to.assert_called()
        assert bot.vespene_geyser.closest_to.call_count >= 1
        assert bot._last_build_attempt_failed is False

    asyncio.run(_run())


def test_apply_build_illegal_placement_noop_with_penalty_flag():
    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "BUILD",
                        "unit_type": "SPAWNINGPOOL",
                        "target_pos": [40.0, 40.0],
                    }
                ],
            }
        )
        bot.scripted_manager._emit_step_if_available()
        worker = MagicMock()
        worker.tag = 1
        worker.position = Point2((30.0, 30.0))
        worker.build = MagicMock()
        bot.workers = [worker]
        bot.do = MagicMock()
        bot.can_afford = MagicMock(return_value=True)
        bot.already_pending = MagicMock(return_value=0)
        bot.structures = MagicMock(return_value=MagicMock(exists=False))
        bot._placement_cell_legal = MagicMock(return_value=False)

        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        bot.do.assert_not_called()
        worker.build.assert_not_called()
        assert bot._last_build_attempt_failed is True
        reward = bot._compute_reward()
        assert reward <= float(bot.rewards_cfg.get("step_penalty", -0.05)) + float(
            bot.rewards_cfg.get("invalid_build_penalty", -1.0)
        ) + 1e-6

    asyncio.run(_run())


def test_apply_train_without_pool_noop_with_penalty_flag():
    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "TRAIN",
                        "unit_type": "ZERGLING",
                        "count": 1,
                    }
                ],
            }
        )
        bot.scripted_manager.emit_next_if_idle()
        bot._train_baseline = 0
        bot.do = MagicMock()
        bot.can_afford = MagicMock(return_value=True)
        bot.structures = MagicMock(return_value=MagicMock(exists=False, ready=MagicMock(exists=False)))
        # structures(...).ready.exists path
        no_pool = MagicMock()
        no_pool.ready = MagicMock()
        no_pool.ready.exists = False
        bot.structures = MagicMock(return_value=no_pool)
        bot.workers = [MagicMock(tag=1)]
        bot.larva = MagicMock(exists=True)

        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        bot.do.assert_not_called()
        assert bot._last_build_attempt_failed is True

    asyncio.run(_run())


def test_placement_cell_legal_requires_pathable_creep_for_pool():
    bot = ZergBuildBot(training_conf={})
    bot.command_queue.enqueue(
        Command(
            command_type=CommandType.BUILD,
            unit_type_id=UnitTypeId.SPAWNINGPOOL,
            target_pos=Point2((32.0, 32.0)),
        )
    )
    bot.command_queue.activate_next()

    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[32, 32] = 1
    bot.game_info = MagicMock()
    bot.game_info.pathing_grid = MagicMock()
    bot.game_info.pathing_grid.data_numpy = grid
    bot.game_info.pathing_grid.__getitem__ = lambda self, key: grid[key[1], key[0]]

    creep_data = np.zeros((64, 64), dtype=np.uint8)
    creep_data[32, 32] = 1

    class _Creep:
        data_numpy = creep_data

    bot.state = MagicMock()
    bot.state.creep = _Creep()

    center_idx = 8 * 16 + 8
    assert bot._placement_cell_legal(bot.command_queue.active, center_idx)
    assert not bot._placement_cell_legal(bot.command_queue.active, 0)


def test_placement_cell_legal_extractor_allows_any_cell_when_geyser_resolved():
    """Gas builds target the geyser Unit; spatial cell must not block issuance."""
    bot = ZergBuildBot(training_conf={})
    geyser_pos = Point2((40.0, 41.0))
    bot._target_geyser_pos = geyser_pos
    bot._spawn_hatchery_pos = Point2((40.0, 40.0))
    geyser = MagicMock()
    geyser.position = geyser_pos
    geyser.distance_to = lambda p: 0.0
    bot.vespene_geyser = MagicMock()
    bot.vespene_geyser.exists = True
    bot.vespene_geyser.closest_to = MagicMock(return_value=geyser)
    bot.command_queue.enqueue(
        Command(
            command_type=CommandType.BUILD,
            unit_type_id=UnitTypeId.EXTRACTOR,
            target_pos=geyser_pos,
        )
    )
    bot.command_queue.activate_next()
    # Corner cell used to be illegal and delayed the first extractor until ~200 minerals.
    assert bot._placement_cell_legal(bot.command_queue.active, 0)
    assert bot._placement_cell_legal(bot.command_queue.active, 8 * 16 + 8)


def test_patch_script_targets_updates_extractor_each_episode():
    """Regression: episode 2+ must not keep episode 1 geyser coordinates."""
    bot = ZergBuildBot(
        training_conf={
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "EXTRACTOR",
                    "target_pos": [-1, -1],
                },
            ],
        }
    )
    assert bot._script_anchor_to_hatch[0] is True

    hatch1 = Point2((50.0, 50.0))
    geyser1 = Point2((56.0, 52.0))
    bot._spawn_hatchery_pos = hatch1
    g1 = MagicMock()
    g1.position = geyser1
    g1.distance_to = lambda p, _g=geyser1: _g.distance_to(getattr(p, "position", p))
    bot.vespene_geyser = MagicMock()
    bot.vespene_geyser.exists = True
    bot.vespene_geyser.closest_to = MagicMock(return_value=g1)
    bot._patch_script_targets()
    assert bot.scripted_manager.script_steps[0].command.target_pos == geyser1

    hatch2 = Point2((150.0, 150.0))
    geyser2 = Point2((156.0, 152.0))
    bot._spawn_hatchery_pos = hatch2
    g2 = MagicMock()
    g2.position = geyser2
    g2.distance_to = lambda p, _g=geyser2: _g.distance_to(getattr(p, "position", p))
    bot.vespene_geyser.closest_to = MagicMock(return_value=g2)
    # Stale leftover from episode 1 must be overwritten.
    bot._target_geyser_pos = geyser1
    bot._patch_script_targets()
    assert bot.scripted_manager.script_steps[0].command.target_pos == geyser2
    assert bot._target_geyser_pos == geyser2


def test_apply_build_extractor_ignores_stale_episode_target_pos():
    """Regression: queued EXTRACTOR must rebind to current hatch geyser."""

    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "BUILD",
                        "unit_type": "EXTRACTOR",
                        "target_pos": [56.0, 52.0],  # stale ep1 coords
                    }
                ],
            }
        )
        bot.scripted_manager._emit_step_if_available()
        # Episode 2 hatch far from stale coords.
        bot._spawn_hatchery_pos = Point2((150.0, 150.0))
        bot._target_geyser_pos = Point2((56.0, 52.0))  # leftover

        local = MagicMock()
        local.position = Point2((156.0, 152.0))
        local.tag = 2
        local.distance_to = lambda p: Point2((156.0, 152.0)).distance_to(
            getattr(p, "position", p)
        )
        stale = MagicMock()
        stale.position = Point2((56.0, 52.0))
        stale.tag = 1
        stale.distance_to = lambda p: Point2((56.0, 52.0)).distance_to(
            getattr(p, "position", p)
        )

        def _closest(p):
            anchor = getattr(p, "position", p)
            if Point2((156.0, 152.0)).distance_to(anchor) <= Point2(
                (56.0, 52.0)
            ).distance_to(anchor):
                return local
            return stale

        geysers = MagicMock()
        geysers.exists = True
        geysers.closest_to = MagicMock(side_effect=_closest)
        bot.vespene_geyser = geysers

        worker = MagicMock()
        worker.tag = 7
        worker.position = Point2((150.0, 150.0))
        worker.build = MagicMock(return_value=MagicMock())
        bot.workers = [worker]
        bot.do = MagicMock(return_value=True)
        bot.can_afford = MagicMock(return_value=True)
        bot.already_pending = MagicMock(return_value=0)
        bot.structures = MagicMock(return_value=MagicMock(exists=False))
        bot._placement_cell_legal = MagicMock(return_value=True)

        await bot.apply_build_action(np.array([0, 8 * 16 + 8], dtype=np.int64))
        worker.build.assert_called_once()
        assert worker.build.call_args[0][1] is local
        assert bot.command_queue.active.target_pos == Point2((156.0, 152.0))
        assert bot._target_geyser_pos == Point2((156.0, 152.0))

    asyncio.run(_run())


def test_cell_has_geyser_ignores_far_map_geysers():
    bot = ZergBuildBot(training_conf={})
    bot._spawn_hatchery_pos = Point2((50.0, 50.0))
    bot._target_geyser_pos = Point2((56.0, 52.0))
    far = MagicMock()
    far.position = Point2((180.0, 180.0))
    bot.vespene_geyser = [far]
    assert bot._cell_has_geyser(56, 52) is True
    assert bot._cell_has_geyser(180, 180) is False


def test_apply_build_extractor_uses_local_geyser_not_far_closest():
    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "BUILD",
                        "unit_type": "EXTRACTOR",
                        "target_pos": [56.0, 52.0],
                    }
                ],
            }
        )
        bot.scripted_manager._emit_step_if_available()
        bot._spawn_hatchery_pos = Point2((50.0, 50.0))
        bot._target_geyser_pos = Point2((56.0, 52.0))

        local = MagicMock()
        local.position = Point2((56.0, 52.0))
        local.tag = 1
        local.distance_to = lambda p: Point2((56.0, 52.0)).distance_to(
            getattr(p, "position", p)
        )
        far = MagicMock()
        far.position = Point2((180.0, 180.0))
        far.tag = 2
        far.distance_to = lambda p: Point2((180.0, 180.0)).distance_to(
            getattr(p, "position", p)
        )

        geysers = MagicMock()
        geysers.exists = True
        geysers.closest_to = MagicMock(side_effect=lambda p: far)  # would pick far
        # Iteration / local helper needs real list behavior via closest within range.
        def _closest(p):
            # Prefer local when near hatch/geyser anchor.
            anchor = getattr(p, "position", p)
            if Point2((56.0, 52.0)).distance_to(anchor) <= 20:
                return local
            return far

        geysers.closest_to = MagicMock(side_effect=_closest)
        bot.vespene_geyser = geysers

        worker = MagicMock()
        worker.tag = 7
        worker.position = Point2((50.0, 50.0))
        worker.build = MagicMock(return_value=MagicMock())
        bot.workers = [worker]
        bot.do = MagicMock(return_value=True)
        bot.can_afford = MagicMock(return_value=True)
        bot.already_pending = MagicMock(return_value=0)
        bot.structures = MagicMock(return_value=MagicMock(exists=False))
        bot._placement_cell_legal = MagicMock(return_value=True)

        await bot.apply_build_action(np.array([0, 8 * 16 + 8], dtype=np.int64))
        assert worker.build.call_args[0][1] is local

    asyncio.run(_run())


def test_train_tech_ready_by_unit_type():
    bot = ZergBuildBot(training_conf={})

    def _structs(ready: bool):
        group = MagicMock()
        group.ready = MagicMock()
        group.ready.exists = ready
        return group

    # Drones / overlords: always ready
    assert bot._train_tech_ready(
        Command(command_type=CommandType.TRAIN, unit_type_id=UnitTypeId.DRONE)
    )
    assert bot._train_tech_ready(
        Command(command_type=CommandType.TRAIN, unit_type_id=UnitTypeId.OVERLORD)
    )

    # Zergling needs pool
    bot.structures = MagicMock(side_effect=lambda ut: _structs(ut == UnitTypeId.SPAWNINGPOOL))
    assert bot._train_tech_ready(
        Command(command_type=CommandType.TRAIN, unit_type_id=UnitTypeId.ZERGLING)
    )
    bot.structures = MagicMock(return_value=_structs(False))
    assert not bot._train_tech_ready(
        Command(command_type=CommandType.TRAIN, unit_type_id=UnitTypeId.ZERGLING)
    )

    # Roach needs warren (pool alone is not enough)
    bot.structures = MagicMock(
        side_effect=lambda ut: _structs(ut == UnitTypeId.SPAWNINGPOOL)
    )
    assert not bot._train_tech_ready(
        Command(command_type=CommandType.TRAIN, unit_type_id=UnitTypeId.ROACH)
    )
    bot.structures = MagicMock(
        side_effect=lambda ut: _structs(ut == UnitTypeId.ROACHWARREN)
    )
    assert bot._train_tech_ready(
        Command(command_type=CommandType.TRAIN, unit_type_id=UnitTypeId.ROACH)
    )


def test_apply_build_waits_without_penalty_when_unaffordable():
    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "BUILD",
                        "unit_type": "SPAWNINGPOOL",
                        "target_pos": [40.0, 40.0],
                    }
                ],
            }
        )
        bot.scripted_manager._emit_step_if_available()
        worker = MagicMock()
        worker.tag = 1
        worker.position = Point2((30.0, 30.0))
        bot.workers = [worker]
        bot.do = MagicMock()
        bot.can_afford = MagicMock(return_value=False)
        bot.already_pending = MagicMock(return_value=0)
        bot.structures = MagicMock(return_value=MagicMock(exists=False))

        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        bot.do.assert_not_called()
        assert bot._last_build_attempt_failed is False

    asyncio.run(_run())


def test_apply_standing_mine_assigns_n_nearest_workers():
    bot = ZergBuildBot(
        training_conf={"mining_controller": "scripted", "spawn": {"drones": 3}}
    )
    hatch = Point2((50.0, 50.0))
    bot._spawn_hatchery_pos = hatch
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.closest_to = MagicMock(return_value=MagicMock(position=hatch))
    bot.townhalls.first.position = hatch

    drones = []
    for i, dist in enumerate((5.0, 2.0, 9.0)):
        d = MagicMock()
        d.tag = i + 1
        d.distance_to = lambda p, _d=dist: _d
        drones.append(d)
    bot.workers = drones
    bot.command_queue.enqueue(
        Command(
            command_type=CommandType.MINE,
            target_pos=hatch,
            worker_count=2,
        )
    )
    bot._apply_standing_mine_command()
    assert bot._mine_anchor_pos == hatch
    assert bot.worker_registry.get_role(2) == WorkerRole.MINING  # closest
    assert bot.worker_registry.get_role(1) == WorkerRole.MINING
    assert bot.worker_registry.get_role(3) is None


def test_apply_standing_mine_gas_sets_quota():
    bot = ZergBuildBot(training_conf={"mining_controller": "speed"})
    hatch = Point2((50.0, 50.0))
    bot._spawn_hatchery_pos = hatch
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.closest_to = MagicMock(return_value=MagicMock(position=hatch))
    bot.command_queue.enqueue(
        Command(
            command_type=CommandType.MINE_GAS,
            target_pos=hatch,
            worker_count=3,
        )
    )
    bot._apply_standing_mine_gas_command()
    assert bot._gas_worker_quota == 3
    assert bot._gas_anchor_pos == hatch

    bot.command_queue.enqueue(
        Command(command_type=CommandType.MINE_GAS, target_pos=hatch, worker_count=0)
    )
    bot._apply_standing_mine_gas_command()
    assert bot._gas_worker_quota == 0


def test_standing_mine_does_not_steal_build_role_workers():
    """Regression: mining must not cancel an in-flight worker.build each step."""
    bot = ZergBuildBot(
        training_conf={"mining_controller": "speed", "spawn": {"drones": 3}}
    )
    hatch = Point2((50.0, 50.0))
    bot._spawn_hatchery_pos = hatch
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.closest_to = MagicMock(return_value=MagicMock(position=hatch))
    bot.townhalls.first.position = hatch

    drones = []
    for i, dist in enumerate((5.0, 2.0, 9.0)):
        d = MagicMock()
        d.tag = i + 1
        d.distance_to = lambda p, _d=dist: _d
        d.is_idle = False
        drones.append(d)
    bot.workers = drones
    bot.worker_registry.assign(2, WorkerRole.BUILD)  # builder (closest drone)
    bot.command_queue.enqueue(
        Command(
            command_type=CommandType.MINE,
            target_pos=hatch,
            worker_count=3,
        )
    )
    bot._apply_standing_mine_command()
    assert bot.worker_registry.get_role(2) == WorkerRole.BUILD
    assert bot.worker_registry.get_role(1) == WorkerRole.MINING
    assert bot.worker_registry.get_role(3) == WorkerRole.MINING
    assert 2 not in bot.worker_registry.mining_tags()


def test_ensure_training_mine_command_covers_all_workers():
    bot = ZergBuildBot(
        training_conf={"mining_controller": "scripted", "spawn": {"drones": 4}}
    )
    hatch = Point2((10.0, 10.0))
    bot._spawn_hatchery_pos = hatch
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.first.position = hatch
    bot.townhalls.closest_to = MagicMock(return_value=MagicMock(position=hatch))
    drones = []
    for tag in range(1, 5):
        d = MagicMock()
        d.tag = tag
        d.distance_to = lambda p: 1.0
        drones.append(d)
    bot.workers = drones
    assert bot.command_queue.standing_mine is None
    bot._ensure_training_mine_command()
    assert bot.command_queue.standing_mine is not None
    assert bot.command_queue.standing_mine.worker_count == 4
    bot._apply_standing_mine_command()
    for tag in range(1, 5):
        assert bot.worker_registry.get_role(tag) == WorkerRole.MINING


def test_assign_initial_worker_roles_splits_mining_and_build():
    bot = ZergBuildBot(
        training_conf={
            "spawn": {"drones": 3, "mining_drones": 2},
            "mining_controller": "scripted",
        }
    )
    drones = []
    for tag in (10, 20, 30):
        d = MagicMock()
        d.tag = tag
        drones.append(d)
    bot.workers = drones
    bot._assign_initial_worker_roles()
    assert bot.worker_registry.get_role(10) == WorkerRole.MINING
    assert bot.worker_registry.get_role(20) == WorkerRole.MINING
    assert bot.worker_registry.get_role(30) == WorkerRole.BUILD


def test_patch_script_targets_points_build_at_hatch():
    bot = ZergBuildBot(
        training_conf={
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "SPAWNINGPOOL",
                    "target_pos": [-1, -1],
                },
                {
                    "command_type": "TRAIN",
                    "unit_type": "ZERGLING",
                },
            ],
        }
    )
    hatch = Point2((55.0, 66.0))
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.first.position = hatch
    bot._patch_script_targets()
    assert bot.scripted_manager.script_steps[0].command.target_pos == hatch


def test_patch_script_targets_reanchors_after_hatch_moves():
    """Regression: soft-reset hatch move must update BUILD region, not keep ep1 coords."""
    bot = ZergBuildBot(
        training_conf={
            "manager_script": [
                {
                    "command_type": "MINE",
                    "target_pos": [-1, -1],
                    "worker_count": 8,
                },
                {
                    "command_type": "BUILD",
                    "unit_type": "SPAWNINGPOOL",
                    "target_pos": [-1, -1],
                },
            ],
        }
    )
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    hatch_a = Point2((20.0, 30.0))
    hatch_b = Point2((80.0, 90.0))
    bot.townhalls.first.position = hatch_a
    bot._patch_script_targets()
    assert bot.scripted_manager.script_steps[0].command.target_pos == hatch_a
    assert bot.scripted_manager.script_steps[1].command.target_pos == hatch_a

    bot.townhalls.first.position = hatch_b
    bot._spawn_hatchery_pos = hatch_b
    bot._patch_script_targets()
    assert bot.scripted_manager.script_steps[0].command.target_pos == hatch_b
    assert bot.scripted_manager.script_steps[1].command.target_pos == hatch_b


def test_patch_script_targets_keeps_explicit_coordinates():
    bot = ZergBuildBot(
        training_conf={
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "SPAWNINGPOOL",
                    "target_pos": [10.0, 12.0],
                },
            ],
        }
    )
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.first.position = Point2((55.0, 66.0))
    bot._patch_script_targets()
    assert bot.scripted_manager.script_steps[0].command.target_pos == Point2(
        (10.0, 12.0)
    )


def test_episode_complete_requires_full_command_script():
    from sc2.ids.unit_typeid import UnitTypeId
    from src.agent.commands import CommandType

    bot = ZergBuildBot(
        training_conf={
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "SPAWNINGPOOL",
                    "target_pos": [10, 10],
                },
                {
                    "command_type": "TRAIN",
                    "unit_type": "ZERGLING",
                    "count": 6,
                },
            ],
        }
    )
    # Completion-gated: only first macro is active; TRAIN is not pending yet.
    assert bot.scripted_manager.emit_next_if_idle() is True
    assert bot.command_queue.active is not None
    assert bot.command_queue.active.command_type == CommandType.BUILD
    assert bot.command_queue.peek_pending() == []
    assert bot._episode_complete() is False

    # Fake pool started (pending) → complete BUILD without waiting for .ready
    bot.already_pending = MagicMock(return_value=1)
    bot.structures = MagicMock(return_value=[])
    bot.units = MagicMock(return_value=MagicMock(amount=0))
    assert bot._try_complete_active_command() is True
    assert bot.command_queue.active.command_type == CommandType.TRAIN
    assert bot.command_queue.active.count == 6
    assert bot._episode_complete() is False

    # Pending morphs count toward TRAIN completion (pipeline while hatching).
    bot.already_pending = MagicMock(return_value=0)
    bot.units = MagicMock(return_value=MagicMock(amount=1))
    assert bot._try_complete_active_command() is False

    bot.already_pending = MagicMock(return_value=5)  # 1 alive + 5 pending
    bot.units = MagicMock(return_value=MagicMock(amount=1))
    # Use DRONE so pending is 1:1 (zergling eggs are ×2 equivalents).
    bot.command_queue.active.unit_type_id = UnitTypeId.DRONE
    bot.command_queue.active.count = 6
    bot._train_baseline = 0
    assert bot._try_complete_active_command() is True
    assert bot._episode_complete() is True


def test_build_completes_when_structure_started_not_only_ready():
    """Regression: next macro should start once the building morph/order begins."""
    bot = ZergBuildBot(
        training_conf={
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "SPAWNINGPOOL",
                    "target_pos": [10, 10],
                },
                {
                    "command_type": "TRAIN",
                    "unit_type": "ZERGLING",
                    "count": 2,
                },
            ],
        }
    )
    assert bot.scripted_manager.emit_next_if_idle() is True
    bot._spawn_hatchery_pos = Point2((10.0, 10.0))
    unfinished = MagicMock()
    unfinished.position = Point2((12.0, 12.0))
    unfinished.is_ready = False
    unfinished.distance_to = lambda p: Point2((12.0, 12.0)).distance_to(
        getattr(p, "position", p)
    )
    bot.structures = MagicMock(return_value=[unfinished])
    bot.already_pending = MagicMock(return_value=0)
    assert bot._try_complete_active_command() is True
    assert bot.command_queue.active.command_type == CommandType.TRAIN


def test_train_target_count_defaults_to_one():
    cmd = Command(command_type=CommandType.TRAIN, unit_type_id=None, count=None)
    assert ZergBuildBot._train_target_count(cmd) == 1
    cmd.count = 0
    assert ZergBuildBot._train_target_count(cmd) == 1
    cmd.count = 6
    assert ZergBuildBot._train_target_count(cmd) == 6


def test_apply_train_keeps_morphing_until_batch_count():
    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "TRAIN",
                        "unit_type": "ROACH",
                        "count": 3,
                    }
                ],
            }
        )
        bot.scripted_manager.emit_next_if_idle()
        bot._train_baseline = 0
        bot.do = MagicMock(return_value=True)
        bot.can_afford = MagicMock(return_value=True)
        pool_ready = MagicMock()
        pool_ready.exists = True
        bot.structures = MagicMock(return_value=pool_ready)
        larva_unit = MagicMock()
        larva_unit.train = MagicMock(return_value=MagicMock())
        larva = MagicMock()
        larva.exists = True
        larva.random = larva_unit
        bot.larva = larva
        bot.workers = [MagicMock(tag=1)]

        # 0 alive + 0 pending → morph
        bot.units = MagicMock(return_value=MagicMock(amount=0))
        bot.already_pending = MagicMock(return_value=0)
        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        bot.do.assert_called()
        bot.do.reset_mock()

        # 1 alive + 1 pending (=2) still under count=3 → morph again
        bot.units = MagicMock(return_value=MagicMock(amount=1))
        bot.already_pending = MagicMock(return_value=1)
        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        bot.do.assert_called()
        bot.do.reset_mock()

        # 2 alive + 1 pending (=3) meets count → stop
        bot.units = MagicMock(return_value=MagicMock(amount=2))
        bot.already_pending = MagicMock(return_value=1)
        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        bot.do.assert_not_called()

    asyncio.run(_run())


def test_apply_train_zergling_counts_two_units_per_egg():
    """Regression: count=6 must stop after 3 morphs, not 4 (which yields 8 lings)."""

    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "TRAIN",
                        "unit_type": "ZERGLING",
                        "count": 6,
                    }
                ],
            }
        )
        bot.scripted_manager.emit_next_if_idle()
        bot._train_baseline = 0
        bot.do = MagicMock(return_value=True)
        bot.can_afford = MagicMock(return_value=True)
        bot.structures = MagicMock(
            return_value=MagicMock(exists=True, ready=MagicMock(exists=True))
        )
        larva_unit = MagicMock()
        larva_unit.train = MagicMock(return_value=MagicMock())
        larva = MagicMock()
        larva.exists = True
        larva.random = larva_unit
        bot.larva = larva
        bot.workers = [MagicMock(tag=1)]
        bot.units = MagicMock(return_value=MagicMock(amount=0))

        # 0 eggs → morph #1
        bot.already_pending = MagicMock(return_value=0)
        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        assert bot.do.call_count == 1
        bot.do.reset_mock()

        # 1 egg (=2 units equiv) → morph #2
        bot.already_pending = MagicMock(return_value=1)
        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        assert bot.do.call_count == 1
        bot.do.reset_mock()

        # 2 eggs (=4) → morph #3
        bot.already_pending = MagicMock(return_value=2)
        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        assert bot.do.call_count == 1
        bot.do.reset_mock()

        # 3 eggs (=6) → stop (old bug issued a 4th morph → 8 lings)
        bot.already_pending = MagicMock(return_value=3)
        await bot.apply_build_action(np.array([0, 0], dtype=np.int64))
        bot.do.assert_not_called()

    asyncio.run(_run())


def test_reset_emits_only_first_macro_not_full_script():
    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "spawn": {"minerals": 0},
                "manager_script": [
                    {
                        "command_type": "MINE",
                        "target_pos": [-1, -1],
                        "worker_count": 8,
                    },
                    {
                        "command_type": "BUILD",
                        "unit_type": "SPAWNINGPOOL",
                        "target_pos": [-1, -1],
                    },
                    {
                        "command_type": "TRAIN",
                        "unit_type": "ZERGLING",
                        "count": 6,
                    },
                ],
            }
        )
        bot.client = AsyncMock()
        bot._ensure_map_visible = AsyncMock()
        bot._clear_debug_spawned_neutrals = AsyncMock()
        bot._clear_map_units = AsyncMock()
        bot._suppress_seat_filler = AsyncMock()
        bot._ensure_own_units_cleared = AsyncMock()
        bot._spawn_scenario = AsyncMock()
        bot._patch_script_targets = MagicMock()
        bot._kick_economy_orders = AsyncMock()
        bot.structures = MagicMock()
        bot.structures.__len__ = MagicMock(return_value=0)
        bot.minerals = 0
        bot.workers = []
        bot.townhalls = MagicMock()
        bot.townhalls.exists = False

        await bot.reset_game_state(skip_cleanup=True)

        assert bot.command_queue.standing_mine is not None
        assert bot.command_queue.active is not None
        assert bot.command_queue.active.command_type == CommandType.BUILD
        assert bot.command_queue.peek_pending() == []
        # TRAIN not emitted yet — still remaining in script
        assert bot.scripted_manager._script_index == 2
        assert len(bot.scripted_manager.script_steps) == 3

    asyncio.run(_run())


def test_candidate_base_locations_prefers_expansions_over_starts():
    bot = ZergBuildBot(training_conf={})
    start_a = Point2((20.0, 30.0))
    natural = Point2((45.0, 50.0))
    bot.game_info = MagicMock()
    bot.game_info.start_locations = [start_a]
    type(bot).enemy_start_locations = property(lambda self: [])
    type(bot).start_location = property(lambda self: start_a)
    type(bot).expansion_locations_list = property(
        lambda self: [start_a, natural]
    )

    locs = bot._candidate_base_locations()
    assert start_a in locs
    assert natural in locs


def test_candidate_base_locations_falls_back_to_starts_without_expansions():
    bot = ZergBuildBot(training_conf={})
    start_a = Point2((20.0, 30.0))
    start_b = Point2((100.0, 110.0))
    bot.game_info = MagicMock()
    bot.game_info.start_locations = [start_a, start_b]
    type(bot).enemy_start_locations = property(lambda self: [])
    type(bot).start_location = property(lambda self: None)
    type(bot).expansion_locations_list = property(lambda self: [])
    type(bot).expansion_locations_dict = property(lambda self: {})
    bot.mineral_field = []

    locs = bot._candidate_base_locations()
    assert start_a in locs
    assert start_b in locs


def test_resolve_hatch_spawn_pos_uses_base_not_map_center():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.map_name = "TestMap"
    bot.game_info.map_center = Point2((64.0, 64.0))
    bot.game_info.placement_grid = None
    bases = [Point2((20.0, 30.0)), Point2((80.0, 90.0))]
    expected = {Point2((20.5, 30.5)), Point2((80.5, 90.5))}
    bot._candidate_base_locations = MagicMock(return_value=bases)
    bot._map_start_locations = MagicMock(return_value=[])
    type(bot).expansion_locations_dict = property(lambda self: {})
    bot.vespene_geyser = MagicMock()
    bot.vespene_geyser.exists = False

    def _fields_for_base(pos):
        fields = []
        for _ in range(4):
            m = MagicMock()
            m.mineral_contents = 1500
            m.position = pos
            m.distance_to = lambda p, base=pos: abs(p.x - base.x) + abs(p.y - base.y)
            fields.append(m)
        return fields

    bot.mineral_field = _fields_for_base(Point2((20.0, 30.0))) + _fields_for_base(
        Point2((80.0, 90.0))
    )

    pos = bot._pick_episode_base_pos(rng=random.Random(0))
    assert pos in expected
    assert pos != bot.game_info.map_center


def test_pick_episode_base_does_not_fall_back_to_random_pathable():
    """Unpathable expansion center must still place at that base, not map wander."""
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.map_name = "TestMap"
    bot.game_info.start_locations = []
    bot.game_info.placement_grid = None
    base = Point2((20.0, 30.0))
    bot._candidate_base_locations = MagicMock(return_value=[base])
    bot._map_start_locations = MagicMock(return_value=[])
    bot.vespene_geyser = MagicMock()
    bot.vespene_geyser.exists = False
    bot._default_hatch_location = MagicMock(return_value=Point2((1.0, 1.0)))
    bot._find_buildable_near = MagicMock(return_value=Point2((1.0, 1.0)))
    bot._is_pathable = MagicMock(return_value=False)
    type(bot).enemy_start_locations = property(lambda self: [])
    type(bot).start_location = property(lambda self: None)
    type(bot).expansion_locations_dict = property(lambda self: {})

    fields = []
    for _ in range(4):
        m = MagicMock()
        m.mineral_contents = 1500
        m.position = base
        m.distance_to = lambda p, b=base: ((p.x - b.x) ** 2 + (p.y - b.y) ** 2) ** 0.5
        fields.append(m)
    bot.mineral_field = fields

    pos = bot._pick_episode_base_pos(rng=random.Random(0))
    assert pos == Point2((20.5, 30.5))  # half-tile snap of the base
    bot._default_hatch_location.assert_not_called()
    bot._find_buildable_near.assert_not_called()


def test_optimal_hatch_prefers_map_start_location():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    start = Point2((22.5, 31.5))
    near = Point2((23.0, 32.0))
    bot._map_start_locations = MagicMock(return_value=[start])
    bot._refine_hatch_on_placement_grid = MagicMock(
        return_value=Point2((99.5, 99.5))
    )
    assert bot._optimal_hatch_at_base(near) == start


def test_optimal_hatch_snaps_to_expansion_dict_key():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.placement_grid = None
    expansion = Point2((22.5, 31.5))
    near = Point2((23.0, 32.0))
    bot._map_start_locations = MagicMock(return_value=[])
    type(bot).expansion_locations_dict = property(
        lambda self: {expansion: []}
    )
    assert bot._optimal_hatch_at_base(near) == expansion


def test_half_tile_snaps_townhall_center():
    assert ZergBuildBot._half_tile(Point2((20.0, 30.0))) == Point2((20.5, 30.5))
    assert ZergBuildBot._half_tile(Point2((20.7, 30.2))) == Point2((20.5, 30.5))


def test_refine_hatch_picks_closest_legal_placement():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    # Allow all half-tile candidates on a large grid.
    grid = MagicMock()
    grid.__getitem__ = MagicMock(return_value=1)
    bot.game_info.placement_grid = grid

    resources = []
    for i in range(8):
        r = MagicMock()
        r.position = Point2((40.0 + i * 0.3, 40.0))
        r.distance_to = lambda p, pos=r.position: (
            ((pos.x - p.x) ** 2 + (pos.y - p.y) ** 2) ** 0.5
        )
        r._proto = MagicMock()
        r._proto.unit_type = 0  # mineral, not geyser
        r.name = "MineralField"
        # fix closure
        resources.append(r)
    # rebuild with proper closures
    resources = []
    for i in range(8):
        pos = Point2((40.0 + i * 0.3, 40.0))
        r = MagicMock()
        r.position = pos
        r.distance_to = lambda p, _pos=pos: (
            ((_pos.x - p.x) ** 2 + (_pos.y - p.y) ** 2) ** 0.5
        )
        r._proto = MagicMock()
        r._proto.unit_type = 0
        r.name = "MineralField"
        resources.append(r)

    refined = bot._refine_hatch_on_placement_grid(resources)
    assert refined is not None
    assert abs(refined.x % 1 - 0.5) < 1e-6
    assert abs(refined.y % 1 - 0.5) < 1e-6
    # Must stay outside the min mineral clearance of 6.
    assert all(refined.distance_to(r) >= 6.0 for r in resources)


def test_pick_episode_base_raises_when_no_healthy_minerals():
    from src.sc2_bot import MapCleanupFailedError

    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.map_name = "TestMap"
    bot._candidate_base_locations = MagicMock(return_value=[Point2((20.0, 30.0))])
    bot.mineral_field = []
    bot.vespene_geyser = MagicMock()
    bot.vespene_geyser.exists = False

    try:
        bot._pick_episode_base_pos(rng=random.Random(0))
        assert False, "expected MapCleanupFailedError"
    except MapCleanupFailedError:
        pass


def test_pick_episode_base_raises_when_no_candidates():
    from src.sc2_bot import MapCleanupFailedError

    bot = ZergBuildBot(training_conf={})
    bot._candidate_base_locations = MagicMock(return_value=[])
    try:
        bot._pick_episode_base_pos(rng=random.Random(0))
        assert False, "expected MapCleanupFailedError"
    except MapCleanupFailedError:
        pass


def test_pick_episode_base_is_stochastic_across_seeds():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.map_name = "TestMap"
    bases = [Point2((float(i * 10), float(i * 10))) for i in range(1, 6)]
    bot._candidate_base_locations = MagicMock(return_value=bases)
    bot.vespene_geyser = MagicMock()
    bot.vespene_geyser.exists = False

    fields = []
    for base in bases:
        for _ in range(4):
            m = MagicMock()
            m.mineral_contents = 1500
            m.distance_to = lambda p, b=base: ((p.x - b.x) ** 2 + (p.y - b.y) ** 2) ** 0.5
            fields.append(m)
    bot.mineral_field = fields

    picks = {bot._pick_episode_base_pos(rng=random.Random(s)) for s in range(20)}
    assert len(picks) > 1


def test_bases_from_mineral_clusters_offsets_toward_map_center():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.map_center = Point2((100.0, 100.0))

    fields = []
    for i in range(8):
        pos = Point2((10.0 + i * 0.5, 10.0))
        m = MagicMock()
        m.position = pos
        m.distance_to = lambda p, _pos=pos: (
            ((_pos.x - p.x) ** 2 + (_pos.y - p.y) ** 2) ** 0.5
        )
        fields.append(m)
    bot.mineral_field = fields

    centers = bot._bases_from_mineral_clusters()
    assert len(centers) == 1
    # Hatch should be pulled toward map center relative to mineral centroid.
    assert centers[0].x > 10.0
    assert centers[0].y > 10.0


def test_mining_reward_bonus_and_idle_penalty():
    bot = ZergBuildBot(
        training_conf={
            "rewards": {"mining_active_bonus": 0.02, "idle_miner_penalty": -0.05}
        }
    )
    active = MagicMock()
    active.tag = 1
    active.is_gathering = True
    active.is_carrying_minerals = False
    active.is_carrying_vespene = False
    active.is_idle = False
    idle = MagicMock()
    idle.tag = 2
    idle.is_gathering = False
    idle.is_carrying_minerals = False
    idle.is_carrying_vespene = False
    idle.is_idle = True
    bot.workers = [active, idle]
    bot.worker_registry.assign(1, WorkerRole.MINING)
    bot.worker_registry.assign(2, WorkerRole.MINING)

    r = bot._compute_mining_reward()
    assert abs(r - (0.02 - 0.05)) < 1e-6


def test_assert_banked_minerals_tolerates_leftover_bank_for_soft_reset():
    """Leftover bank must not raise — soft resets cannot zero minerals."""
    bot = ZergBuildBot(training_conf={"spawn": {"minerals": 0}})
    bot.minerals = 400
    bot._assert_banked_minerals_match_spawn()  # no raise


def test_assert_banked_minerals_allows_default_opening_bank():
    bot = ZergBuildBot(training_conf={"spawn": {"minerals": 0}})
    bot.minerals = 50
    bot._assert_banked_minerals_match_spawn()


def test_soft_reset_does_not_assert_mineral_bank():
    async def _run():
        bot = ZergBuildBot(training_conf={"spawn": {"minerals": 0}})
        bot.client = AsyncMock()
        bot._ensure_map_visible = AsyncMock()
        bot._request_map_soft_reset_resources = AsyncMock()
        bot._clear_debug_spawned_neutrals = AsyncMock()
        bot._clear_map_units = AsyncMock()
        bot._suppress_seat_filler = AsyncMock()
        bot._ensure_own_units_cleared = AsyncMock()
        bot._spawn_scenario = AsyncMock()
        bot._patch_script_targets = MagicMock()
        bot.scripted_manager.emit_next_if_idle = MagicMock(return_value=False)
        bot._kick_economy_orders = AsyncMock()
        bot._assert_banked_minerals_match_spawn = MagicMock()
        bot.structures = MagicMock()
        bot.structures.__len__ = MagicMock(return_value=0)
        bot.minerals = 900  # leftover bank from prior episode

        await bot.reset_game_state(skip_cleanup=False)

        bot._assert_banked_minerals_match_spawn.assert_not_called()
        bot._request_map_soft_reset_resources.assert_awaited_once()
        bot._clear_map_units.assert_awaited_once_with(strict_empty_map=True)
        bot._spawn_scenario.assert_awaited_once()

    asyncio.run(_run())


def test_soft_reset_sends_soft_reset_chat():
    async def _run():
        bot = ZergBuildBot(training_conf={"spawn": {"minerals": 0}})
        bot.client = AsyncMock()
        bot.chat_send = AsyncMock()
        bot._refresh_state = AsyncMock()
        await bot._request_map_soft_reset_resources()
        bot.chat_send.assert_awaited_once_with("SOFT_RESET")
        assert bot.client.step.await_count == 2
        bot._refresh_state.assert_awaited_once()

    asyncio.run(_run())


def test_fresh_session_reset_skips_soft_reset_chat():
    async def _run():
        bot = ZergBuildBot(training_conf={"spawn": {"minerals": 0}})
        bot.client = AsyncMock()
        bot._ensure_map_visible = AsyncMock()
        bot._request_map_soft_reset_resources = AsyncMock()
        bot._clear_map_units = AsyncMock()
        bot._suppress_seat_filler = AsyncMock()
        bot._spawn_scenario = AsyncMock()
        bot._patch_script_targets = MagicMock()
        bot.scripted_manager.emit_next_if_idle = MagicMock(return_value=False)
        bot._kick_economy_orders = AsyncMock()
        bot.structures = MagicMock()
        bot.structures.__len__ = MagicMock(return_value=0)

        await bot.reset_game_state(skip_cleanup=True)

        bot._request_map_soft_reset_resources.assert_not_called()
        # Still wipe melee openers / Computer seat filler on a fresh session.
        bot._clear_map_units.assert_awaited_once_with(strict_empty_map=False)
        bot._suppress_seat_filler.assert_awaited_once()

    asyncio.run(_run())


def test_reset_game_state_ensures_map_visible_once():
    async def _run():
        bot = ZergBuildBot(training_conf={"spawn": {"minerals": 0}})
        bot.client = AsyncMock()
        bot._refresh_state = AsyncMock()
        bot._clear_map_units = AsyncMock()
        bot._suppress_seat_filler = AsyncMock()
        bot._spawn_scenario = AsyncMock()
        bot._patch_script_targets = MagicMock()
        bot._assign_initial_worker_roles = MagicMock()
        bot.scripted_manager.emit_next_if_idle = MagicMock(return_value=False)
        bot._kick_economy_orders = AsyncMock()
        bot.structures = MagicMock()
        bot.structures.__len__ = MagicMock(return_value=0)
        bot.minerals = 0

        await bot.reset_game_state(skip_cleanup=True)
        bot.client.debug_show_map.assert_awaited_once()
        assert bot._debug_show_map_enabled is True

        bot.client.debug_show_map.reset_mock()
        await bot.reset_game_state(skip_cleanup=True)
        bot.client.debug_show_map.assert_not_awaited()

    asyncio.run(_run())


def test_grant_spawn_resources_uses_debug_minerals():
    async def _run():
        bot = ZergBuildBot(training_conf={})
        bot.client = AsyncMock()
        bot._refresh_state = AsyncMock()
        await bot._grant_spawn_resources(500, 0)
        bot.client.debug_minerals.assert_awaited_once()
        bot.client.debug_gas.assert_not_awaited()

    asyncio.run(_run())


def test_spawn_scenario_places_hatch_at_resolved_base():
    async def _run():
        bot = ZergBuildBot(training_conf={"env": {"spawn": {"drones": 1, "minerals": 0}}})
        bot.client = AsyncMock()
        bot.player_id = 1
        bot.game_info = MagicMock()
        bot.game_info.playable_area.contains = lambda pos: True
        grid = np.ones((128, 128), dtype=np.uint8)
        bot.game_info.pathing_grid = MagicMock()
        bot.game_info.pathing_grid.__getitem__ = lambda self, key: grid[key[1], key[0]]
        bot.game_info.playable_area.x = 0
        bot.game_info.playable_area.y = 0
        bot.game_info.playable_area.width = 128
        bot.game_info.playable_area.height = 128
        bot.game_info._proto = MagicMock()

        geyser = MagicMock()
        geyser.position = Point2((40.0, 40.0))
        bot.vespene_geyser = MagicMock()
        bot.vespene_geyser.exists = True
        bot.vespene_geyser.closest_to.return_value = geyser
        hatch_pos = Point2((34.0, 34.0))
        bot._resolve_hatch_spawn_pos = MagicMock(return_value=hatch_pos)
        bot._spawn_drones_near_hatch = AsyncMock()
        bot._grant_spawn_resources = AsyncMock()
        bot._refresh_state = AsyncMock()
        await bot._spawn_scenario()

        create_calls = bot.client.debug_create_unit.call_args_list
        assert any(
            c[0][0][0][0].name == "HATCHERY" and c[0][0][0][2] == hatch_pos
            for c in create_calls
        )
        bot._spawn_drones_near_hatch.assert_awaited_once()
        bot._grant_spawn_resources.assert_awaited_once_with(0, 0)

    asyncio.run(_run())


def test_reset_game_state_clears_spawn_session_cache():
    async def _run():
        bot = ZergBuildBot(training_conf={"spawn": {"minerals": 0}})
        bot._cached_hatch_spawn_pos = Point2((12.0, 13.0))
        bot._cached_spawn_geometry_key = "cached"
        bot._target_geyser_pos = Point2((40.0, 40.0))
        bot._spawn_hatchery_pos = Point2((12.0, 13.0))
        bot._debug_spawned_neutral_tags = {99, 100}

        bot.client = AsyncMock()
        bot._ensure_map_visible = AsyncMock()
        bot._request_map_soft_reset_resources = AsyncMock()
        bot._clear_debug_spawned_neutrals = AsyncMock()
        bot._clear_map_units = AsyncMock()
        bot._suppress_seat_filler = AsyncMock()
        bot._ensure_own_units_cleared = AsyncMock()
        bot._spawn_scenario = AsyncMock()
        bot._patch_script_targets = MagicMock()
        bot._assign_initial_worker_roles = MagicMock()
        bot.scripted_manager.emit_next_if_idle = MagicMock(return_value=False)
        bot._kick_economy_orders = AsyncMock()
        bot.structures = MagicMock()
        bot.structures.__len__ = MagicMock(return_value=0)
        bot.minerals = 0

        await bot.reset_game_state(skip_cleanup=False)

        assert bot._cached_hatch_spawn_pos is None
        assert bot._cached_spawn_geometry_key is None
        assert bot._target_geyser_pos is None
        assert bot._spawn_hatchery_pos is None
        bot._clear_debug_spawned_neutrals.assert_awaited_once()
        bot._ensure_own_units_cleared.assert_awaited_once()
        bot._clear_map_units.assert_awaited_once_with(strict_empty_map=True)

    asyncio.run(_run())


def test_spawn_scenario_spawns_resources_when_map_has_no_geysers():
    async def _run():
        bot = ZergBuildBot(training_conf={"env": {"spawn": {"drones": 1, "minerals": 0}}})
        bot.client = AsyncMock()
        bot.player_id = 1
        bot.game_info = MagicMock()
        bot.game_info.playable_area.contains = lambda pos: True
        grid = np.ones((64, 64), dtype=np.uint8)
        bot.game_info.pathing_grid = MagicMock()
        bot.game_info.pathing_grid.__getitem__ = lambda self, key: grid[key[1], key[0]]
        bot.game_info.playable_area.x = 0
        bot.game_info.playable_area.y = 0
        bot.game_info.playable_area.width = 64
        bot.game_info.playable_area.height = 64
        bot.game_info._proto = MagicMock()
        bot.vespene_geyser = MagicMock()
        bot.vespene_geyser.exists = False
        bot._resolve_hatch_spawn_pos = MagicMock(return_value=Point2((20.0, 20.0)))
        bot._spawn_drones_near_hatch = AsyncMock()
        bot._grant_spawn_resources = AsyncMock()
        bot._refresh_state = AsyncMock()
        bot._tag_debug_spawned_neutrals = MagicMock()
        await bot._spawn_scenario()

        created_types = [c[0][0][0][0].name for c in bot.client.debug_create_unit.call_args_list]
        assert "VESPENEGEYSER" in created_types
        assert "MINERALFIELD" in created_types
        assert bot._target_geyser_pos is not None
        bot._grant_spawn_resources.assert_awaited_once_with(0, 0)

    asyncio.run(_run())


def test_clear_map_units_kills_self_extractor_despite_vespene_flag():
    """Regression: leftover Extractors were skipped because is_vespene_geyser=True."""

    async def _run():
        from sc2.data import Alliance

        bot = ZergBuildBot(training_conf={})
        bot.client = AsyncMock()

        extractor = MagicMock()
        extractor.tag = 77
        extractor.alliance = Alliance.Self
        extractor.type_id = UnitTypeId.EXTRACTOR
        extractor.is_mineral_field = False
        # burnysc2 marks gas buildings as vespene geysers.
        extractor.is_vespene_geyser = True
        extractor.owner_id = 1

        killable = MagicMock()
        killable.amount = 1
        killable.__iter__ = lambda self: iter([extractor])
        empty = _EmptyKillable()

        def filter_side_effect(pred):
            assert pred(extractor) is True
            if bot.all_units.filter.call_count <= 1:
                return killable
            return empty

        bot.all_units = MagicMock()
        bot.all_units.filter = MagicMock(side_effect=filter_side_effect)
        bot._refresh_state = AsyncMock()

        await bot._clear_map_units()
        bot.client.debug_kill_unit.assert_awaited()
        assert 77 in bot.client.debug_kill_unit.await_args[0][0]

    asyncio.run(_run())


def test_build_complete_ignores_far_leftover_extractor():
    """Leftover ep1 Extractor must not complete BUILD at a new hatch."""
    bot = ZergBuildBot(
        training_conf={
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "EXTRACTOR",
                    "target_pos": [-1, -1],
                }
            ]
        }
    )
    bot.scripted_manager._emit_step_if_available()
    bot._spawn_hatchery_pos = Point2((150.0, 150.0))

    leftover = MagicMock()
    leftover.position = Point2((50.0, 50.0))
    leftover.is_ready = True
    leftover.distance_to = lambda p: Point2((50.0, 50.0)).distance_to(
        getattr(p, "position", p)
    )

    ready = MagicMock()
    ready.exists = True
    structs = MagicMock()
    structs.exists = True
    structs.ready = ready
    structs.__iter__ = lambda self: iter([leftover])
    bot.structures = MagicMock(return_value=structs)

    assert bot._has_structure_near_episode_hatch(UnitTypeId.EXTRACTOR) is False
    assert bot._try_complete_active_command() is False
    assert bot.command_queue.active is not None


def test_clear_map_units_kills_self_and_enemy_seat_filler():
    async def _run():
        from sc2.data import Alliance

        bot = ZergBuildBot(training_conf={})
        bot.client = AsyncMock()

        unit = MagicMock()
        unit.tag = 99
        unit.alliance = Alliance.Self
        unit.type_id = UnitTypeId.HATCHERY
        unit.is_mineral_field = False
        unit.is_vespene_geyser = False
        unit.owner_id = 1

        killable = MagicMock()
        killable.amount = 1
        killable.__iter__ = lambda self: iter([unit])

        empty = _EmptyKillable()

        def filter_side_effect(pred):
            if bot.all_units.filter.call_count <= 1:
                return killable
            return empty

        bot.all_units = MagicMock()
        bot.all_units.filter = MagicMock(side_effect=filter_side_effect)
        bot.player_id = 1
        bot._refresh_state = AsyncMock()

        await bot._clear_map_units()

        bot.client.debug_kill_unit.assert_awaited()
        assert bot.client.debug_kill_unit.await_args[0][0] == [99]
        first_pred = bot.all_units.filter.call_args_list[0][0][0]
        assert first_pred(unit) is True
        # Computer seat filler must also be cleared (ladder 2nd player).
        enemy = MagicMock()
        enemy.alliance = Alliance.Enemy
        enemy.type_id = UnitTypeId.COMMANDCENTER
        enemy.is_mineral_field = False
        enemy.is_vespene_geyser = False
        assert first_pred(enemy) is True
        # Regression: raw proto int 3 = Neutral (minerals) must NOT be killed.
        mineral = MagicMock()
        mineral.alliance = 3
        mineral.type_id = UnitTypeId.MINERALFIELD
        mineral.is_mineral_field = True
        mineral.is_vespene_geyser = False
        mineral.owner_id = 16
        assert first_pred(mineral) is False
        mineral_int_only = MagicMock()
        mineral_int_only.alliance = 3
        mineral_int_only.type_id = UnitTypeId.MINERALFIELD
        mineral_int_only.is_mineral_field = False
        mineral_int_only.is_vespene_geyser = False
        mineral_int_only.owner_id = 16
        assert first_pred(mineral_int_only) is False

    asyncio.run(_run())


def test_suppress_seat_filler_kills_enemy_units_and_structures():
    async def _run():
        bot = ZergBuildBot(training_conf={})
        bot.client = AsyncMock()
        eu = MagicMock(tag=11)
        es = MagicMock(tag=22)
        bot.enemy_units = [eu]
        bot.enemy_structures = [es]
        await bot._suppress_seat_filler()
        bot.client.debug_kill_unit.assert_awaited_once_with([11, 22])

    asyncio.run(_run())
