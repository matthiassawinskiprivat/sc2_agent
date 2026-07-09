import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
from sc2.position import Point2

from src.agent.build_constants import MAX_BUILD_WORKERS
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


def test_action_masks_mask_unused_worker_slots():
    bot = ZergBuildBot(training_conf={})
    drone = MagicMock()
    drone.tag = 7
    bot.workers = [drone]
    mask = bot.action_masks()
    assert mask.dtype == bool
    assert mask[0]
    assert not mask[1:MAX_BUILD_WORKERS].any()
    assert mask[MAX_BUILD_WORKERS:].all()


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


def test_scripted_manager_reads_flattened_manager_script():
    bot = ZergBuildBot(
        training_conf={
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "EXTRACTOR",
                    "target_pos": [0, 0],
                    "urgent": True,
                }
            ],
            "macro_tick_interval": 8,
            "env": {"realtime": False},
        }
    )
    assert len(bot.scripted_manager.script_steps) == 1
    bot.scripted_manager._emit_step_if_available()
    assert bot.command_queue.has_active_macro()
    assert bot.command_queue.active.unit_type_id.name == "EXTRACTOR"


def test_apply_build_action_issues_order_when_macro_command_active():
    async def _run():
        from sc2.ids.unit_typeid import UnitTypeId
        from src.agent.commands import Command, CommandType

        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "BUILD",
                        "unit_type": "EXTRACTOR",
                        "target_pos": [40.0, 40.0],
                        "urgent": True,
                    }
                ],
            }
        )
        bot.scripted_manager._emit_step_if_available()
        bot._patch_script_targets = MagicMock()

        worker = MagicMock()
        worker.tag = 42
        worker.position = Point2((30.0, 30.0))
        bot.workers = [worker]
        bot.do = MagicMock(return_value=True)
        bot._nearest_geyser = MagicMock(return_value=MagicMock())

        applied = await bot.apply_build_action(np.array([0, 8 * 16 + 8], dtype=np.int64))
        assert applied is True
        bot.do.assert_called_once()

    asyncio.run(_run())


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


def test_resolve_hatch_spawn_pos_reuses_cached_position():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.map_name = "TestMap"
    bot.vespene_geyser = MagicMock()
    bot.vespene_geyser.exists = False
    bot._default_hatch_location = MagicMock(return_value=Point2((12.0, 13.0)))
    bot._spawn_reference_point = MagicMock(return_value=Point2((0.0, 0.0)))
    bot._is_pathable = MagicMock(return_value=True)

    first = bot._resolve_hatch_spawn_pos()
    bot._default_hatch_location.reset_mock()
    second = bot._resolve_hatch_spawn_pos()

    assert first == Point2((12.0, 13.0))
    assert second == first
    bot._default_hatch_location.assert_not_called()


def test_reset_game_state_ensures_map_visible_once():
    async def _run():
        bot = ZergBuildBot(training_conf={})
        bot.client = AsyncMock()
        bot._refresh_state = AsyncMock()
        bot._clear_map_units = AsyncMock()
        bot._spawn_scenario = AsyncMock()
        bot._patch_script_targets = MagicMock()
        bot._assign_initial_worker_roles = MagicMock()
        bot.scripted_manager._emit_step_if_available = MagicMock()
        bot._kick_economy_orders = AsyncMock()
        bot.structures = MagicMock()
        bot.structures.__len__ = MagicMock(return_value=0)

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


def test_spawn_scenario_uses_map_geyser_when_present():
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
        bot._spawn_reference_point = MagicMock(return_value=Point2((10.0, 10.0)))
        bot._spawn_drones_near_hatch = AsyncMock()
        bot._grant_spawn_resources = AsyncMock()

        hatch_pos = Point2((34.0, 34.0))
        bot._find_buildable_near = MagicMock(return_value=hatch_pos)
        bot._refresh_state = AsyncMock()
        await bot._spawn_scenario()

        bot.vespene_geyser.closest_to.assert_called()
        bot._find_buildable_near.assert_called_with(geyser.position)
        assert bot._target_geyser_pos == geyser.position
        bot._spawn_drones_near_hatch.assert_awaited_once()
        bot._grant_spawn_resources.assert_awaited_once_with(0, 0)

    asyncio.run(_run())


def test_reset_game_state_clears_spawn_session_cache():
    async def _run():
        bot = ZergBuildBot(training_conf={})
        bot._cached_hatch_spawn_pos = Point2((12.0, 13.0))
        bot._cached_spawn_geometry_key = "cached"
        bot._target_geyser_pos = Point2((40.0, 40.0))
        bot._spawn_hatchery_pos = Point2((12.0, 13.0))
        bot._debug_spawned_neutral_tags = {99, 100}

        bot.client = AsyncMock()
        bot._ensure_map_visible = AsyncMock()
        bot._clear_debug_spawned_neutrals = AsyncMock()
        bot._clear_map_units = AsyncMock()
        bot._ensure_own_workers_cleared = AsyncMock()
        bot._spawn_scenario = AsyncMock()
        bot._patch_script_targets = MagicMock()
        bot._assign_initial_worker_roles = MagicMock()
        bot.scripted_manager._emit_step_if_available = MagicMock()
        bot._kick_economy_orders = AsyncMock()
        bot.structures = MagicMock()
        bot.structures.__len__ = MagicMock(return_value=0)

        await bot.reset_game_state(skip_cleanup=False)

        assert bot._cached_hatch_spawn_pos is None
        assert bot._cached_spawn_geometry_key is None
        assert bot._target_geyser_pos is None
        assert bot._spawn_hatchery_pos is None
        bot._clear_debug_spawned_neutrals.assert_awaited_once()
        bot._ensure_own_workers_cleared.assert_awaited_once()

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
        bot._default_hatch_location = MagicMock(return_value=Point2((20.0, 20.0)))
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


def test_clear_map_units_kills_killable_units():
    async def _run():
        bot = ZergBuildBot(training_conf={})
        bot.client = AsyncMock()

        unit = MagicMock()
        unit.tag = 99
        unit.alliance.name = "Self"
        unit.type_id.name = "HATCHERY"
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
        unit.owner_id = 2
        assert first_pred(unit) is False

    asyncio.run(_run())
