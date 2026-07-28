import asyncio
import random
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
from sc2.position import Point2

from src.agent.build_constants import MAX_BUILD_WORKERS
from src.agent.commands import Command, CommandType
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

    asyncio.run(_run())


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

    # Fake pool ready → complete BUILD, emit TRAIN batch
    ready = MagicMock()
    ready.exists = True
    bot.structures = MagicMock(return_value=MagicMock(ready=ready))
    bot.units = MagicMock(return_value=MagicMock(amount=0))
    assert bot._try_complete_active_command() is True
    assert bot.command_queue.active.command_type == CommandType.TRAIN
    assert bot.command_queue.active.count == 6
    assert bot._episode_complete() is False

    # One ling is not enough for count=6
    bot.units = MagicMock(return_value=MagicMock(amount=1))
    assert bot._try_complete_active_command() is False
    assert bot._episode_complete() is False

    bot.units = MagicMock(return_value=MagicMock(amount=6))
    assert bot._try_complete_active_command() is True
    assert bot._episode_complete() is True


def test_train_target_count_defaults_to_one():
    cmd = Command(command_type=CommandType.TRAIN, unit_type_id=None, count=None)
    assert ZergBuildBot._train_target_count(cmd) == 1
    cmd.count = 0
    assert ZergBuildBot._train_target_count(cmd) == 1
    cmd.count = 6
    assert ZergBuildBot._train_target_count(cmd) == 6


def test_apply_train_keeps_morphing_until_batch_count():
    async def _run():
        from sc2.ids.unit_typeid import UnitTypeId

        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "TRAIN",
                        "unit_type": "ZERGLING",
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


def test_candidate_base_locations_prefers_start_locations_not_expansions():
    bot = ZergBuildBot(training_conf={})
    start_a = Point2((20.0, 30.0))
    start_b = Point2((100.0, 110.0))
    natural = Point2((45.0, 50.0))  # expansion, must not be preferred
    bot.game_info = MagicMock()
    bot.game_info.start_locations = [start_a, start_b]
    # BotAI properties — override via type so we do not pull expansions in.
    type(bot).enemy_start_locations = property(lambda self: [])
    type(bot).start_location = property(lambda self: None)
    type(bot).expansion_locations_list = property(
        lambda self: [start_a, natural, start_b]
    )

    locs = bot._candidate_base_locations()
    assert start_a in locs
    assert start_b in locs
    assert natural not in locs


def test_resolve_hatch_spawn_pos_uses_start_not_map_center():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.map_name = "TestMap"
    bot.game_info.map_center = Point2((64.0, 64.0))
    starts = [Point2((20.0, 30.0)), Point2((80.0, 90.0))]
    bot._candidate_base_locations = MagicMock(return_value=starts)
    bot.vespene_geyser = MagicMock()
    bot.vespene_geyser.exists = False

    def _fields_for_base(pos):
        fields = []
        for _ in range(4):
            m = MagicMock()
            m.mineral_contents = 1500
            m.distance_to = lambda p, base=pos: abs(p.x - base.x) + abs(p.y - base.y)
            fields.append(m)
        return fields

    bot.mineral_field = _fields_for_base(Point2((20.0, 30.0))) + _fields_for_base(
        Point2((80.0, 90.0))
    )
    bot._is_pathable = MagicMock(return_value=True)

    pos = bot._pick_episode_base_pos(rng=random.Random(0))
    assert pos in starts
    assert pos != bot.game_info.map_center


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


def test_pick_episode_base_is_stochastic_across_seeds():
    bot = ZergBuildBot(training_conf={})
    bot.game_info = MagicMock()
    bot.game_info.map_name = "TestMap"
    bases = [Point2((float(i * 10), float(i * 10))) for i in range(1, 6)]
    bot._candidate_base_locations = MagicMock(return_value=bases)
    bot.vespene_geyser = MagicMock()
    bot.vespene_geyser.exists = False
    bot._is_pathable = MagicMock(return_value=True)

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
        bot._spawn_scenario = AsyncMock()
        bot._patch_script_targets = MagicMock()
        bot.scripted_manager.emit_next_if_idle = MagicMock(return_value=False)
        bot._kick_economy_orders = AsyncMock()
        bot.structures = MagicMock()
        bot.structures.__len__ = MagicMock(return_value=0)

        await bot.reset_game_state(skip_cleanup=True)

        bot._request_map_soft_reset_resources.assert_not_called()

    asyncio.run(_run())


def test_reset_game_state_ensures_map_visible_once():
    async def _run():
        bot = ZergBuildBot(training_conf={"spawn": {"minerals": 0}})
        bot.client = AsyncMock()
        bot._refresh_state = AsyncMock()
        bot._clear_map_units = AsyncMock()
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
