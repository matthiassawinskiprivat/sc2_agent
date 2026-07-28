from unittest.mock import MagicMock

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.build_constants import BUILD_PLACEMENT_CELLS, MAX_BUILD_WORKERS
from src.agent.commands import Command, CommandType
from src.zerg_build_bot import ZergBuildBot


def test_build_action_masks_hide_padded_worker_slots():
    bot = ZergBuildBot(training_conf={})
    w1 = MagicMock()
    w1.tag = 1
    w2 = MagicMock()
    w2.tag = 2
    bot.workers = [w1, w2]
    mask = bot.action_masks()
    assert len(mask) == MAX_BUILD_WORKERS + BUILD_PLACEMENT_CELLS
    assert mask[:2].all()
    assert not mask[2:MAX_BUILD_WORKERS].any()


def test_build_action_masks_all_false_workers_when_none():
    bot = ZergBuildBot(training_conf={})
    bot.workers = []
    mask = bot.action_masks()
    assert not mask[:MAX_BUILD_WORKERS].any()


def test_build_placement_mask_requires_pathable_creep_for_pool():
    bot = ZergBuildBot(training_conf={})
    bot.workers = []
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
    grid[33, 32] = 1
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

    placement = bot._placement_action_mask()
    assert placement.dtype == bool
    assert placement.any()
    assert not placement.all()
    # Center cell (8,8) -> world ~32.5,32.5 -> pathable+creep at (32,32)
    center_idx = 8 * 16 + 8
    assert placement[center_idx]
