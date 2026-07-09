from unittest.mock import MagicMock

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId

from src.agent.build_constants import BUILD_PLACEMENT_CELLS, MAX_BUILD_WORKERS
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
    assert mask[MAX_BUILD_WORKERS:].all()


def test_build_action_masks_all_false_when_no_workers():
    bot = ZergBuildBot(training_conf={})
    bot.workers = []
    mask = bot.action_masks()
    assert not mask[:MAX_BUILD_WORKERS].any()
    assert mask[MAX_BUILD_WORKERS:].all()
