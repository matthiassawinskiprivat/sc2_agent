"""Build agent action masks are vacuous (always True) for MaskablePPO."""

import numpy as np
from unittest.mock import MagicMock

from src.agent.build_constants import BUILD_PLACEMENT_CELLS, MAX_BUILD_WORKERS
from src.zerg_build_bot import ZergBuildBot


def test_build_action_masks_always_all_true_with_workers():
    bot = ZergBuildBot(training_conf={})
    w1 = MagicMock()
    w1.tag = 1
    w2 = MagicMock()
    w2.tag = 2
    bot.workers = [w1, w2]
    mask = bot.action_masks()
    assert mask.dtype == bool
    assert mask.shape == (MAX_BUILD_WORKERS + BUILD_PLACEMENT_CELLS,)
    assert mask.all()


def test_build_action_masks_always_all_true_with_no_workers():
    bot = ZergBuildBot(training_conf={})
    bot.workers = []
    mask = bot.action_masks()
    assert mask.all()
