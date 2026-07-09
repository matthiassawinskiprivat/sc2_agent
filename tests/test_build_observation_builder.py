from unittest.mock import MagicMock

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.commands import Command, CommandType
from src.runtime.worker_registry import WorkerRegistry, WorkerRole
from src.transformers.build_observation_builder import BuildObservationBuilder


def _mock_bot():
    bot = MagicMock()
    path = np.ones((64, 64), dtype=np.uint8)
    height = np.full((64, 64), 8.0, dtype=np.float32)
    bot.game_info.pathing_grid.data_numpy = path
    bot.game_info.terrain_height.data_numpy = height
    bot.game_info.placement_grid.data_numpy = path
    bot.game_info.map_name = "test_map"
    bot.game_info.playable_area.width = 64
    bot.game_info.playable_area.height = 64
    bot.game_data = MagicMock()
    bot.state.creep = None
    bot.workers = []
    bot.structures = []
    bot.vespene_geyser = []
    bot.townhalls.exists = False
    bot.start_location = Point2((32.0, 32.0))
    bot.minerals = 100
    bot.vespene = 0
    bot.supply_used = 10
    bot.supply_cap = 15
    bot.supply_army = 0
    bot.game_loop = 100
    return bot


def test_build_worker_features_use_cargo_used():
    builder = BuildObservationBuilder({"race": "zerg"})
    bot = _mock_bot()
    worker = MagicMock()
    worker.tag = 42
    worker.position.x = 10.0
    worker.position.y = 20.0
    worker.health = 40.0
    worker.health_max = 40.0
    worker.is_idle = False
    worker.cargo_used = 4
    bot.workers = [worker]
    reg = WorkerRegistry()
    reg.assign(42, WorkerRole.BUILD)
    feats = builder._build_worker_features(bot, reg, (64.0, 64.0))
    assert feats[0, 4] == 0.5


def test_build_observation_builder_local_map_shape():
    builder = BuildObservationBuilder(
        {
            "build_map_layers": [
                "terrain_pathable",
                "terrain_creep",
                "terrain_buildable",
            ],
            "race": "zerg",
        }
    )
    cmd = Command(
        command_type=CommandType.BUILD,
        unit_type_id=UnitTypeId.EXTRACTOR,
        target_pos=Point2((32.0, 32.0)),
    )
    obs = builder.build(_mock_bot(), WorkerRegistry(), cmd, (64.0, 64.0))
    assert obs["local_map"].shape == (3, 16, 16)
    assert obs["worker_features"].shape[0] == 24
    assert obs["macro_vector"].shape == (8,)
    assert obs["command"].shape == (12,)
