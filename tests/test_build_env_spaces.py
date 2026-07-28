from unittest.mock import MagicMock, patch

import numpy as np
from gymnasium import spaces

from src.agent.build_constants import (
    BUILD_MAP_RESOLUTION,
    BUILD_PLACEMENT_CELLS,
    COMMAND_VECTOR_DIM,
    MACRO_VECTOR_DIM,
    MAX_BUILD_WORKERS,
    WORKER_FEAT_DIM,
)
from train.envs.zerg_build_env import ZergBuildEnv
from src.zerg_build_bot import ZergBuildBot


def test_build_env_spaces_match_constants():
    mock_cm = MagicMock()
    mock_cm.get_cumulative_build_map_layers.return_value = [
        "terrain_pathable",
        "terrain_creep",
        "terrain_buildable",
    ]
    mock_cm.get_cumulative_build_commands.return_value = ["spawningpool", "zergling"]
    mock_cm.get_cumulative_active_layers.return_value = []
    mock_cm.get_cumulative_active_tactical_layers.return_value = []
    mock_cm.get_cumulative_action_heads.return_value = ["move"]
    mock_cm.get_cumulative_categorical_heads.return_value = []
    mock_cm.get_agent_architecture.return_value = "build"
    mock_cm.get_level_config.return_value = {
        "training": {"training_steps": 1000},
        "env": {
            "maps_dir": "train/maps/128x128",
            "max_steps": 100,
            "rewards": {},
        },
    }

    training_conf = {
        "config_dir": "config_build_zerg",
        "level_name": "test_build",
        "race": "zerg",
    }

    with patch("train.envs.level_0_env.ConfigManager", return_value=mock_cm), patch(
        "train.envs.level_0_env.SC2Process"
    ), patch("train.envs.level_0_env.Client"), patch(
        "train.envs.build_env.ConfigManager", return_value=mock_cm
    ), patch(
        "train.envs.level_0_env.Level0Env._discover_maps",
        return_value=["test.SC2Map"],
    ):
        env = ZergBuildEnv(
            training_conf=training_conf,
            level_name="test_build",
            max_steps=100,
        )

    assert env.observation_space.spaces["local_map"].shape == (
        3,
        BUILD_MAP_RESOLUTION,
        BUILD_MAP_RESOLUTION,
    )
    assert env.observation_space.spaces["worker_features"].shape == (
        MAX_BUILD_WORKERS,
        WORKER_FEAT_DIM,
    )
    assert env.observation_space.spaces["macro_vector"].shape == (MACRO_VECTOR_DIM,)
    assert env.observation_space.spaces["command"].shape == (COMMAND_VECTOR_DIM,)
    assert list(env.action_space.nvec) == [MAX_BUILD_WORKERS, BUILD_PLACEMENT_CELLS]
    assert env.agent_architecture == "build"
    assert env._bot_cls is ZergBuildBot
