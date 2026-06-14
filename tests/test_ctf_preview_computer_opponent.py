from unittest.mock import MagicMock

import numpy as np

from src.agent.ctf_bot import CaptureTheFlagBot
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


def test_ctf_preview_computer_opponent_flag():
    env = CaptureTheFlagSingleEnv(
        level_name="ctf_level_01",
        training_conf={"preview_computer_opponent": True},
    )
    assert env._preview_computer_opponent() is True

    env2 = CaptureTheFlagSingleEnv(level_name="ctf_level_01", training_conf={})
    assert env2._preview_computer_opponent() is False


def test_ctf_bot_zero_observation_universal_includes_tactical():
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={
            "agent_architecture": "universal",
            "active_layers": ["allied_unit_biological", "enemy_unit_biological"],
            "active_tactical_layers": ["allied_unit_biological", "terrain_pathable"],
        },
    )
    obs = bot._get_zero_observation()
    assert isinstance(obs, dict)
    assert obs["tactical"].shape == (2, 32, 32)


def test_ctf_env_ensure_preview_focal_unit_selects_agent_unit():
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.bot = MagicMock()
    env.bot.active_unit_tag = None
    env.bot.all_units.find_by_tag.return_value = None
    env.unit_pools_config = [{"agent": True, "region": "A"}]
    env.unit_pool_index = {101: 0, 202: 0}
    env.player_units = {1: [101, 999], 2: [202]}
    env.squad_unit_tags = []
    env.current_squad_index = 0
    env.active_player_id = 1

    env._ensure_preview_focal_unit()
    assert env.bot.active_unit_tag == 101
