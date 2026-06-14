"""Regression: CTF apply_action must not decode spatial clicks without game_info."""
import asyncio
from unittest.mock import MagicMock

import numpy as np
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def test_apply_action_refreshes_last_game_info_before_spatial_decode():
    bot = CaptureTheFlagBot.__new__(CaptureTheFlagBot)
    bot.player_id = 1
    bot.training_conf = {
        "agent_architecture": "universal",
        "action_heads": ["move_global", "move_local", "attack_local"],
    }
    bot.actions = []
    bot.do = MagicMock()
    bot._unit_last_order = {}
    bot.last_action_was_move_by_unit = {}

    unit = MagicMock()
    unit.owner_id = 1
    unit.tag = 42
    unit.position = Point2((50.0, 50.0))

    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit

    mock_game_info = MagicMock()
    mock_game_info.playable_area = MagicMock(x=0, y=0, width=128, height=128)
    mock_game_info.map_size.width = 128
    mock_game_info.map_size.height = 128
    bot.game_info = mock_game_info

    agent = MagicMock()
    agent.last_game_info = None
    agent.convert_spatial_action_to_world.return_value = (60.0, 70.0)
    bot.agent_system = MagicMock(unit_agents={"active_agent": agent})

    asyncio.run(bot.apply_action(42, np.array([0, 100], dtype=np.int64)))

    assert agent.last_game_info is mock_game_info
    agent.convert_spatial_action_to_world.assert_called_once()
    bot.do.assert_called_once()
