"""CTF attack head maps Disruptor → Purification Nova at the policy target."""
import asyncio
from unittest.mock import MagicMock

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _bot_with_unit(unit):
    bot = MagicMock(spec=CaptureTheFlagBot)
    bot.player_id = 1
    bot.training_conf = {
        "agent_architecture": "universal",
        "action_heads": ["move_global", "move_local", "attack_local"],
    }
    bot.actions = []
    bot.do = MagicMock()
    bot._unit_last_order = {}
    bot.last_action_was_move_by_unit = {}
    bot.game_info = None
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    # MagicMock defaults are truthy; empty cast locks must not skip attack.
    bot._refresh_heal_lock = lambda u: False
    bot._refresh_snipe_lock = lambda u: False
    bot._refresh_aoe_lock = lambda u: False
    agent = MagicMock()
    agent.last_game_info = None
    agent.convert_spatial_action_to_world.return_value = (60.0, 70.0)
    bot.agent_system = MagicMock(unit_agents={"active_agent": agent})
    return bot


def test_ctf_apply_action_attack_disruptor_casts_purification_nova():
    unit = MagicMock()
    unit.owner_id = 1
    unit.tag = 42
    unit.type_id = UnitTypeId.DISRUPTOR
    unit.position = Point2((50.0, 50.0))
    cast_cmd = object()
    unit.return_value = cast_cmd

    bot = _bot_with_unit(unit)
    # attack_local is index 2
    asyncio.run(CaptureTheFlagBot.apply_action(bot, 42, np.array([2, 100], dtype=np.int64)))

    unit.assert_called_once_with(AbilityId.EFFECT_PURIFICATIONNOVA, Point2((60.0, 70.0)))
    bot.do.assert_called_once_with(cast_cmd)
    unit.attack.assert_not_called()


def test_ctf_apply_squad_action_attack_disruptor_casts_purification_nova():
    unit = MagicMock()
    unit.owner_id = 1
    unit.tag = 7
    unit.type_id = UnitTypeId.DISRUPTOR
    unit.position = Point2((40.0, 40.0))
    cast_cmd = object()
    unit.return_value = cast_cmd

    bot = _bot_with_unit(unit)

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[7],
            anchor_tag=7,
            command_idx=2,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    unit.assert_called_once_with(AbilityId.EFFECT_PURIFICATIONNOVA, Point2((60.0, 70.0)))
    bot.do.assert_called_once_with(cast_cmd)
    unit.attack.assert_not_called()


def test_ctf_apply_squad_action_attack_nova_ball_moves():
    unit = MagicMock()
    unit.owner_id = 1
    unit.tag = 99
    unit.type_id = UnitTypeId.DISRUPTORPHASED
    unit.position = Point2((40.0, 40.0))
    move_cmd = object()
    unit.move.return_value = move_cmd

    bot = _bot_with_unit(unit)

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[99],
            anchor_tag=99,
            command_idx=2,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    unit.move.assert_called_once_with(Point2((60.0, 70.0)))
    bot.do.assert_called_once_with(move_cmd)
    unit.attack.assert_not_called()
    unit.assert_not_called()
