"""Categorical cloak_burrow / uncloak_unburrow actuation."""
import asyncio
from unittest.mock import MagicMock

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_unit(tag, type_id, x=50.0, y=50.0):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.owner_id = 1
    u.position = Point2((x, y))
    u.is_structure = False
    cast_cmd = object()
    u.return_value = cast_cmd
    return u


def _bot(unit):
    bot = MagicMock(spec=CaptureTheFlagBot)
    bot.player_id = 1
    bot.training_conf = {
        "agent_architecture": "squad",
        "action_heads": ["move_global", "move_local", "attack_local", "heal_local"],
        "categorical_heads": ["cloak_burrow", "uncloak_unburrow"],
    }
    bot.do = MagicMock()
    bot._unit_last_order = {}
    bot.last_action_was_move_by_unit = {}
    bot.game_info = None
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    agent = MagicMock()
    agent.last_game_info = None
    agent.convert_spatial_action_to_world.return_value = (60.0, 70.0)
    bot.agent_system = MagicMock(unit_agents={"active_agent": agent})
    bot._issue_cloak_burrow = lambda u, conceal: CaptureTheFlagBot._issue_cloak_burrow(
        bot, u, conceal=conceal
    )
    bot._BURROW_DOWN_BY_TYPE = CaptureTheFlagBot._BURROW_DOWN_BY_TYPE
    bot._BURROW_UP_BY_TYPE = CaptureTheFlagBot._BURROW_UP_BY_TYPE
    return bot


def test_squad_cloak_burrow_ghost_cloaks():
    unit = _make_unit(1, UnitTypeId.GHOST)
    bot = _bot(unit)
    # cloak_burrow is first categorical → index 4
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=4,
            spatial_idx=0,
            perspective_player_id=1,
        )
    )
    unit.assert_called_once_with(AbilityId.BEHAVIOR_CLOAKON_GHOST)
    bot.do.assert_called_once()


def test_squad_uncloak_ghost():
    unit = _make_unit(1, UnitTypeId.GHOST)
    bot = _bot(unit)
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=5,
            spatial_idx=0,
            perspective_player_id=1,
        )
    )
    unit.assert_called_once_with(AbilityId.BEHAVIOR_CLOAKOFF_GHOST)


def test_squad_cloak_burrow_zergling_burrows():
    unit = _make_unit(1, UnitTypeId.ZERGLING)
    bot = _bot(unit)
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=4,
            spatial_idx=0,
            perspective_player_id=1,
        )
    )
    unit.assert_called_once_with(AbilityId.BURROWDOWN_ZERGLING)


def test_squad_unburrow_zergling():
    unit = _make_unit(1, UnitTypeId.ZERGLINGBURROWED)
    bot = _bot(unit)
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=5,
            spatial_idx=0,
            perspective_player_id=1,
        )
    )
    unit.assert_called_once_with(AbilityId.BURROWUP_ZERGLING)


def test_squad_cloak_marine_noop():
    unit = _make_unit(1, UnitTypeId.MARINE)
    bot = _bot(unit)
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=4,
            spatial_idx=0,
            perspective_player_id=1,
        )
    )
    unit.assert_not_called()
    bot.do.assert_not_called()


def test_squad_cloak_dark_templar_noop():
    unit = _make_unit(1, UnitTypeId.DARKTEMPLAR)
    bot = _bot(unit)
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=4,
            spatial_idx=0,
            perspective_player_id=1,
        )
    )
    unit.assert_not_called()
    bot.do.assert_not_called()
