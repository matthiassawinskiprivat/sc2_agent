"""CTF attack head maps Swarm Host → Spawn Locusts at the policy target."""
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
        "categorical_heads": [],
    }
    bot.actions = []
    bot.do = MagicMock()
    bot._unit_last_order = {}
    bot.last_action_was_move_by_unit = {}
    bot.last_action_was_attack_by_unit = {}
    bot._command_attempt_tags = {
        "move": set(),
        "attack": set(),
        "cloak": set(),
        "uncloak": set(),
        "aoe": set(),
        "unload": set(),
        "load": set(),
    }
    bot.game_info = None
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    # Bind real helpers: MagicMock defaults are truthy and would skip attack
    # via ``elif self._refresh_heal_lock(unit)``.
    bot._refresh_heal_lock = lambda u: CaptureTheFlagBot._refresh_heal_lock(bot, u)
    bot._mark_command_attempt = (
        lambda u, kind: CaptureTheFlagBot._mark_command_attempt(bot, u, kind)
    )
    bot._set_last_order_flags = (
        lambda unit_tag, *, moved=False, attacked=False: (
            CaptureTheFlagBot._set_last_order_flags(
                bot, unit_tag, moved=moved, attacked=attacked
            )
        )
    )
    agent = MagicMock()
    agent.last_game_info = None
    agent.convert_spatial_action_to_world.return_value = (60.0, 70.0)
    bot.agent_system = MagicMock(unit_agents={"active_agent": agent})
    return bot


def test_ctf_apply_action_attack_swarmhost_spawns_locusts():
    unit = MagicMock()
    unit.owner_id = 1
    unit.tag = 42
    unit.type_id = UnitTypeId.SWARMHOSTMP
    unit.position = Point2((50.0, 50.0))
    cast_cmd = object()
    unit.return_value = cast_cmd

    bot = _bot_with_unit(unit)
    # attack_local is index 2
    asyncio.run(CaptureTheFlagBot.apply_action(bot, 42, np.array([2, 100], dtype=np.int64)))

    unit.assert_called_once_with(AbilityId.EFFECT_SPAWNLOCUSTS, Point2((60.0, 70.0)))
    bot.do.assert_called_once_with(cast_cmd)
    unit.attack.assert_not_called()


def test_ctf_apply_squad_action_attack_swarmhost_spawns_locusts():
    unit = MagicMock()
    unit.owner_id = 1
    unit.tag = 7
    unit.type_id = UnitTypeId.SWARMHOSTMP
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

    unit.assert_called_once_with(AbilityId.EFFECT_SPAWNLOCUSTS, Point2((60.0, 70.0)))
    bot.do.assert_called_once_with(cast_cmd)
    unit.attack.assert_not_called()
