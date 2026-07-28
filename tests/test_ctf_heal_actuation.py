"""CTF heal_local actuation: Medivac/SCV/Queen abilities; battery/non-healers no-op."""
import asyncio
from unittest.mock import MagicMock

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_unit(tag, type_id, x, y, owner_id=1):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.owner_id = owner_id
    u.position = Point2((x, y))
    u.is_structure = type_id == UnitTypeId.SHIELDBATTERY
    cast_cmd = object()
    u.return_value = cast_cmd
    return u


def _bot_with_healers(healer, ally):
    bot = MagicMock(spec=CaptureTheFlagBot)
    bot.player_id = 1
    bot.training_conf = {
        "agent_architecture": "universal",
        "action_heads": ["move_global", "move_local", "attack_local", "heal_local"],
    }
    bot.actions = []
    bot.do = MagicMock()
    bot._unit_last_order = {}
    bot.last_action_was_move_by_unit = {}
    bot.game_info = None

    def find_by_tag(tag):
        if tag == healer.tag:
            return healer
        if tag == ally.tag:
            return ally
        return None

    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = find_by_tag
    bot.units = [healer, ally]
    bot.enemy_units = []

    # Bind real helpers onto the mock so apply_squad_action works.
    bot._issue_heal_at_point = lambda unit, target: CaptureTheFlagBot._issue_heal_at_point(
        bot, unit, target
    )
    bot._resolve_heal_target_in_cell = (
        lambda unit, target: CaptureTheFlagBot._resolve_heal_target_in_cell(
            bot, unit, target
        )
    )
    bot._allied_units_for_owner = (
        lambda owner_id: CaptureTheFlagBot._allied_units_for_owner(bot, owner_id)
    )
    bot._unit_in_spatial_cell = CaptureTheFlagBot._unit_in_spatial_cell

    agent = MagicMock()
    agent.last_game_info = None
    agent.convert_spatial_action_to_world.return_value = (
        float(ally.position.x),
        float(ally.position.y),
    )
    bot.agent_system = MagicMock(unit_agents={"active_agent": agent})
    return bot


def test_squad_heal_medivac_issues_heal():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 60.0, 70.0)
    bot = _bot_with_healers(healer, ally)

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=3,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    healer.assert_called_once_with(AbilityId.MEDIVACHEAL_HEAL, ally)
    bot.do.assert_called_once()
    healer.move.assert_not_called()


def test_squad_heal_scv_issues_repair():
    healer = _make_unit(1, UnitTypeId.SCV, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 60.0, 70.0)
    bot = _bot_with_healers(healer, ally)

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=3,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    healer.assert_called_once_with(AbilityId.EFFECT_REPAIR_SCV, ally)
    bot.do.assert_called_once()


def test_squad_heal_queen_issues_transfusion():
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.ROACH, 60.0, 70.0)
    bot = _bot_with_healers(healer, ally)

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=3,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    healer.assert_called_once_with(AbilityId.TRANSFUSION_TRANSFUSION, ally)
    bot.do.assert_called_once()


def test_squad_heal_non_healer_noop():
    marine = _make_unit(1, UnitTypeId.MARINE, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 60.0, 70.0)
    bot = _bot_with_healers(marine, ally)

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=3,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    marine.assert_not_called()
    marine.move.assert_not_called()
    bot.do.assert_not_called()


def test_squad_heal_battery_noop():
    battery = _make_unit(1, UnitTypeId.SHIELDBATTERY, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.ZEALOT, 60.0, 70.0)
    bot = _bot_with_healers(battery, ally)

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=3,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    battery.assert_not_called()
    battery.move.assert_not_called()
    bot.do.assert_not_called()


def test_squad_heal_no_ally_in_cell_noop():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 80.0, 80.0)  # far from decode target
    bot = _bot_with_healers(healer, ally)
    bot.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world.return_value = (
        60.0,
        70.0,
    )

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=3,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    healer.assert_not_called()
    bot.do.assert_not_called()


def test_apply_action_heal_medivac():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 60.0, 70.0)
    bot = _bot_with_healers(healer, ally)

    asyncio.run(CaptureTheFlagBot.apply_action(bot, 1, np.array([3, 100], dtype=np.int64)))

    healer.assert_called_once_with(AbilityId.MEDIVACHEAL_HEAL, ally)
    bot.do.assert_called_once()
