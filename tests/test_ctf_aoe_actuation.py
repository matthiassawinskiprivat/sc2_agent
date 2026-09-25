"""CTF aoe_local actuation: splash spells by caster; non-casters no-op."""
import asyncio
from unittest.mock import MagicMock

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_unit(tag, type_id, x, y, owner_id=1, is_flying=False, is_structure=False):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.owner_id = owner_id
    u.position = Point2((x, y))
    u.is_flying = is_flying
    u.is_structure = is_structure
    u.is_alive = True
    cast_cmd = object()
    u.return_value = cast_cmd
    return u


def _bot_with_caster(caster, extra_units=None):
    extras = list(extra_units or [])
    bot = MagicMock(spec=CaptureTheFlagBot)
    bot.player_id = 1
    bot.training_conf = {
        "agent_architecture": "universal",
        "action_heads": ["move_global", "move_local", "attack_local", "aoe_local"],
    }
    bot.actions = []
    bot.do = MagicMock()
    bot._unit_last_order = {}
    bot.last_action_was_move_by_unit = {}
    bot.game_info = None

    by_tag = {caster.tag: caster}
    for u in extras:
        by_tag[u.tag] = u

    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = lambda tag: by_tag.get(tag)
    bot.units = [caster] + [u for u in extras if u.owner_id == 1]
    bot.structures = []
    bot.enemy_units = [u for u in extras if u.owner_id != 1]
    bot.enemy_structures = []
    # MagicMock(spec=...) leaves unbound methods as truthy mocks; heal-lock
    # would otherwise short-circuit every AoE before issue_aoe_at_point runs.
    bot._refresh_heal_lock = lambda unit: False
    bot._mark_command_attempt = MagicMock()

    agent = MagicMock()
    agent.last_game_info = None
    agent.convert_spatial_action_to_world.return_value = (60.0, 70.0)
    bot.agent_system = MagicMock(unit_agents={"active_agent": agent})
    return bot


def _run_aoe(bot, caster_tag=1):
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[caster_tag],
            anchor_tag=caster_tag,
            command_idx=3,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )


def test_squad_aoe_infestor_fungal_at_point():
    caster = _make_unit(1, UnitTypeId.INFESTOR, 50.0, 50.0)
    bot = _bot_with_caster(caster)
    _run_aoe(bot)
    caster.assert_called_once_with(AbilityId.FUNGALGROWTH_FUNGALGROWTH, Point2((60.0, 70.0)))
    bot.do.assert_called_once()
    caster.move.assert_not_called()
    bot._mark_command_attempt.assert_called_once_with(caster, "aoe")


def test_squad_aoe_burrowed_infestor_fungal():
    caster = _make_unit(1, UnitTypeId.INFESTORBURROWED, 50.0, 50.0)
    bot = _bot_with_caster(caster)
    _run_aoe(bot)
    caster.assert_called_once_with(AbilityId.FUNGALGROWTH_FUNGALGROWTH, Point2((60.0, 70.0)))


def test_squad_aoe_high_templar_storm():
    caster = _make_unit(1, UnitTypeId.HIGHTEMPLAR, 50.0, 50.0)
    bot = _bot_with_caster(caster)
    _run_aoe(bot)
    caster.assert_called_once_with(AbilityId.PSISTORM_PSISTORM, Point2((60.0, 70.0)))


def test_squad_aoe_ghost_emp():
    caster = _make_unit(1, UnitTypeId.GHOST, 50.0, 50.0)
    bot = _bot_with_caster(caster)
    _run_aoe(bot)
    caster.assert_called_once_with(AbilityId.EMP_EMP, Point2((60.0, 70.0)))


def test_squad_aoe_ravager_bile():
    caster = _make_unit(1, UnitTypeId.RAVAGER, 50.0, 50.0)
    bot = _bot_with_caster(caster)
    _run_aoe(bot)
    caster.assert_called_once_with(AbilityId.EFFECT_CORROSIVEBILE, Point2((60.0, 70.0)))


def test_squad_aoe_viper_bombs_air_in_cell():
    caster = _make_unit(1, UnitTypeId.VIPER, 50.0, 50.0)
    muta = _make_unit(2, UnitTypeId.MUTALISK, 60.2, 70.1, owner_id=2, is_flying=True)
    bot = _bot_with_caster(caster, extra_units=[muta])
    _run_aoe(bot)
    caster.assert_called_once_with(AbilityId.PARASITICBOMB_PARASITICBOMB, muta)
    bot.do.assert_called_once()


def test_squad_aoe_viper_empty_cell_noop():
    caster = _make_unit(1, UnitTypeId.VIPER, 50.0, 50.0)
    marine = _make_unit(2, UnitTypeId.MARINE, 60.2, 70.1, owner_id=2, is_flying=False)
    bot = _bot_with_caster(caster, extra_units=[marine])
    _run_aoe(bot)
    caster.assert_not_called()
    bot.do.assert_not_called()


def test_squad_aoe_non_caster_noop():
    marine = _make_unit(1, UnitTypeId.MARINE, 50.0, 50.0)
    bot = _bot_with_caster(marine)
    _run_aoe(bot)
    marine.assert_not_called()
    marine.move.assert_not_called()
    bot.do.assert_not_called()
    bot._mark_command_attempt.assert_called_once_with(marine, "aoe")


def test_squad_aoe_mixed_selection_only_casters_fire():
    ravager = _make_unit(1, UnitTypeId.RAVAGER, 50.0, 50.0)
    marine = _make_unit(2, UnitTypeId.MARINE, 51.0, 50.0)
    bot = _bot_with_caster(ravager, extra_units=[marine])
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1, 2],
            anchor_tag=1,
            command_idx=3,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )
    ravager.assert_called_once_with(AbilityId.EFFECT_CORROSIVEBILE, Point2((60.0, 70.0)))
    marine.assert_not_called()
    marine.move.assert_not_called()


def test_apply_action_aoe_ravager():
    caster = _make_unit(1, UnitTypeId.RAVAGER, 50.0, 50.0)
    bot = _bot_with_caster(caster)
    asyncio.run(CaptureTheFlagBot.apply_action(bot, 1, np.array([3, 100], dtype=np.int64)))
    caster.assert_called_once_with(AbilityId.EFFECT_CORROSIVEBILE, Point2((60.0, 70.0)))
    bot.do.assert_called_once()
    bot._mark_command_attempt.assert_called_once_with(caster, "aoe")


def test_squad_unload_marks_unload_command():
    medivac = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    bot = _bot_with_caster(medivac)
    bot.training_conf["action_heads"] = ["move_local", "unload_local"]
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=1,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )
    bot._mark_command_attempt.assert_called_once_with(medivac, "unload")
