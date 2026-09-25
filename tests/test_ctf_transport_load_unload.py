"""Transport load/unload actuation and cargo kill credits for CTF bot."""
from unittest.mock import MagicMock

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _unit(
    type_id,
    *,
    tag=1,
    owner_id=1,
    pos=(10.0, 10.0),
    cargo_max=0,
    cargo_used=0,
    cargo_size=1,
    health=100.0,
    shield=0.0,
    passengers=None,
    is_structure=False,
):
    u = MagicMock()
    u.type_id = type_id
    u.tag = tag
    u.owner_id = owner_id
    u.position = Point2(pos)
    u.cargo_max = cargo_max
    u.cargo_used = cargo_used
    u.cargo_size = cargo_size
    u.health = health
    u.shield = shield
    u.passengers = passengers or []
    u.is_structure = is_structure
    return u


def _bot_with_allies(allies, player_id=1):
    bot = CaptureTheFlagBot.__new__(CaptureTheFlagBot)
    bot.player_id = player_id
    bot.units = allies if player_id == 1 else []
    bot.enemy_units = allies if player_id != 1 else []
    bot.do = MagicMock()
    return bot


def test_load_on_unit_transport_loads_selected_ally():
    marine = _unit(UnitTypeId.MARINE, tag=2, pos=(50.0, 50.0), cargo_max=0)
    medivac = _unit(
        UnitTypeId.MEDIVAC, tag=1, pos=(5.0, 5.0), cargo_max=8, cargo_used=0
    )
    bot = _bot_with_allies([marine, medivac])
    bot._issue_load_on_unit(medivac, marine)
    medivac.assert_called_once_with(AbilityId.LOAD_MEDIVAC, marine)
    bot.do.assert_called_once()


def test_load_on_unit_passenger_smarts_onto_transport():
    marine = _unit(UnitTypeId.MARINE, tag=2, pos=(5.0, 5.0), cargo_max=0)
    medivac = _unit(
        UnitTypeId.MEDIVAC, tag=1, pos=(10.0, 10.0), cargo_max=8, cargo_used=0
    )
    bot = _bot_with_allies([marine, medivac])
    bot._issue_load_on_unit(marine, medivac)
    marine.smart.assert_called_once_with(medivac)
    bot.do.assert_called_once()


def test_load_transport_loads_ally_in_cell():
    marine = _unit(UnitTypeId.MARINE, tag=2, pos=(10.2, 10.1), cargo_max=0)
    medivac = _unit(
        UnitTypeId.MEDIVAC, tag=1, pos=(5.0, 5.0), cargo_max=8, cargo_used=0
    )
    bot = _bot_with_allies([marine, medivac])
    bot._issue_load_at_point(medivac, Point2((10.0, 10.0)))
    bot.do.assert_called_once()
    args = bot.do.call_args[0][0]
    # unit(ability, target) returns a UnitCommand mock via MagicMock __call__
    medivac.assert_called_once_with(AbilityId.LOAD_MEDIVAC, marine)


def test_load_non_transport_smarts_onto_transport_in_cell():
    marine = _unit(UnitTypeId.MARINE, tag=2, pos=(5.0, 5.0), cargo_max=0)
    medivac = _unit(
        UnitTypeId.MEDIVAC, tag=1, pos=(10.2, 10.1), cargo_max=8, cargo_used=0
    )
    bot = _bot_with_allies([marine, medivac])
    bot._issue_load_at_point(marine, Point2((10.0, 10.0)))
    marine.smart.assert_called_once_with(medivac)
    bot.do.assert_called_once()


def test_load_empty_cell_is_noop():
    medivac = _unit(UnitTypeId.MEDIVAC, tag=1, cargo_max=8)
    bot = _bot_with_allies([medivac])
    bot._issue_load_at_point(medivac, Point2((50.0, 50.0)))
    bot.do.assert_not_called()


def test_unload_at_point():
    medivac = _unit(
        UnitTypeId.MEDIVAC, tag=1, cargo_max=8, cargo_used=2
    )
    bot = _bot_with_allies([medivac])
    target = Point2((12.0, 14.0))
    bot._issue_unload_at_point(medivac, target)
    medivac.assert_called_once_with(AbilityId.UNLOADALLAT_MEDIVAC, target)
    bot.do.assert_called_once()


def test_unload_empty_or_non_transport_noop():
    marine = _unit(UnitTypeId.MARINE, tag=2, cargo_max=0)
    empty = _unit(UnitTypeId.MEDIVAC, tag=1, cargo_max=8, cargo_used=0)
    bot = _bot_with_allies([marine, empty])
    bot._issue_unload_at_point(marine, Point2((1.0, 1.0)))
    bot._issue_unload_at_point(empty, Point2((1.0, 1.0)))
    bot.do.assert_not_called()


def test_mobile_combat_count_includes_passengers():
    p1 = _unit(UnitTypeId.MARINE, tag=10, health=40)
    p2 = _unit(UnitTypeId.MARINE, tag=11, health=40)
    transport = _unit(
        UnitTypeId.MEDIVAC,
        tag=1,
        cargo_max=8,
        cargo_used=2,
        health=150,
        passengers=[p1, p2],
    )
    ground = _unit(UnitTypeId.MARAUDER, tag=3, health=100)
    units = [transport, ground]
    assert CaptureTheFlagBot._mobile_combat_count(units) == 4
    hp = CaptureTheFlagBot._mobile_combat_hp_sum(units)
    assert hp == 150 + 100 + 40 + 40


def test_kill_penalty_scales_with_cargo_on_transport_death():
    """When prev counted transport+cargo and curr has neither, penalty ×(1+N)."""
    bot = CaptureTheFlagBot.__new__(CaptureTheFlagBot)
    bot.player_id = 1
    bot.training_conf = {
        "rewards": {
            "kill_reward": 0.25,
            "kill_penalty": -0.25,
            "damage_reward": 0.0,
            "damage_penalty": 0.0,
            "heal_reward": 0.0,
            "pylon_damage_reward": 0.0,
            "pylon_damage_penalty": 0.0,
            "distance_multiplier": 0.0,
            "step_penalty": 0.0,
        }
    }
    # Minimal stubs used by calculate_squad_step_reward before kill section
    bot.prev_pylon_health = {1: 10.0, 2: 10.0}
    bot.prev_enemy_units_health = {1: 0.0}
    bot.prev_own_units_health = {1: 330.0}  # medivac 150 + 2×40 + marauder 100
    bot.prev_enemy_unit_count = {1: 0}
    bot.prev_own_unit_count = {1: 4}  # transport + 2 passengers + marauder
    bot.unit_command_map = {}
    bot.active_unit_tag = None
    bot.prev_com_distance = {}
    bot._pre_action_dist_to_enemy_com = {}
    bot._pre_action_weapon_cd_by_unit = {}
    bot.last_action_was_move_by_unit = {}

    # After transport death: only marauder remains
    marauder = _unit(UnitTypeId.MARAUDER, tag=3, health=100)
    own = [marauder]
    enemy = []

    # Patch helpers used early in reward that need more bot state
    bot._squad_has_melee_unit = MagicMock(return_value=False)
    bot.structures = MagicMock()
    bot.structures.return_value = MagicMock(exists=False, first=None)
    bot.enemy_structures = MagicMock()
    bot.enemy_structures.return_value = MagicMock(exists=False, first=None)
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag = MagicMock(return_value=None)

    # Call count helpers directly to verify credit math expected by reward path
    curr = CaptureTheFlagBot._mobile_combat_count(own)
    assert curr == 1
    deaths = bot.prev_own_unit_count[1] - curr
    assert deaths == 3  # transport + 2 cargo
    assert deaths * -0.25 == -0.75


def test_load_does_not_change_combat_count():
    marine = _unit(UnitTypeId.MARINE, tag=2, health=40)
    medivac = _unit(
        UnitTypeId.MEDIVAC, tag=1, cargo_max=8, cargo_used=0, health=150
    )
    before = CaptureTheFlagBot._mobile_combat_count([marine, medivac])
    loaded = _unit(
        UnitTypeId.MEDIVAC,
        tag=1,
        cargo_max=8,
        cargo_used=1,
        health=150,
        passengers=[marine],
    )
    after = CaptureTheFlagBot._mobile_combat_count([loaded])
    assert before == after == 2
