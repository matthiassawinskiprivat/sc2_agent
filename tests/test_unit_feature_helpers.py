"""Tests for the weapon_range_upgrade_deltas helper."""
import pytest
from unittest.mock import MagicMock

try:
    from sc2.ids.unit_typeid import UnitTypeId
    from sc2.ids.upgrade_id import UpgradeId
    _SC2 = True
except ImportError:
    _SC2 = False

from src.utils.unit_feature_helpers import (
    counts_for_combat_hp_reward,
    is_carrier_interceptor,
    is_force_melee_unit,
    is_purification_nova_ball,
    is_temporary_attrition_unit,
    disable_cyclone_lock_on_autocast,
    issue_aoe_at_point,
    issue_attack_at_point,
    resolve_corruptor_target_at_point,
    resolve_enemy_at_point,
    heal_click_cell_half,
    heal_click_cells_from_conf,
    heal_click_hits_unit,
    unit_center_in_spatial_cell,
    unit_click_radius,
    uses_aoe_spell,
    unit_can_aoe,
    unit_can_attack,
    uses_caustic_spray_on_structures,
    uses_lock_on_for_attack,
    uses_purification_nova_for_attack,
    uses_spawn_locusts_for_attack,
    weapon_range_upgrade_deltas,
)

pytestmark = pytest.mark.skipif(not _SC2, reason="SC2 not installed")


def _mock_unit(uid):
    unit = MagicMock()
    unit.type_id = uid
    return unit


def test_no_upgrades_returns_zero():
    unit = _mock_unit(UnitTypeId.MARINE)
    g, a = weapon_range_upgrade_deltas(unit, frozenset())
    assert g == 0.0
    assert a == 0.0


def test_hydralisk_grooved_spines():
    unit = _mock_unit(UnitTypeId.HYDRALISK)
    g, a = weapon_range_upgrade_deltas(unit, frozenset({UpgradeId.EVOLVEGROOVEDSPINES}))
    assert g == 1.0
    assert a == 1.0


def test_colossus_thermal_lance():
    unit = _mock_unit(UnitTypeId.COLOSSUS)
    g, a = weapon_range_upgrade_deltas(unit, frozenset({UpgradeId.EXTENDEDTHERMALLANCE}))
    assert g == 3.0
    assert a == 0.0


def test_phoenix_range_upgrade():
    unit = _mock_unit(UnitTypeId.PHOENIX)
    g, a = weapon_range_upgrade_deltas(unit, frozenset({UpgradeId.PHOENIXRANGEUPGRADE}))
    assert g == 0.0
    assert a == 2.0


def test_unrelated_upgrade_returns_zero():
    unit = _mock_unit(UnitTypeId.MARINE)
    g, a = weapon_range_upgrade_deltas(unit, frozenset({UpgradeId.EVOLVEGROOVEDSPINES}))
    assert g == 0.0
    assert a == 0.0


def test_empty_upgrades_frozenset():
    unit = _mock_unit(UnitTypeId.HYDRALISK)
    g, a = weapon_range_upgrade_deltas(unit, frozenset())
    assert g == 0.0 and a == 0.0


def test_force_melee_includes_archon_and_helliontank():
    assert is_force_melee_unit(_mock_unit(UnitTypeId.ARCHON)) is True
    assert is_force_melee_unit(_mock_unit(UnitTypeId.HELLIONTANK)) is True
    assert is_force_melee_unit(_mock_unit(UnitTypeId.MARINE)) is False
    assert is_force_melee_unit(_mock_unit(UnitTypeId.ZEALOT)) is False


def test_temporary_attrition_includes_broodling_and_locusts():
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.BROODLING)) is True
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.LOCUSTMP)) is True
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.LOCUSTMPFLYING)) is True
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.DISRUPTORPHASED)) is True
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.INTERCEPTOR)) is True
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.ADEPTPHASESHIFT)) is True
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.INFESTORTERRAN)) is True
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.CHANGELING)) is True
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.BROODLORD)) is False
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.SWARMHOSTMP)) is False
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.DISRUPTOR)) is False
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.MARINE)) is False


def test_interceptor_excluded_from_combat_hp_ledger_but_carrier_counts():
    assert is_carrier_interceptor(_mock_unit(UnitTypeId.INTERCEPTOR)) is True
    assert is_carrier_interceptor(_mock_unit(UnitTypeId.CARRIER)) is False
    assert counts_for_combat_hp_reward(_mock_unit(UnitTypeId.INTERCEPTOR)) is False
    assert counts_for_combat_hp_reward(_mock_unit(UnitTypeId.CARRIER)) is True
    assert counts_for_combat_hp_reward(_mock_unit(UnitTypeId.CORRUPTOR)) is True


def test_uses_spawn_locusts_for_swarm_hosts_only():
    assert uses_spawn_locusts_for_attack(_mock_unit(UnitTypeId.SWARMHOSTMP)) is True
    assert uses_spawn_locusts_for_attack(_mock_unit(UnitTypeId.SWARMHOSTBURROWEDMP)) is True
    assert uses_spawn_locusts_for_attack(_mock_unit(UnitTypeId.MARINE)) is False
    assert uses_spawn_locusts_for_attack(_mock_unit(UnitTypeId.LOCUSTMP)) is False
    assert uses_spawn_locusts_for_attack(_mock_unit(UnitTypeId.DISRUPTOR)) is False


def test_uses_purification_nova_for_disruptor_only():
    assert uses_purification_nova_for_attack(_mock_unit(UnitTypeId.DISRUPTOR)) is True
    assert uses_purification_nova_for_attack(_mock_unit(UnitTypeId.DISRUPTORPHASED)) is False
    assert uses_purification_nova_for_attack(_mock_unit(UnitTypeId.MARINE)) is False
    assert uses_purification_nova_for_attack(_mock_unit(UnitTypeId.SWARMHOSTMP)) is False


def test_is_purification_nova_ball_only_phased():
    assert is_purification_nova_ball(_mock_unit(UnitTypeId.DISRUPTORPHASED)) is True
    assert is_purification_nova_ball(_mock_unit(UnitTypeId.DISRUPTOR)) is False
    assert is_purification_nova_ball(_mock_unit(UnitTypeId.MARINE)) is False


def test_uses_lock_on_for_cyclone_only():
    assert uses_lock_on_for_attack(_mock_unit(UnitTypeId.CYCLONE)) is True
    assert uses_lock_on_for_attack(_mock_unit(UnitTypeId.MARINE)) is False
    assert uses_lock_on_for_attack(_mock_unit(UnitTypeId.SIEGETANK)) is False


def test_issue_attack_at_point_swarmhost_casts_spawn_locusts():
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    bot = MagicMock()
    unit = MagicMock()
    unit.type_id = UnitTypeId.SWARMHOSTMP
    cast_cmd = object()
    unit.return_value = cast_cmd
    target = Point2((12.0, 34.0))

    issue_attack_at_point(bot, unit, target)

    unit.assert_called_once_with(AbilityId.EFFECT_SPAWNLOCUSTS, target)
    bot.do.assert_called_once_with(cast_cmd)
    unit.attack.assert_not_called()


def test_issue_attack_at_point_disruptor_casts_purification_nova():
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    bot = MagicMock()
    unit = MagicMock()
    unit.type_id = UnitTypeId.DISRUPTOR
    cast_cmd = object()
    unit.return_value = cast_cmd
    target = Point2((12.0, 34.0))

    issue_attack_at_point(bot, unit, target)

    unit.assert_called_once_with(AbilityId.EFFECT_PURIFICATIONNOVA, target)
    bot.do.assert_called_once_with(cast_cmd)
    unit.attack.assert_not_called()


def test_issue_attack_at_point_nova_ball_moves():
    from sc2.position import Point2

    bot = MagicMock()
    unit = MagicMock()
    unit.type_id = UnitTypeId.DISRUPTORPHASED
    move_cmd = object()
    unit.move.return_value = move_cmd
    target = Point2((12.0, 34.0))

    issue_attack_at_point(bot, unit, target)

    unit.move.assert_called_once_with(target)
    bot.do.assert_called_once_with(move_cmd)
    unit.attack.assert_not_called()
    unit.assert_not_called()


def test_issue_attack_at_point_marine_uses_attack():
    from sc2.position import Point2

    bot = MagicMock()
    # Explicit empty collections so resolve does not touch mock iterables.
    bot.enemy_units = []
    bot.enemy_structures = []
    unit = MagicMock()
    unit.type_id = UnitTypeId.MARINE
    target = Point2((12.0, 34.0))

    issue_attack_at_point(bot, unit, target)

    unit.attack.assert_called_once_with(target)
    bot.do.assert_called_once()
    unit.assert_not_called()


def test_unit_center_in_spatial_cell_half_gu():
    from sc2.position import Point2

    unit = MagicMock()
    unit.position = Point2((10.4, 10.4))
    assert unit_center_in_spatial_cell(unit, Point2((10.0, 10.0))) is True
    unit.position = Point2((10.6, 10.0))
    assert unit_center_in_spatial_cell(unit, Point2((10.0, 10.0))) is False


def test_heal_click_hits_unit_uses_body_radius():
    from sc2.position import Point2

    thor = MagicMock()
    thor.position = Point2((12.0, 10.0))
    thor.radius = 1.875
    click = Point2((10.5, 10.0))
    assert unit_center_in_spatial_cell(thor, click) is False
    assert heal_click_hits_unit(thor, click) is True
    assert unit_click_radius(thor) == pytest.approx(1.875)

    cyclone = MagicMock()
    cyclone.position = Point2((12.0, 10.0))
    cyclone.radius = 0.75
    assert heal_click_hits_unit(cyclone, click) is False


def test_heal_click_cells_from_conf_defaults_and_clamps():
    assert heal_click_cells_from_conf({}) == 1.0
    assert heal_click_cells_from_conf({"env": {}}) == 1.0
    assert heal_click_cell_half({}) == pytest.approx(0.5)
    assert heal_click_cells_from_conf({"env": {"heal_click_cells": 3}}) == 3.0
    assert heal_click_cell_half({"heal_click_cells": 3}) == pytest.approx(1.5)
    assert heal_click_cells_from_conf({"env": {"heal_click_cells": 5}}) == 5.0
    assert heal_click_cells_from_conf({"env": {"heal_click_cells": 0}}) == 1.0
    assert heal_click_cells_from_conf({"env": {"heal_click_cells": 9}}) == 5.0
    assert heal_click_cells_from_conf({"env": {"heal_click_cells": "nope"}}) == 1.0


def test_resolve_enemy_at_point_picks_closer_in_cell():
    from sc2.position import Point2

    near = MagicMock()
    near.tag = 1
    near.position = Point2((10.1, 10.0))
    near.is_alive = True
    far_in_cell = MagicMock()
    far_in_cell.tag = 2
    far_in_cell.position = Point2((10.4, 10.0))
    far_in_cell.is_alive = True
    outside = MagicMock()
    outside.tag = 3
    outside.position = Point2((12.0, 10.0))
    outside.is_alive = True

    bot = MagicMock()
    bot.enemy_units = [near, far_in_cell, outside]
    bot.enemy_structures = []

    hit = resolve_enemy_at_point(bot, Point2((10.0, 10.0)))
    assert hit is near


def test_resolve_enemy_at_point_includes_structures():
    from sc2.position import Point2

    pylon = MagicMock()
    pylon.tag = 99
    pylon.position = Point2((20.2, 20.1))
    pylon.is_alive = True

    bot = MagicMock()
    bot.enemy_units = []
    bot.enemy_structures = [pylon]

    assert resolve_enemy_at_point(bot, Point2((20.0, 20.0))) is pylon


def test_issue_attack_at_point_cyclone_lock_on_hard_target():
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    enemy = MagicMock()
    enemy.tag = 55
    enemy.position = Point2((12.2, 34.1))
    enemy.is_alive = True

    bot = MagicMock()
    bot.enemy_units = [enemy]
    bot.enemy_structures = []
    unit = MagicMock()
    unit.type_id = UnitTypeId.CYCLONE
    lock_cmd = object()
    unit.return_value = lock_cmd
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is enemy
    unit.assert_called_once_with(AbilityId.LOCKON_LOCKON, enemy)
    bot.do.assert_called_once_with(lock_cmd)
    unit.attack.assert_not_called()


def test_issue_attack_at_point_cyclone_attack_moves_empty_cell():
    from sc2.position import Point2

    enemy = MagicMock()
    enemy.tag = 55
    enemy.position = Point2((40.0, 40.0))
    enemy.is_alive = True

    bot = MagicMock()
    bot.enemy_units = [enemy]
    bot.enemy_structures = []
    unit = MagicMock()
    unit.type_id = UnitTypeId.CYCLONE
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is None
    unit.attack.assert_called_once_with(target)
    unit.assert_not_called()


def test_disable_cyclone_lock_on_autocast_toggles_new_tags_once():
    import asyncio
    from sc2.ids.ability_id import AbilityId

    cyclone = MagicMock()
    cyclone.type_id = UnitTypeId.CYCLONE
    cyclone.tag = 101
    marine = MagicMock()
    marine.type_id = UnitTypeId.MARINE
    marine.tag = 202

    bot = MagicMock()
    bot.units = [cyclone, marine]
    bot.enemy_units = []
    bot.all_units = [cyclone, marine]
    bot.client.toggle_autocast = MagicMock(return_value=None)

    assert asyncio.run(disable_cyclone_lock_on_autocast(bot)) == 1
    bot.client.toggle_autocast.assert_called_once_with([cyclone], AbilityId.LOCKON_LOCKON)
    assert asyncio.run(disable_cyclone_lock_on_autocast(bot)) == 0
    assert bot.client.toggle_autocast.call_count == 1


def test_disable_cyclone_lock_on_autocast_includes_enemy_cyclones():
    import asyncio
    from sc2.ids.ability_id import AbilityId

    enemy_cyclone = MagicMock()
    enemy_cyclone.type_id = UnitTypeId.CYCLONE
    enemy_cyclone.tag = 303

    bot = MagicMock()
    bot.all_units = []
    bot.units = []
    bot.enemy_units = [enemy_cyclone]
    bot.client.toggle_autocast = MagicMock(return_value=None)

    assert asyncio.run(disable_cyclone_lock_on_autocast(bot)) == 1
    bot.client.toggle_autocast.assert_called_once_with(
        [enemy_cyclone], AbilityId.LOCKON_LOCKON
    )


def test_issue_attack_at_point_hard_targets_enemy_in_cell():
    from sc2.position import Point2

    enemy = MagicMock()
    enemy.tag = 55
    enemy.position = Point2((12.2, 34.1))
    enemy.is_alive = True

    bot = MagicMock()
    bot.enemy_units = [enemy]
    bot.enemy_structures = []
    unit = MagicMock()
    unit.type_id = UnitTypeId.BANELING
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is enemy
    unit.attack.assert_called_once_with(enemy)
    bot.do.assert_called_once()


def test_issue_attack_at_point_falls_back_to_point_when_cell_empty():
    from sc2.position import Point2

    enemy = MagicMock()
    enemy.tag = 55
    enemy.position = Point2((40.0, 40.0))
    enemy.is_alive = True

    bot = MagicMock()
    bot.enemy_units = [enemy]
    bot.enemy_structures = []
    unit = MagicMock()
    unit.type_id = UnitTypeId.MARINE
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is None
    unit.attack.assert_called_once_with(target)


def test_issue_attack_at_point_p2_caster_targets_trainee_not_allies():
    """God-mode opponent (owner 2) must hard-target P1, not its own army."""
    from sc2.position import Point2

    ally = MagicMock()
    ally.tag = 201
    ally.owner_id = 2
    ally.position = Point2((12.1, 34.0))
    ally.is_alive = True
    ally.is_structure = False

    trainee = MagicMock()
    trainee.tag = 101
    trainee.owner_id = 1
    trainee.position = Point2((12.2, 34.0))
    trainee.is_alive = True
    trainee.is_structure = False

    bot = MagicMock()
    bot.player_id = 1
    bot.enemy_units = [ally]
    bot.enemy_structures = []
    bot.units = [trainee]
    bot.structures = []

    unit = MagicMock()
    unit.type_id = UnitTypeId.CORRUPTOR
    unit.owner_id = 2
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is trainee
    unit.attack.assert_called_once_with(trainee)


def test_issue_attack_at_point_hard_targets_neuraled_same_owner():
    """Controller and original owner can both hard-target a neuralled body."""
    from sc2.ids.buff_id import BuffId
    from sc2.position import Point2

    stolen = MagicMock()
    stolen.tag = 7
    stolen.owner_id = 1
    stolen.position = Point2((12.05, 34.0))
    stolen.is_alive = True
    stolen.is_structure = False
    stolen.buffs = {BuffId.NEURALPARASITE}

    bot = MagicMock()
    bot.all_units = [stolen]
    bot.units = [stolen]
    bot.enemy_units = []
    bot.structures = []
    bot.enemy_structures = []

    caster = MagicMock()
    caster.tag = 1
    caster.type_id = UnitTypeId.ROACH
    caster.owner_id = 1
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, caster, target) is stolen
    caster.attack.assert_called_once_with(stolen)


def test_issue_attack_at_point_p1_ignores_allies_leaked_into_enemy_units():
    """P1 must not hard-target owner-1 units even if they appear in enemy_units."""
    from sc2.position import Point2

    ally = MagicMock()
    ally.tag = 11
    ally.owner_id = 1
    ally.position = Point2((12.05, 34.0))
    ally.is_alive = True
    ally.is_structure = False

    enemy = MagicMock()
    enemy.tag = 22
    enemy.owner_id = 2
    enemy.position = Point2((12.4, 34.0))
    enemy.is_alive = True
    enemy.is_structure = False

    bot = MagicMock()
    bot.player_id = 1
    bot.enemy_units = [ally, enemy]
    bot.enemy_structures = []
    bot.units = [ally]
    bot.structures = []

    unit = MagicMock()
    unit.type_id = UnitTypeId.MARINE
    unit.owner_id = 1
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is enemy
    unit.attack.assert_called_once_with(enemy)


def test_issue_attack_at_point_p2_ignores_allies_mixed_into_units():
    """After control_enemy both armies may sit in bot.units; P2 must still skip allies."""
    from sc2.position import Point2

    ally = MagicMock()
    ally.tag = 201
    ally.owner_id = 2
    ally.position = Point2((12.05, 34.0))
    ally.is_alive = True
    ally.is_structure = False

    trainee = MagicMock()
    trainee.tag = 101
    trainee.owner_id = 1
    trainee.position = Point2((12.4, 34.0))
    trainee.is_alive = True
    trainee.is_structure = False

    bot = MagicMock()
    bot.player_id = 1
    bot.units = [ally, trainee]
    bot.structures = []
    bot.enemy_units = []
    bot.enemy_structures = []

    unit = MagicMock()
    unit.type_id = UnitTypeId.MARINE
    unit.owner_id = 2
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is trainee
    unit.attack.assert_called_once_with(trainee)
    unit.move.assert_not_called()


def test_issue_attack_at_point_p2_empty_cell_attack_moves():
    """Empty-cell P2 must attack-move like P1 (move-only made them never shoot)."""
    from sc2.position import Point2

    bot = MagicMock()
    bot.player_id = 1
    bot.units = []
    bot.structures = []
    bot.enemy_units = []
    bot.enemy_structures = []

    unit = MagicMock()
    unit.type_id = UnitTypeId.MARINE
    unit.owner_id = 2
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is None
    unit.attack.assert_called_once_with(target)
    unit.move.assert_not_called()


def _mock_enemy(*, tag, position, is_alive=True, is_flying=False, is_structure=False, owner_id=2):
    enemy = MagicMock()
    enemy.tag = tag
    enemy.position = position
    enemy.is_alive = is_alive
    enemy.is_flying = is_flying
    enemy.is_structure = is_structure
    enemy.owner_id = owner_id
    return enemy


def test_uses_caustic_spray_on_structures_corruptor_only():
    assert uses_caustic_spray_on_structures(_mock_unit(UnitTypeId.CORRUPTOR)) is True
    assert uses_caustic_spray_on_structures(_mock_unit(UnitTypeId.MUTALISK)) is False
    assert uses_caustic_spray_on_structures(_mock_unit(UnitTypeId.MARINE)) is False
    assert uses_caustic_spray_on_structures(_mock_unit(UnitTypeId.BROODLORD)) is False


def test_issue_attack_at_point_corruptor_sprays_building():
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    pylon = _mock_enemy(
        tag=99,
        position=Point2((20.2, 20.1)),
        is_structure=True,
    )
    bot = MagicMock()
    bot.enemy_units = []
    bot.enemy_structures = [pylon]
    unit = MagicMock()
    unit.type_id = UnitTypeId.CORRUPTOR
    spray_cmd = object()
    unit.return_value = spray_cmd
    target = Point2((20.0, 20.0))

    assert issue_attack_at_point(bot, unit, target) is pylon
    unit.assert_called_once_with(AbilityId.CAUSTICSPRAY_CAUSTICSPRAY, pylon)
    bot.do.assert_called_once_with(spray_cmd)
    unit.attack.assert_not_called()


def test_issue_attack_at_point_corruptor_sprays_pylon_when_click_on_footprint():
    """Pylon is 2×2; a click 0.8 GU off center is still on the building."""
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    pylon = _mock_enemy(
        tag=99,
        position=Point2((20.0, 20.0)),
        is_structure=True,
    )
    pylon.radius = 1.125
    bot = MagicMock()
    bot.enemy_units = []
    bot.enemy_structures = [pylon]
    unit = MagicMock()
    unit.type_id = UnitTypeId.CORRUPTOR
    spray_cmd = object()
    unit.return_value = spray_cmd
    target = Point2((20.8, 20.0))

    assert issue_attack_at_point(bot, unit, target) is pylon
    unit.assert_called_once_with(AbilityId.CAUSTICSPRAY_CAUSTICSPRAY, pylon)
    bot.do.assert_called_once_with(spray_cmd)
    unit.attack.assert_not_called()


def test_issue_attack_at_point_corruptor_sprays_god_mode_pylon_in_own_structures():
    """After debug_control_enemy, P2 pylons often sit in bot.structures, not enemy_structures."""
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    pylon = _mock_enemy(
        tag=99,
        position=Point2((20.0, 20.0)),
        is_structure=True,
        owner_id=2,
    )
    pylon.radius = 1.125
    bot = MagicMock()
    bot.player_id = 1
    bot.units = []
    bot.structures = [pylon]
    bot.enemy_units = []
    bot.enemy_structures = []
    bot.all_units = [pylon]
    unit = MagicMock()
    unit.type_id = UnitTypeId.CORRUPTOR
    unit.owner_id = 1
    spray_cmd = object()
    unit.return_value = spray_cmd
    target = Point2((20.6, 20.2))

    assert issue_attack_at_point(bot, unit, target) is pylon
    unit.assert_called_once_with(AbilityId.CAUSTICSPRAY_CAUSTICSPRAY, pylon)
    bot.do.assert_called_once_with(spray_cmd)


def test_issue_attack_at_point_corruptor_does_not_spray_off_footprint():
    """Click outside the 2×2 pylon is still an air attack-move, not nearest-building."""
    from sc2.position import Point2

    pylon = _mock_enemy(
        tag=99,
        position=Point2((20.0, 20.0)),
        is_structure=True,
    )
    pylon.radius = 1.125
    bot = MagicMock()
    bot.enemy_units = []
    bot.enemy_structures = [pylon]
    unit = MagicMock()
    unit.type_id = UnitTypeId.CORRUPTOR
    target = Point2((21.3, 20.0))

    assert issue_attack_at_point(bot, unit, target) is None
    unit.attack.assert_called_once_with(target)
    unit.assert_not_called()


def test_issue_attack_at_point_corruptor_attacks_air_in_cell():
    from sc2.position import Point2

    viking = _mock_enemy(
        tag=7,
        position=Point2((12.2, 34.1)),
        is_flying=True,
    )
    bot = MagicMock()
    bot.enemy_units = [viking]
    bot.enemy_structures = []
    unit = MagicMock()
    unit.type_id = UnitTypeId.CORRUPTOR
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is viking
    unit.attack.assert_called_once_with(viking)
    unit.assert_not_called()


def test_issue_attack_at_point_corruptor_prefers_air_over_building():
    from sc2.position import Point2

    # Building is closer to the click than the air unit; air still wins.
    pylon = _mock_enemy(
        tag=1,
        position=Point2((10.05, 10.0)),
        is_structure=True,
    )
    mutalisk = _mock_enemy(
        tag=2,
        position=Point2((10.4, 10.0)),
        is_flying=True,
    )
    bot = MagicMock()
    bot.enemy_units = [mutalisk]
    bot.enemy_structures = [pylon]
    unit = MagicMock()
    unit.type_id = UnitTypeId.CORRUPTOR
    target = Point2((10.0, 10.0))

    assert issue_attack_at_point(bot, unit, target) is mutalisk
    unit.attack.assert_called_once_with(mutalisk)
    unit.assert_not_called()


def test_issue_attack_at_point_corruptor_sprays_building_when_only_ground_also_present():
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    marine = _mock_enemy(
        tag=3,
        position=Point2((10.1, 10.0)),
        is_flying=False,
        is_structure=False,
    )
    bunker = _mock_enemy(
        tag=4,
        position=Point2((10.4, 10.0)),
        is_structure=True,
    )
    bot = MagicMock()
    bot.enemy_units = [marine]
    bot.enemy_structures = [bunker]
    unit = MagicMock()
    unit.type_id = UnitTypeId.CORRUPTOR
    spray_cmd = object()
    unit.return_value = spray_cmd
    target = Point2((10.0, 10.0))

    assert issue_attack_at_point(bot, unit, target) is bunker
    unit.assert_called_once_with(AbilityId.CAUSTICSPRAY_CAUSTICSPRAY, bunker)
    unit.attack.assert_not_called()


def test_issue_attack_at_point_corruptor_ignores_ground_only_cell():
    from sc2.position import Point2

    marine = _mock_enemy(
        tag=3,
        position=Point2((12.2, 34.1)),
        is_flying=False,
        is_structure=False,
    )
    bot = MagicMock()
    bot.enemy_units = [marine]
    bot.enemy_structures = []
    unit = MagicMock()
    unit.type_id = UnitTypeId.CORRUPTOR
    target = Point2((12.0, 34.0))

    assert issue_attack_at_point(bot, unit, target) is None
    unit.attack.assert_called_once_with(target)
    unit.assert_not_called()


def test_issue_attack_at_point_marine_still_attacks_building():
    from sc2.position import Point2

    pylon = _mock_enemy(
        tag=99,
        position=Point2((20.2, 20.1)),
        is_structure=True,
    )
    bot = MagicMock()
    bot.enemy_units = []
    bot.enemy_structures = [pylon]
    unit = MagicMock()
    unit.type_id = UnitTypeId.MARINE
    target = Point2((20.0, 20.0))

    assert issue_attack_at_point(bot, unit, target) is pylon
    unit.attack.assert_called_once_with(pylon)
    unit.assert_not_called()


def test_resolve_corruptor_target_prefers_closer_air_when_several():
    from sc2.position import Point2

    near = _mock_enemy(tag=1, position=Point2((10.1, 10.0)), is_flying=True)
    far = _mock_enemy(tag=2, position=Point2((10.4, 10.0)), is_flying=True)
    pylon = _mock_enemy(tag=3, position=Point2((10.05, 10.0)), is_structure=True)
    bot = MagicMock()
    bot.enemy_units = [near, far]
    bot.enemy_structures = [pylon]

    assert resolve_corruptor_target_at_point(bot, Point2((10.0, 10.0))) is near


def test_uses_aoe_spell_only_casters():
    assert uses_aoe_spell(_mock_unit(UnitTypeId.INFESTOR)) is True
    assert uses_aoe_spell(_mock_unit(UnitTypeId.INFESTORBURROWED)) is True
    assert uses_aoe_spell(_mock_unit(UnitTypeId.HIGHTEMPLAR)) is True
    assert uses_aoe_spell(_mock_unit(UnitTypeId.GHOST)) is True
    assert uses_aoe_spell(_mock_unit(UnitTypeId.RAVAGER)) is True
    assert uses_aoe_spell(_mock_unit(UnitTypeId.VIPER)) is True
    assert uses_aoe_spell(_mock_unit(UnitTypeId.MARINE)) is False
    assert uses_aoe_spell(_mock_unit(UnitTypeId.RAVAGERBURROWED)) is False


def _caster(uid, energy=200.0):
    unit = _mock_unit(uid)
    unit.energy = energy
    return unit


def _armed(uid, *, ground_dps=10.0, is_burrowed=False, weapon_ready=True):
    unit = _mock_unit(uid)
    unit.ground_dps = ground_dps
    unit.air_dps = 0.0
    unit.is_burrowed = is_burrowed
    unit.weapon_ready = weapon_ready
    unit.weapon_cooldown = 0.0 if weapon_ready else 8.0
    return unit


def test_weapon_ready_ignores_ability_cache():
    """Attack cooldown / retreat read proto weapon_cooldown, not spell abilities."""
    from src.utils.unit_feature_helpers import unit_can_attack, unit_weapon_ready

    cooling = _armed(UnitTypeId.MARINE, weapon_ready=False)
    cooling.weapon_cooldown = 8.0
    cooling.tag = 1
    bot = MagicMock()
    bot._available_abilities_by_tag = {1: frozenset()}
    cooling._bot = bot
    assert unit_weapon_ready(cooling) is False
    assert unit_can_attack(cooling) is False
    ready = _armed(UnitTypeId.MARINE, weapon_ready=True)
    ready.weapon_cooldown = 0.0
    ready.tag = 1
    assert unit_weapon_ready(ready) is True
    assert unit_can_attack(ready) is True


def test_unit_can_attack_excludes_burrowed_non_lurkers():
    from sc2.ids.upgrade_id import UpgradeId

    assert unit_can_attack(_armed(UnitTypeId.ZERGLING)) is True
    assert unit_can_attack(
        _armed(UnitTypeId.ZERGLINGBURROWED, is_burrowed=True)
    ) is False
    # Tunneling Claws make a burrowed Roach mobile, but it still cannot fire.
    roach = _armed(UnitTypeId.ROACHBURROWED, is_burrowed=True)
    assert unit_can_attack(roach, frozenset({UpgradeId.TUNNELINGCLAWS})) is False
    assert unit_can_attack(_armed(UnitTypeId.MARINE, weapon_ready=False)) is False


def test_unit_can_attack_includes_weaponless_ability_attackers():
    # Swarm Host / Disruptor have no DPS; attack remaps to Spawn Locusts /
    # Purification Nova. Without this gate, attack_local stays masked forever.
    host = _mock_unit(UnitTypeId.SWARMHOSTMP)
    host.ground_dps = 0.0
    host.air_dps = 0.0
    assert unit_can_attack(host) is True
    burrowed = _mock_unit(UnitTypeId.SWARMHOSTBURROWEDMP)
    burrowed.ground_dps = 0.0
    burrowed.air_dps = 0.0
    burrowed.is_burrowed = True
    assert unit_can_attack(burrowed) is True
    disruptor = _mock_unit(UnitTypeId.DISRUPTOR)
    disruptor.ground_dps = 0.0
    disruptor.air_dps = 0.0
    assert unit_can_attack(disruptor) is True
    # Unarmed medivac still cannot attack.
    medivac = _mock_unit(UnitTypeId.MEDIVAC)
    medivac.ground_dps = 0.0
    medivac.air_dps = 0.0
    assert unit_can_attack(medivac) is False


def test_unit_can_attack_keeps_lurker_and_planted_forms():
    lurker = _armed(UnitTypeId.LURKERMPBURROWED, is_burrowed=True)
    assert unit_can_attack(lurker) is True
    assert unit_can_attack(_armed(UnitTypeId.LURKERMP, is_burrowed=False)) is False
    assert unit_can_attack(_armed(UnitTypeId.SIEGETANKSIEGED)) is True
    assert unit_can_attack(
        _armed(UnitTypeId.WIDOWMINEBURROWED, is_burrowed=True)
    ) is True


def test_unit_can_aoe_energy_and_ghost_vs_protoss_and_zerg():
    from sc2.ids.upgrade_id import UpgradeId

    storm = frozenset({UpgradeId.PSISTORMTECH})
    assert unit_can_aoe(_caster(UnitTypeId.HIGHTEMPLAR, 75.0), upgrades=storm) is True
    assert unit_can_aoe(_caster(UnitTypeId.HIGHTEMPLAR, 75.0), upgrades=frozenset()) is False
    assert unit_can_aoe(_caster(UnitTypeId.HIGHTEMPLAR, 74.0), upgrades=storm) is False
    assert unit_can_aoe(_caster(UnitTypeId.INFESTOR, 75.0)) is True
    assert unit_can_aoe(_caster(UnitTypeId.INFESTORBURROWED, 75.0)) is True
    assert unit_can_aoe(_caster(UnitTypeId.VIPER, 125.0)) is True
    assert unit_can_aoe(_caster(UnitTypeId.VIPER, 125.0), has_tactical_air_enemy=False) is False
    assert unit_can_aoe(_caster(UnitTypeId.VIPER, 124.0)) is False
    assert unit_can_aoe(_caster(UnitTypeId.RAVAGER, 0.0)) is True
    assert unit_can_aoe(_caster(UnitTypeId.MARINE, 200.0)) is False

    ghost = _caster(UnitTypeId.GHOST, 75.0)
    assert unit_can_aoe(ghost, enemy_race="protoss") is True
    assert unit_can_aoe(ghost, enemy_race="zerg") is True
    assert unit_can_aoe(ghost, enemy_race="Zerg") is True
    assert unit_can_aoe(ghost, enemy_race="terran") is False
    assert unit_can_aoe(ghost) is False
    assert unit_can_aoe(_caster(UnitTypeId.GHOST, 74.0), enemy_race="zerg") is False
    assert unit_can_aoe(_caster(UnitTypeId.GHOST, 74.0), enemy_race="protoss") is False


def test_unit_can_aoe_respects_ability_cooldown_cache():
    from sc2.ids.ability_id import AbilityId

    ravager = _caster(UnitTypeId.RAVAGER, 0.0)
    ravager.tag = 42
    bot = MagicMock()
    bot._available_abilities_by_tag = {42: {AbilityId.EFFECT_CORROSIVEBILE}}
    assert unit_can_aoe(ravager, bot=bot) is True

    bot._available_abilities_by_tag = {42: set()}
    assert unit_can_aoe(ravager, bot=bot) is False


def test_unit_is_neuraled_is_the_control_buff_only():
    from sc2.ids.buff_id import BuffId
    from src.utils.unit_feature_helpers import unit_is_neuraled

    stolen = MagicMock()
    stolen.buffs = {BuffId.NEURALPARASITE}
    caster = MagicMock()
    caster.buffs = {BuffId.NEURALPARASITECHILDREN}
    windup = MagicMock()
    windup.buffs = [BuffId.NEURALPARASITEWAIT]
    raw = MagicMock()
    raw.buffs = [BuffId.NEURALPARASITE.value]
    plain = MagicMock()
    plain.buffs = []

    assert unit_is_neuraled(stolen) is True
    assert unit_is_neuraled(raw) is True
    assert unit_is_neuraled(caster) is False
    assert unit_is_neuraled(windup) is False
    assert unit_is_neuraled(plain) is False
    assert unit_is_neuraled(MagicMock()) is False
    assert unit_is_neuraled(None) is False


def test_unit_neural_remaining_tracks_the_visible_timer():
    from sc2.ids.buff_id import BuffId
    from src.utils.unit_feature_helpers import unit_neural_remaining

    stolen = MagicMock()
    stolen.buffs = {BuffId.NEURALPARASITE}
    stolen.buff_duration_remain = 336
    stolen.buff_duration_max = 336
    assert unit_neural_remaining(stolen) == pytest.approx(1.0)

    stolen.buff_duration_remain = 84
    assert unit_neural_remaining(stolen) == pytest.approx(0.25)

    stolen.buff_duration_remain = 0
    assert unit_neural_remaining(stolen) == pytest.approx(0.0)

    # Buff still up, client reported no timer bar: stay at 1 until it drops.
    stolen.buff_duration_remain = 0
    stolen.buff_duration_max = 0
    assert unit_neural_remaining(stolen) == pytest.approx(1.0)

    # A stim timer on a unit that is not neuralled must not fill the channel.
    marine = MagicMock()
    marine.buffs = []
    marine.buff_duration_remain = 50
    marine.buff_duration_max = 100
    assert unit_neural_remaining(marine) == pytest.approx(0.0)
    assert unit_neural_remaining(None) == pytest.approx(0.0)


def test_unit_can_snipe_energy_upgrade_and_types():
    from sc2.ids.upgrade_id import UpgradeId
    from src.utils.unit_feature_helpers import (
        snipe_target_is_eligible,
        unit_can_snipe,
    )

    neural = frozenset({UpgradeId.NEURALPARASITE})
    yamato = frozenset({UpgradeId.YAMATOCANNON})
    assert unit_can_snipe(_caster(UnitTypeId.GHOST, 50.0)) is True
    assert unit_can_snipe(_caster(UnitTypeId.GHOST, 49.0)) is False
    assert unit_can_snipe(_caster(UnitTypeId.BATTLECRUISER, 0.0), upgrades=yamato) is True
    assert unit_can_snipe(_caster(UnitTypeId.BATTLECRUISER, 0.0), upgrades=frozenset()) is False
    assert unit_can_snipe(_caster(UnitTypeId.HIGHTEMPLAR, 50.0)) is True
    assert unit_can_snipe(_caster(UnitTypeId.HIGHTEMPLAR, 49.0)) is False
    assert unit_can_snipe(_caster(UnitTypeId.INFESTOR, 100.0), upgrades=neural) is True
    assert unit_can_snipe(_caster(UnitTypeId.INFESTOR, 100.0), upgrades=frozenset()) is False
    assert unit_can_snipe(
        _caster(UnitTypeId.INFESTORBURROWED, 100.0), upgrades=neural
    ) is True
    assert unit_can_snipe(_caster(UnitTypeId.MARINE, 200.0)) is False

    ghost = _caster(UnitTypeId.GHOST, 50.0)
    bio = MagicMock()
    bio.is_biological = True
    bio.is_structure = False
    mech = MagicMock()
    mech.is_biological = False
    mech.is_structure = False
    energy_tgt = MagicMock()
    energy_tgt.is_structure = False
    energy_tgt.energy_max = 200.0
    no_energy = MagicMock()
    no_energy.is_structure = False
    no_energy.energy_max = 0.0
    assert snipe_target_is_eligible(ghost, bio) is True
    assert snipe_target_is_eligible(ghost, mech) is False
    ht = _caster(UnitTypeId.HIGHTEMPLAR, 50.0)
    assert snipe_target_is_eligible(ht, energy_tgt) is True
    assert snipe_target_is_eligible(ht, no_energy) is False


def _bot_at_loop(game_loop: int) -> MagicMock:
    bot = MagicMock()
    bot.state = MagicMock()
    bot.state.game_loop = game_loop
    return bot


def test_local_snipe_stamp_gates_until_ready():
    import math
    from sc2.ids.upgrade_id import UpgradeId
    from src.utils.unit_feature_helpers import (
        GAME_LOOPS_PER_SECOND,
        SNIPE_ABILITY_COOLDOWN_SECONDS,
        mark_spell_ability_cooldown,
        unit_can_snipe,
    )

    yamato = frozenset({UpgradeId.YAMATOCANNON})
    bc = _caster(UnitTypeId.BATTLECRUISER, 0.0)
    bc.tag = 31
    bot = _bot_at_loop(100)
    assert unit_can_snipe(bc, upgrades=yamato, bot=bot) is True

    mark_spell_ability_cooldown(bot, bc, "snipe")
    assert unit_can_snipe(bc, upgrades=yamato, bot=bot) is False

    loops = int(math.ceil(SNIPE_ABILITY_COOLDOWN_SECONDS[UnitTypeId.BATTLECRUISER] * GAME_LOOPS_PER_SECOND))
    bot.state.game_loop = 100 + loops
    assert unit_can_snipe(bc, upgrades=yamato, bot=bot) is True


def test_local_aoe_stamp_is_separate_from_snipe():
    from src.utils.unit_feature_helpers import mark_spell_ability_cooldown, unit_can_snipe

    ghost = _caster(UnitTypeId.GHOST, 75.0)
    ghost.tag = 32
    bot = _bot_at_loop(50)
    mark_spell_ability_cooldown(bot, ghost, "snipe")
    assert unit_can_snipe(ghost, bot=bot) is False
    assert unit_can_aoe(ghost, enemy_race="protoss", bot=bot) is True

    mark_spell_ability_cooldown(bot, ghost, "aoe")
    assert unit_can_aoe(ghost, enemy_race="protoss", bot=bot) is False


def test_issue_snipe_stamps_cooldown_by_default():
    from src.utils.unit_feature_helpers import issue_snipe_on_unit, unit_can_snipe

    ghost = _caster(UnitTypeId.GHOST, 50.0)
    ghost.tag = 33
    ghost.owner_id = 1
    target = MagicMock()
    target.owner_id = 2
    target.is_structure = False
    target.is_biological = True
    bot = _bot_at_loop(20)

    assert issue_snipe_on_unit(bot, ghost, target) is True
    assert 33 in bot._snipe_ability_ready_after_loop
    assert unit_can_snipe(ghost, bot=bot) is False


def test_stolen_battlecruisers_can_yamato_each_other_but_not_real_friendlies():
    from sc2.ids.buff_id import BuffId
    from src.utils.unit_feature_helpers import issue_snipe_on_unit

    def _bc(tag):
        unit = MagicMock()
        unit.tag = tag
        unit.type_id = UnitTypeId.BATTLECRUISER
        unit.owner_id = 1
        unit.is_structure = False
        unit.buffs = {BuffId.NEURALPARASITE}
        return unit

    first = _bc(1)
    second = _bc(2)
    friendly = MagicMock()
    friendly.tag = 3
    friendly.type_id = UnitTypeId.MARINE
    friendly.owner_id = 1
    friendly.is_structure = False
    friendly.buffs = set()
    bot = MagicMock()

    assert issue_snipe_on_unit(bot, first, second, stamp_cooldown=False) is True
    assert issue_snipe_on_unit(bot, first, friendly, stamp_cooldown=False) is False
    assert issue_snipe_on_unit(bot, first, first, stamp_cooldown=False) is False


def test_issue_aoe_reissue_does_not_restamp():
    from sc2.position import Point2
    from src.utils.unit_feature_helpers import unit_can_aoe

    ravager = _caster(UnitTypeId.RAVAGER, 0.0)
    ravager.tag = 34
    ravager.owner_id = 1
    bot = _bot_at_loop(20)
    target = Point2((8.0, 9.0))

    assert issue_aoe_at_point(bot, ravager, target, stamp_cooldown=False) is True
    store = getattr(bot, "_aoe_ability_ready_after_loop", None)
    assert not isinstance(store, dict) or 34 not in store
    assert unit_can_aoe(ravager, bot=bot) is True

    assert issue_aoe_at_point(bot, ravager, target) is True
    assert 34 in bot._aoe_ability_ready_after_loop
    assert unit_can_aoe(ravager, bot=bot) is False


def test_refresh_ability_cache_queries_aoe_with_resources_ignored():
    """AoE CD query must not double-gate energy (cloak stayed open, AoE closed)."""
    import asyncio
    from sc2.ids.ability_id import AbilityId
    from src.utils.unit_feature_helpers import refresh_healer_ability_cache

    infestor = _caster(UnitTypeId.INFESTOR, 75.0)
    infestor.tag = 7
    medivac = _caster(UnitTypeId.MEDIVAC, 50.0)
    medivac.tag = 8

    async def query(units, ignore_resource_requirements=False):
        assert ignore_resource_requirements is True
        out = {}
        for u in units:
            if u.type_id == UnitTypeId.INFESTOR:
                out[u.tag] = {AbilityId.FUNGALGROWTH_FUNGALGROWTH}
            else:
                out[u.tag] = {AbilityId.MEDIVACHEAL_HEAL}
        return out

    bot = MagicMock()
    bot.all_units = [infestor, medivac]
    bot.client = MagicMock()
    bot.client.query_available_abilities_with_tag = query

    asyncio.run(refresh_healer_ability_cache(bot))
    assert isinstance(bot._available_abilities_by_tag[7], frozenset)
    assert isinstance(bot._available_abilities_by_tag[8], frozenset)
    assert AbilityId.FUNGALGROWTH_FUNGALGROWTH in bot._available_abilities_by_tag[7]
    assert AbilityId.MEDIVACHEAL_HEAL in bot._available_abilities_by_tag[8]


def test_refresh_ability_cache_skips_snipe_only_when_flag_off():
    """Battlecruiser is snipe-only — do not query it before the snipe head exists."""
    import asyncio
    from sc2.ids.ability_id import AbilityId
    from src.utils.unit_feature_helpers import refresh_healer_ability_cache

    bc = _caster(UnitTypeId.BATTLECRUISER, 0.0)
    bc.tag = 9
    ravager = _caster(UnitTypeId.RAVAGER, 0.0)
    ravager.tag = 10
    queried = []

    async def query(units, ignore_resource_requirements=False):
        queried.extend(list(units))
        return {u.tag: {AbilityId.EFFECT_CORROSIVEBILE} for u in units}

    bot = MagicMock()
    bot.all_units = [bc, ravager]
    bot.client = MagicMock()
    bot.client.query_available_abilities_with_tag = query

    asyncio.run(
        refresh_healer_ability_cache(bot, include_aoe=True, include_snipe=False)
    )
    assert ravager in queried
    assert bc not in queried
    assert 9 not in bot._available_abilities_by_tag
    assert 10 in bot._available_abilities_by_tag


def test_refresh_ability_cache_skips_aoe_only_when_flag_off():
    import asyncio
    from src.utils.unit_feature_helpers import refresh_healer_ability_cache

    ravager = _caster(UnitTypeId.RAVAGER, 0.0)
    ravager.tag = 11
    bc = _caster(UnitTypeId.BATTLECRUISER, 0.0)
    bc.tag = 12
    queried = []

    async def query(units, ignore_resource_requirements=False):
        queried.extend(list(units))
        return {u.tag: set() for u in units}

    bot = MagicMock()
    bot.all_units = [ravager, bc]
    bot.client = MagicMock()
    bot.client.query_available_abilities_with_tag = query

    asyncio.run(
        refresh_healer_ability_cache(bot, include_aoe=False, include_snipe=True)
    )
    assert bc in queried
    assert ravager not in queried
    assert 12 in bot._available_abilities_by_tag
    assert 11 not in bot._available_abilities_by_tag


def test_refresh_ability_cache_skips_heal_when_flag_off():
    import asyncio
    from src.utils.unit_feature_helpers import refresh_healer_ability_cache

    medivac = _caster(UnitTypeId.MEDIVAC, 50.0)
    medivac.tag = 13
    ravager = _caster(UnitTypeId.RAVAGER, 0.0)
    ravager.tag = 14
    queried = []

    async def query(units, ignore_resource_requirements=False):
        queried.extend(list(units))
        return {u.tag: set() for u in units}

    bot = MagicMock()
    bot.all_units = [medivac, ravager]
    bot.client = MagicMock()
    bot.client.query_available_abilities_with_tag = query

    asyncio.run(
        refresh_healer_ability_cache(
            bot, include_heal=False, include_aoe=True, include_snipe=False
        )
    )
    assert ravager in queried
    assert medivac not in queried
    assert 14 in bot._available_abilities_by_tag
    assert 13 not in bot._available_abilities_by_tag


def test_refresh_ability_cache_single_rpc_and_units_arg():
    import asyncio
    from src.utils.unit_feature_helpers import refresh_healer_ability_cache

    medivac = _caster(UnitTypeId.MEDIVAC, 50.0)
    medivac.tag = 15
    ghost = _caster(UnitTypeId.GHOST, 75.0)
    ghost.tag = 16
    leftover = _caster(UnitTypeId.INFESTOR, 75.0)
    leftover.tag = 17
    calls = []

    async def query(units, ignore_resource_requirements=False):
        calls.append(list(units))
        return {u.tag: set() for u in units}

    bot = MagicMock()
    bot.all_units = [medivac, ghost, leftover]
    bot.client = MagicMock()
    bot.client.query_available_abilities_with_tag = query

    asyncio.run(
        refresh_healer_ability_cache(
            bot,
            include_heal=True,
            include_aoe=True,
            include_snipe=True,
            units=[medivac, ghost],
        )
    )
    assert len(calls) == 1
    assert leftover not in calls[0]
    assert {u.tag for u in calls[0]} == {15, 16}


def test_issue_aoe_at_point_ravager_bile():
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    bot = MagicMock()
    unit = MagicMock()
    unit.type_id = UnitTypeId.RAVAGER
    unit.owner_id = 1
    cast_cmd = object()
    unit.return_value = cast_cmd
    target = Point2((12.0, 34.0))

    assert issue_aoe_at_point(bot, unit, target) is True
    unit.assert_called_once_with(AbilityId.EFFECT_CORROSIVEBILE, target)
    bot.do.assert_called_once_with(cast_cmd)
    unit.attack.assert_not_called()
    unit.move.assert_not_called()


def test_issue_aoe_at_point_viper_bombs_air():
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    muta = _mock_enemy(
        tag=7,
        position=Point2((12.2, 34.1)),
        is_flying=True,
    )
    bot = MagicMock()
    bot.player_id = 1
    bot.enemy_units = [muta]
    bot.enemy_structures = []
    unit = MagicMock()
    unit.type_id = UnitTypeId.VIPER
    unit.owner_id = 1
    target = Point2((12.0, 34.0))

    assert issue_aoe_at_point(bot, unit, target) is True
    unit.assert_called_once_with(AbilityId.PARASITICBOMB_PARASITICBOMB, muta)


def test_issue_aoe_at_point_viper_opponent_uses_bot_units():
    from sc2.ids.ability_id import AbilityId
    from sc2.position import Point2

    muta = _mock_enemy(
        tag=7,
        position=Point2((12.2, 34.1)),
        is_flying=True,
        owner_id=1,
    )
    bot = MagicMock()
    bot.player_id = 1
    bot.units = [muta]
    bot.structures = []
    bot.enemy_units = []
    bot.enemy_structures = []
    unit = MagicMock()
    unit.type_id = UnitTypeId.VIPER
    unit.owner_id = 2
    target = Point2((12.0, 34.0))

    assert issue_aoe_at_point(bot, unit, target) is True
    unit.assert_called_once_with(AbilityId.PARASITICBOMB_PARASITICBOMB, muta)


def test_refresh_ability_cache_skips_rpc_when_all_families_off():
    import asyncio
    from src.utils.unit_feature_helpers import refresh_healer_ability_cache

    ghost = _caster(UnitTypeId.GHOST, 75.0)
    ghost.tag = 99
    queried = []

    async def query(units, ignore_resource_requirements=False):
        queried.extend(list(units))
        return {u.tag: set() for u in units}

    bot = MagicMock()
    bot.all_units = [ghost]
    bot.client = MagicMock()
    bot.client.query_available_abilities_with_tag = query
    bot._available_abilities_by_tag = {99: frozenset()}

    asyncio.run(
        refresh_healer_ability_cache(
            bot, include_heal=False, include_aoe=False, include_snipe=False
        )
    )
    assert queried == []
    assert bot._available_abilities_by_tag == {}


def test_issue_aoe_at_point_marine_noop():
    from sc2.position import Point2

    bot = MagicMock()
    unit = MagicMock()
    unit.type_id = UnitTypeId.MARINE
    target = Point2((12.0, 34.0))

    assert issue_aoe_at_point(bot, unit, target) is False
    bot.do.assert_not_called()
    unit.attack.assert_not_called()
    unit.move.assert_not_called()


def test_borrowed_spellcaster_tags_and_command_gate():
    from sc2.ids.buff_id import BuffId
    from src.agent.squad_constants import MAX_SQUAD_UNITS
    from src.utils.unit_feature_helpers import (
        borrowed_neural_spellcaster_tags,
        neural_command_allowed,
        squad_pointer_tags,
    )

    templar = MagicMock()
    templar.tag = 20
    templar.type_id = UnitTypeId.HIGHTEMPLAR
    templar.owner_id = 1
    templar.is_structure = False
    templar.buffs = {BuffId.NEURALPARASITE}
    zealot = MagicMock()
    zealot.tag = 21
    zealot.type_id = UnitTypeId.ZEALOT
    zealot.owner_id = 1
    zealot.is_structure = False
    zealot.buffs = {BuffId.NEURALPARASITE}

    assert borrowed_neural_spellcaster_tags([templar, zealot], 1, [10]) == [20]
    assert borrowed_neural_spellcaster_tags([templar], 2, [10]) == []
    assert borrowed_neural_spellcaster_tags([templar], 1, [20]) == []

    tags = squad_pointer_tags([30, 10], [20])
    assert tags == [10, 30, 20]
    full = squad_pointer_tags(list(range(MAX_SQUAD_UNITS)), [99])
    assert 99 not in full
    assert len(full) == MAX_SQUAD_UNITS

    assert neural_command_allowed(templar, 1, "aoe_local") is True
    assert neural_command_allowed(templar, 1, "snipe") is True
    assert neural_command_allowed(templar, 1, "move") is False
    assert neural_command_allowed(templar, 1, "attack_unit") is False
    assert neural_command_allowed(templar, 2, "aoe_local") is False
    assert neural_command_allowed(templar, 2, "move") is True
