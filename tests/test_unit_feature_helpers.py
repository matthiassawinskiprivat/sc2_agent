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
    unit_center_in_spatial_cell,
    uses_aoe_spell,
    unit_can_aoe,
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


def test_unit_can_aoe_energy_and_ghost_vs_protoss():
    assert unit_can_aoe(_caster(UnitTypeId.HIGHTEMPLAR, 75.0)) is True
    assert unit_can_aoe(_caster(UnitTypeId.HIGHTEMPLAR, 74.0)) is False
    assert unit_can_aoe(_caster(UnitTypeId.INFESTOR, 75.0)) is True
    assert unit_can_aoe(_caster(UnitTypeId.INFESTORBURROWED, 75.0)) is True
    assert unit_can_aoe(_caster(UnitTypeId.VIPER, 125.0)) is True
    assert unit_can_aoe(_caster(UnitTypeId.VIPER, 124.0)) is False
    assert unit_can_aoe(_caster(UnitTypeId.RAVAGER, 0.0)) is True
    assert unit_can_aoe(_caster(UnitTypeId.MARINE, 200.0)) is False

    ghost = _caster(UnitTypeId.GHOST, 75.0)
    assert unit_can_aoe(ghost, enemy_race="protoss") is True
    assert unit_can_aoe(ghost, enemy_race="zerg") is False
    assert unit_can_aoe(ghost, enemy_race="terran") is False
    assert unit_can_aoe(ghost) is False
    assert unit_can_aoe(_caster(UnitTypeId.GHOST, 74.0), enemy_race="protoss") is False


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

    assert issue_aoe_at_point(bot, unit, target) is None
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

    assert issue_aoe_at_point(bot, unit, target) is muta
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

    assert issue_aoe_at_point(bot, unit, target) is muta
    unit.assert_called_once_with(AbilityId.PARASITICBOMB_PARASITICBOMB, muta)


def test_issue_aoe_at_point_marine_noop():
    from sc2.position import Point2

    bot = MagicMock()
    unit = MagicMock()
    unit.type_id = UnitTypeId.MARINE
    target = Point2((12.0, 34.0))

    assert issue_aoe_at_point(bot, unit, target) is None
    bot.do.assert_not_called()
    unit.attack.assert_not_called()
    unit.move.assert_not_called()
