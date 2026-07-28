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
    is_force_melee_unit,
    is_temporary_attrition_unit,
    issue_attack_at_point,
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
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.BROODLORD)) is False
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.SWARMHOSTMP)) is False
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.DISRUPTOR)) is False
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.MARINE)) is False


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


def test_issue_attack_at_point_marine_uses_attack():
    from sc2.position import Point2

    bot = MagicMock()
    unit = MagicMock()
    unit.type_id = UnitTypeId.MARINE
    target = Point2((12.0, 34.0))

    issue_attack_at_point(bot, unit, target)

    unit.attack.assert_called_once_with(target)
    bot.do.assert_called_once()
    unit.assert_not_called()
