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
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.BROODLORD)) is False
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.SWARMHOSTMP)) is False
    assert is_temporary_attrition_unit(_mock_unit(UnitTypeId.MARINE)) is False
