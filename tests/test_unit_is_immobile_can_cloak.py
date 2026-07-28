"""is_immobile and can_cloak unit scalar helpers."""
from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from src.utils.unit_feature_helpers import unit_can_cloak, unit_is_immobile


def _unit(type_id, *, is_burrowed=False, is_cloaked=False, energy=0.0):
    u = MagicMock()
    u.type_id = type_id
    u.is_burrowed = is_burrowed
    u.is_cloaked = is_cloaked
    u.energy = energy
    return u


def test_is_immobile_sieged_tank():
    assert unit_is_immobile(_unit(UnitTypeId.SIEGETANKSIEGED)) is True


def test_is_immobile_widow_mine_burrowed():
    assert unit_is_immobile(_unit(UnitTypeId.WIDOWMINEBURROWED)) is True


def test_is_immobile_burrowed_zergling():
    u = _unit(UnitTypeId.ZERGLINGBURROWED, is_burrowed=True)
    assert unit_is_immobile(u) is True


def test_is_immobile_roach_burrowed_without_claws():
    u = _unit(UnitTypeId.ROACHBURROWED, is_burrowed=True)
    assert unit_is_immobile(u, frozenset()) is True


def test_is_immobile_roach_burrowed_with_claws():
    u = _unit(UnitTypeId.ROACHBURROWED, is_burrowed=True)
    assert unit_is_immobile(u, frozenset({UpgradeId.TUNNELINGCLAWS})) is False


def test_is_immobile_swarm_host_burrowed():
    u = _unit(UnitTypeId.SWARMHOSTBURROWEDMP, is_burrowed=True)
    assert unit_is_immobile(u) is False


def test_is_immobile_cloaked_ghost():
    u = _unit(UnitTypeId.GHOST, is_cloaked=True)
    assert unit_is_immobile(u) is False


def test_can_cloak_ghost_with_energy():
    u = _unit(UnitTypeId.GHOST, energy=50.0)
    assert unit_can_cloak(u) is True


def test_can_cloak_ghost_already_cloaked():
    u = _unit(UnitTypeId.GHOST, is_cloaked=True, energy=50.0)
    assert unit_can_cloak(u) is False


def test_can_cloak_ghost_low_energy():
    u = _unit(UnitTypeId.GHOST, energy=10.0)
    assert unit_can_cloak(u) is False


def test_can_cloak_zergling_and_dt():
    assert unit_can_cloak(_unit(UnitTypeId.ZERGLING, energy=100.0)) is False
    assert unit_can_cloak(_unit(UnitTypeId.DARKTEMPLAR, energy=0.0)) is False


def test_can_repair_scv_only():
    from src.utils.unit_feature_helpers import unit_can_repair

    assert unit_can_repair(_unit(UnitTypeId.SCV)) is True
    assert unit_can_repair(_unit(UnitTypeId.MEDIVAC, energy=50.0)) is False
    assert unit_can_repair(_unit(UnitTypeId.MARINE)) is False


def test_can_heal_medivac_queen_battery():
    from src.utils.unit_feature_helpers import unit_can_heal

    assert unit_can_heal(_unit(UnitTypeId.MEDIVAC, energy=10.0)) is True
    assert unit_can_heal(_unit(UnitTypeId.MEDIVAC, energy=2.0)) is False
    assert unit_can_heal(_unit(UnitTypeId.QUEEN, energy=50.0)) is True
    assert unit_can_heal(_unit(UnitTypeId.QUEEN, energy=10.0)) is False
    assert unit_can_heal(_unit(UnitTypeId.SHIELDBATTERY, energy=10.0)) is True
    assert unit_can_heal(_unit(UnitTypeId.SHIELDBATTERY, energy=1.0)) is False
    assert unit_can_heal(_unit(UnitTypeId.SCV)) is False
