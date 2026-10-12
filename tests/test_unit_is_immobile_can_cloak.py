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
    from sc2.ids.upgrade_id import UpgradeId

    u = _unit(UnitTypeId.GHOST, energy=50.0)
    assert unit_can_cloak(u, frozenset({UpgradeId.PERSONALCLOAKING})) is True
    assert unit_can_cloak(u, frozenset()) is False


def test_can_cloak_ghost_already_cloaked():
    from sc2.ids.upgrade_id import UpgradeId

    u = _unit(UnitTypeId.GHOST, is_cloaked=True, energy=50.0)
    assert unit_can_cloak(u, frozenset({UpgradeId.PERSONALCLOAKING})) is False


def test_can_cloak_ghost_low_energy():
    from sc2.ids.upgrade_id import UpgradeId

    u = _unit(UnitTypeId.GHOST, energy=10.0)
    assert unit_can_cloak(u, frozenset({UpgradeId.PERSONALCLOAKING})) is False


def test_can_cloak_zergling_and_dt():
    assert unit_can_cloak(_unit(UnitTypeId.ZERGLING, energy=100.0)) is False
    assert unit_can_cloak(_unit(UnitTypeId.DARKTEMPLAR, energy=0.0)) is False


def test_can_burrow_requires_tech_except_widow_mine():
    from sc2.ids.upgrade_id import UpgradeId
    from src.utils.unit_feature_helpers import unit_can_burrow

    ling = _unit(UnitTypeId.ZERGLING)
    assert unit_can_burrow(ling, frozenset()) is False
    assert unit_can_burrow(ling, frozenset({UpgradeId.BURROW})) is True
    assert unit_can_burrow(_unit(UnitTypeId.WIDOWMINE), frozenset()) is True
    assert unit_can_burrow(_unit(UnitTypeId.ZERGLINGBURROWED, is_burrowed=True)) is False


def test_can_reveal_cloaked_or_burrowed():
    from src.utils.unit_feature_helpers import unit_can_reveal

    ghost = _unit(UnitTypeId.GHOST, is_cloaked=True, energy=50.0)
    assert unit_can_reveal(ghost) is True
    assert unit_can_reveal(_unit(UnitTypeId.GHOST, energy=50.0)) is False
    assert unit_can_reveal(_unit(UnitTypeId.ZERGLINGBURROWED, is_burrowed=True)) is True
    assert unit_can_reveal(_unit(UnitTypeId.ZERGLING)) is False


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


def test_can_heal_uses_sc2_ability_cache_and_cooldown():
    from sc2.ids.ability_id import AbilityId
    from src.utils.unit_feature_helpers import unit_can_heal

    queen = _unit(UnitTypeId.QUEEN, energy=200.0)
    queen.tag = 7
    queen.is_using_ability = lambda abilities: False
    bot = MagicMock()
    bot.player_id = 1
    bot.state = MagicMock()
    bot.state.game_loop = 100
    bot._available_abilities_by_tag = {7: {AbilityId.TRANSFUSION_TRANSFUSION}}
    bot._heal_ability_ready_after_loop = {7: 150}

    assert unit_can_heal(queen, bot) is False

    bot.state.game_loop = 150
    assert unit_can_heal(queen, bot) is True

    bot._available_abilities_by_tag = {7: set()}
    assert unit_can_heal(queen, bot) is False


def test_can_heal_false_while_channeling():
    from sc2.ids.ability_id import AbilityId
    from src.utils.unit_feature_helpers import unit_can_heal

    queen = _unit(UnitTypeId.QUEEN, energy=200.0)
    queen.tag = 8
    queen.is_using_ability = (
        lambda abilities: AbilityId.TRANSFUSION_TRANSFUSION in abilities
    )
    bot = MagicMock()
    bot._available_abilities_by_tag = {8: {AbilityId.TRANSFUSION_TRANSFUSION}}

    assert unit_can_heal(queen, bot) is False


def test_can_repair_requires_minerals_without_ability_cache():
    from src.utils.unit_feature_helpers import unit_can_repair

    scv = _unit(UnitTypeId.SCV)
    scv.tag = 3
    scv.owner_id = 1
    scv.is_using_ability = lambda abilities: False
    bot = MagicMock()
    bot.player_id = 1
    bot.minerals = 0
    bot._available_abilities_by_tag = {}

    assert unit_can_repair(scv, bot) is False

    bot.minerals = 50
    assert unit_can_repair(scv, bot) is True

    other = _unit(UnitTypeId.SCV)
    other.tag = 9
    other.owner_id = 2
    other.is_using_ability = lambda abilities: False
    assert unit_can_repair(other, bot) is False


def test_can_repair_prefers_sc2_ability_cache():
    from sc2.ids.ability_id import AbilityId
    from src.utils.unit_feature_helpers import unit_can_repair

    scv = _unit(UnitTypeId.SCV)
    scv.tag = 4
    scv.owner_id = 1
    bot = MagicMock()
    bot.player_id = 1
    bot.minerals = 0
    bot._available_abilities_by_tag = {4: {AbilityId.EFFECT_REPAIR_SCV}}

    assert unit_can_repair(scv, bot) is True
    bot._available_abilities_by_tag = {4: set()}
    assert unit_can_repair(scv, bot) is False
