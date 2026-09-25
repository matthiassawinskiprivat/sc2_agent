"""Command-dependent actor/target legality arrays for the squad pointer stack."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from sc2.ids.unit_typeid import UnitTypeId

from src.agent.squad_constants import MAX_SQUAD_ENEMIES, MAX_SQUAD_UNITS
from src.agent.squad_target_masks import (
    build_enemy_target_mask,
    build_squad_target_masks,
    zero_target_masks,
)


def _unit(
    tag,
    type_id,
    *,
    health=100.0,
    health_max=100.0,
    energy=200.0,
    ground_dps=10.0,
    is_biological=False,
    is_mechanical=False,
    is_structure=False,
    cargo_max=0,
    cargo_used=0,
    cargo_size=1,
    owner_id=1,
):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.health = health
    u.health_max = health_max
    u.energy = energy
    u.energy_max = 200.0
    u.ground_dps = ground_dps
    u.air_dps = 0.0
    u.is_biological = is_biological
    u.is_mechanical = is_mechanical
    u.is_structure = is_structure
    u.cargo_max = cargo_max
    u.cargo_used = cargo_used
    u.cargo_size = cargo_size
    u.owner_id = owner_id
    u.orders = []
    u.is_cloaked = False
    u.is_burrowed = False
    u.weapon_ready = True
    u.weapon_cooldown = 0.0
    return u


def _slots(*units):
    padded = list(units) + [None] * (MAX_SQUAD_UNITS - len(units))
    return padded


def test_zero_masks_have_the_documented_shapes():
    masks = zero_target_masks()
    assert set(masks) == {
        "heal_actor_mask",
        "heal_target_mask",
        "load_actor_mask",
        "load_target_mask",
        "attack_actor_mask",
        "aoe_actor_mask",
        "unload_actor_mask",
        "cloak_actor_mask",
        "uncloak_actor_mask",
        "move_actor_mask",
    }
    for arr in masks.values():
        assert arr.shape == (MAX_SQUAD_UNITS,)
        assert not arr.any()


def test_ready_queen_is_a_heal_actor_and_wounded_roach_a_patient():
    queen = _unit(1, UnitTypeId.QUEEN, is_biological=True)
    roach = _unit(2, UnitTypeId.ROACH, health=30.0, is_biological=True)
    masks = build_squad_target_masks(_slots(queen, roach))

    assert masks["heal_actor_mask"][0] == 1.0
    assert masks["heal_actor_mask"][1] == 0.0
    assert masks["heal_target_mask"][1] == 1.0
    # The Queen herself is at full health, so she is not a patient.
    assert masks["heal_target_mask"][0] == 0.0


def test_full_health_ally_is_not_a_heal_target():
    queen = _unit(1, UnitTypeId.QUEEN, is_biological=True)
    roach = _unit(2, UnitTypeId.ROACH, health=100.0, is_biological=True)
    masks = build_squad_target_masks(_slots(queen, roach))
    assert not masks["heal_target_mask"].any()


def test_biology_mismatch_leaves_no_heal_target():
    """An SCV cannot repair a Marine, so nothing in this squad is healable."""
    scv = _unit(1, UnitTypeId.SCV, is_mechanical=True)
    marine = _unit(2, UnitTypeId.MARINE, health=10.0, is_biological=True)
    masks = build_squad_target_masks(_slots(scv, marine))

    assert masks["heal_actor_mask"][0] == 1.0
    assert not masks["heal_target_mask"].any()


def test_mixed_squad_marks_the_patient_some_healer_can_affect():
    scv = _unit(1, UnitTypeId.SCV, is_mechanical=True)
    medivac = _unit(2, UnitTypeId.MEDIVAC, is_mechanical=True, cargo_max=8)
    marine = _unit(3, UnitTypeId.MARINE, health=10.0, is_biological=True)
    thor = _unit(4, UnitTypeId.THOR, health=100.0, health_max=400.0, is_mechanical=True)
    masks = build_squad_target_masks(_slots(scv, medivac, marine, thor))

    assert masks["heal_target_mask"][2] == 1.0  # Medivac heals the Marine
    assert masks["heal_target_mask"][3] == 1.0  # SCV repairs the Thor
    # The Medivac is mechanical and wounded-free here, so it is not a patient.
    assert masks["heal_target_mask"][1] == 0.0


def test_no_healers_means_no_heal_targets_even_with_wounded_allies():
    marine = _unit(1, UnitTypeId.MARINE, health=10.0, is_biological=True)
    marauder = _unit(2, UnitTypeId.MARAUDER, health=20.0, is_biological=True)
    masks = build_squad_target_masks(_slots(marine, marauder))
    assert not masks["heal_actor_mask"].any()
    assert not masks["heal_target_mask"].any()


def test_medivac_and_marine_are_load_actors_and_each_others_targets():
    medivac = _unit(1, UnitTypeId.MEDIVAC, cargo_max=8, cargo_used=0, cargo_size=0)
    marine = _unit(2, UnitTypeId.MARINE, cargo_size=1)
    masks = build_squad_target_masks(_slots(medivac, marine))

    assert masks["load_actor_mask"][0] == 1.0
    assert masks["load_actor_mask"][1] == 1.0
    assert masks["load_target_mask"][0] == 1.0  # passenger can board this transport
    assert masks["load_target_mask"][1] == 1.0  # transport can load this passenger


def test_full_medivac_is_not_a_load_actor_or_target():
    full = _unit(1, UnitTypeId.MEDIVAC, cargo_max=8, cargo_used=8, cargo_size=0)
    marine = _unit(2, UnitTypeId.MARINE, cargo_size=1)
    masks = build_squad_target_masks(_slots(full, marine))
    assert not masks["load_actor_mask"].any()
    assert not masks["load_target_mask"].any()


def test_passenger_too_large_for_free_slots_is_not_a_load_target():
    medivac = _unit(1, UnitTypeId.MEDIVAC, cargo_max=8, cargo_used=7, cargo_size=0)
    thor = _unit(2, UnitTypeId.THOR, cargo_size=4)
    masks = build_squad_target_masks(_slots(medivac, thor))
    # Transport has free slots and Thor can embark in principle, but the pair
    # does not fit — so neither side is legal for load_unit this step.
    assert not masks["load_actor_mask"].any()
    assert not masks["load_target_mask"].any()


def test_attack_actor_mask_requires_a_weapon():
    marine = _unit(1, UnitTypeId.MARINE, ground_dps=9.8)
    medivac = _unit(2, UnitTypeId.MEDIVAC, ground_dps=0.0, cargo_max=8)
    masks = build_squad_target_masks(_slots(marine, medivac))
    assert masks["attack_actor_mask"][0] == 1.0
    assert masks["attack_actor_mask"][1] == 0.0


def test_attack_actor_mask_includes_swarmhost_and_disruptor():
    # Weaponless casters whose attack remaps to an ability must stay selectable,
    # or Swarm Host-only squads permanently lose attack_local / attack_unit.
    host = _unit(1, UnitTypeId.SWARMHOSTMP, ground_dps=0.0)
    disruptor = _unit(2, UnitTypeId.DISRUPTOR, ground_dps=0.0)
    medivac = _unit(3, UnitTypeId.MEDIVAC, ground_dps=0.0, cargo_max=8)
    masks = build_squad_target_masks(_slots(host, disruptor, medivac))
    assert masks["attack_actor_mask"][0] == 1.0
    assert masks["attack_actor_mask"][1] == 1.0
    assert masks["attack_actor_mask"][2] == 0.0


def test_attack_actor_mask_excludes_burrowed_ling_keeps_lurker():
    ling = _unit(1, UnitTypeId.ZERGLINGBURROWED, ground_dps=10.0)
    ling.is_burrowed = True
    lurker = _unit(2, UnitTypeId.LURKERMPBURROWED, ground_dps=20.0)
    lurker.is_burrowed = True
    marine = _unit(3, UnitTypeId.MARINE, ground_dps=9.8)
    masks = build_squad_target_masks(_slots(ling, lurker, marine))
    assert masks["attack_actor_mask"][0] == 0.0
    assert masks["attack_actor_mask"][1] == 1.0
    assert masks["attack_actor_mask"][2] == 1.0


def test_attack_actor_mask_requires_weapon_ready():
    ready = _unit(1, UnitTypeId.MARINE, ground_dps=9.8)
    ready.weapon_ready = True
    ready.weapon_cooldown = 0.0
    cooling = _unit(2, UnitTypeId.MARINE, ground_dps=9.8)
    cooling.weapon_ready = False
    cooling.weapon_cooldown = 12.0
    masks = build_squad_target_masks(_slots(ready, cooling))
    assert masks["attack_actor_mask"][0] == 1.0
    assert masks["attack_actor_mask"][1] == 0.0


def test_tactical_enemy_mask_marks_only_enemies_in_fovea():
    from src.agent.squad_target_masks import build_tactical_enemy_mask
    from src.utils.unit_feature_helpers import TACTICAL_FOVEA_MAX_DELTA

    anchor = _unit(1, UnitTypeId.MARINE)
    anchor.position = MagicMock(x=50.0, y=50.0)
    near = _unit(2, UnitTypeId.ZERGLING)
    near.position = MagicMock(x=50.0 + TACTICAL_FOVEA_MAX_DELTA * 0.5, y=50.0)
    near.is_flying = False
    far = _unit(3, UnitTypeId.ZERGLING)
    far.position = MagicMock(x=50.0 + TACTICAL_FOVEA_MAX_DELTA + 1.0, y=50.0)
    far.is_flying = False
    air = _unit(4, UnitTypeId.MUTALISK)
    air.position = MagicMock(x=50.0, y=50.0)
    air.is_flying = True

    mask = build_tactical_enemy_mask([near, far, air], anchor)
    assert mask[0] == 1.0
    assert mask[1] == 0.0
    assert mask[2] == 1.0
    air_only = build_tactical_enemy_mask([near, far, air], anchor, air_only=True)
    assert air_only[0] == 0.0
    assert air_only[2] == 1.0
    assert not build_tactical_enemy_mask([near, far], None).any()


def test_engage_enemy_mask_marks_enemies_near_any_ally():
    """Command gating must open when *some* ally could become this step's anchor."""
    from src.agent.squad_target_masks import (
        build_engage_enemy_mask,
        build_tactical_enemy_mask,
    )
    from src.utils.unit_feature_helpers import TACTICAL_FOVEA_MAX_DELTA

    ling = _unit(1, UnitTypeId.ZERGLING)
    ling.position = MagicMock(x=10.0, y=10.0)
    infestor = _unit(2, UnitTypeId.INFESTOR)
    infestor.position = MagicMock(x=50.0, y=50.0)
    near_infestor = _unit(3, UnitTypeId.STALKER)
    near_infestor.position = MagicMock(
        x=50.0 + TACTICAL_FOVEA_MAX_DELTA * 0.5, y=50.0
    )
    near_infestor.is_flying = False
    far = _unit(4, UnitTypeId.STALKER)
    far.position = MagicMock(x=90.0, y=90.0)
    far.is_flying = False

    mask = build_engage_enemy_mask([near_infestor, far], [ling, infestor])
    assert mask[0] == 1.0
    assert mask[1] == 0.0
    # Hard-anchor = ling alone would miss the near enemy; engage still sees it.
    assert build_tactical_enemy_mask([near_infestor], ling)[0] == 0.0
    assert build_tactical_enemy_mask([near_infestor], infestor)[0] == 1.0

    air = _unit(5, UnitTypeId.MUTALISK)
    air.position = MagicMock(x=50.0, y=50.0)
    air.is_flying = True
    air_only = build_engage_enemy_mask(
        [near_infestor, air], [infestor], air_only=True
    )
    assert air_only[0] == 0.0
    assert air_only[1] == 1.0


def test_engage_enemy_mask_opens_for_nearby_building_without_mobiles():
    """attack_local must stay legal when the only nearby hostile is a structure."""
    from src.agent.squad_target_masks import (
        build_engage_enemy_mask,
        build_tactical_enemy_mask,
    )
    from src.utils.unit_feature_helpers import TACTICAL_FOVEA_MAX_DELTA

    marine = _unit(1, UnitTypeId.MARINE)
    marine.position = MagicMock(x=50.0, y=50.0)
    pylon = _unit(2, UnitTypeId.PYLON, is_structure=True)
    pylon.position = MagicMock(x=50.0 + TACTICAL_FOVEA_MAX_DELTA * 0.5, y=50.0)
    far_pylon = _unit(3, UnitTypeId.PYLON, is_structure=True)
    far_pylon.position = MagicMock(x=90.0, y=90.0)

    only_building = build_engage_enemy_mask(
        [None], [marine], extra_hostiles=[pylon]
    )
    assert only_building.any()
    far_only = build_engage_enemy_mask(
        [None], [marine], extra_hostiles=[far_pylon]
    )
    assert not far_only.any()
    # Viper air check must not treat a pylon as a flying hostile.
    air_only = build_engage_enemy_mask(
        [None], [marine], air_only=True, extra_hostiles=[pylon]
    )
    assert not air_only.any()
    assert build_tactical_enemy_mask(
        [None], marine, extra_hostiles=[pylon]
    ).any()
    assert not build_tactical_enemy_mask(
        [None], marine, extra_hostiles=[far_pylon]
    ).any()


def test_structures_are_never_actors():
    battery = _unit(
        1, UnitTypeId.SHIELDBATTERY, ground_dps=0.0, is_structure=True
    )
    masks = build_squad_target_masks(_slots(battery))
    assert not masks["attack_actor_mask"].any()
    assert not masks["heal_actor_mask"].any()
    assert not masks["load_actor_mask"].any()


def test_enemy_target_mask_marks_occupied_slots_only():
    ling = _unit(1, UnitTypeId.ZERGLING)
    roach = _unit(2, UnitTypeId.ROACH)
    mask = build_enemy_target_mask([ling, roach] + [None] * (MAX_SQUAD_ENEMIES - 2))
    assert mask.shape == (MAX_SQUAD_ENEMIES,)
    assert mask[0] == 1.0 and mask[1] == 1.0
    assert not mask[2:].any()


@pytest.mark.parametrize("count", [0, 1, MAX_SQUAD_ENEMIES + 5])
def test_enemy_target_mask_is_shape_stable(count):
    units = [_unit(i + 1, UnitTypeId.ZERGLING) for i in range(count)]
    mask = build_enemy_target_mask(units)
    assert mask.shape == (MAX_SQUAD_ENEMIES,)
    assert mask.sum() == min(count, MAX_SQUAD_ENEMIES)


def test_masks_are_slot_aligned_with_padding():
    queen = _unit(5, UnitTypeId.QUEEN, is_biological=True)
    wounded = _unit(9, UnitTypeId.ROACH, health=1.0, is_biological=True)
    slots = [None, queen, None, wounded] + [None] * (MAX_SQUAD_UNITS - 4)
    masks = build_squad_target_masks(slots)
    assert np.flatnonzero(masks["heal_actor_mask"]).tolist() == [1]
    assert np.flatnonzero(masks["heal_target_mask"]).tolist() == [3]


def test_aoe_actor_requires_energy_and_psi_storm_tech():
    from sc2.ids.upgrade_id import UpgradeId

    ht = _unit(1, UnitTypeId.HIGHTEMPLAR, energy=75.0)
    marine = _unit(2, UnitTypeId.MARINE)
    without = build_squad_target_masks(_slots(ht, marine), upgrades=frozenset())
    assert not without["aoe_actor_mask"].any()

    with_tech = build_squad_target_masks(
        _slots(ht, marine), upgrades=frozenset({UpgradeId.PSISTORMTECH})
    )
    assert with_tech["aoe_actor_mask"][0] == 1.0
    assert with_tech["aoe_actor_mask"][1] == 0.0


def test_viper_aoe_actor_requires_tactical_air_enemy():
    viper = _unit(1, UnitTypeId.VIPER, energy=125.0)
    without_air = build_squad_target_masks(
        _slots(viper), has_tactical_air_enemy=False
    )
    assert not without_air["aoe_actor_mask"].any()
    with_air = build_squad_target_masks(_slots(viper), has_tactical_air_enemy=True)
    assert with_air["aoe_actor_mask"][0] == 1.0


def test_move_actor_excludes_immobile_units():
    from sc2.ids.upgrade_id import UpgradeId

    marine = _unit(1, UnitTypeId.MARINE)
    sieged = _unit(2, UnitTypeId.SIEGETANKSIEGED)
    burrowed = _unit(3, UnitTypeId.ZERGLINGBURROWED)
    burrowed.is_burrowed = True
    masks = build_squad_target_masks(
        _slots(marine, sieged, burrowed), upgrades=frozenset({UpgradeId.BURROW})
    )
    assert masks["move_actor_mask"][0] == 1.0
    assert masks["move_actor_mask"][1] == 0.0
    assert masks["move_actor_mask"][2] == 0.0


def test_unload_actor_requires_loaded_transport():
    empty = _unit(1, UnitTypeId.MEDIVAC, cargo_max=8, cargo_used=0, cargo_size=0)
    loaded = _unit(2, UnitTypeId.MEDIVAC, cargo_max=8, cargo_used=2, cargo_size=0)
    masks = build_squad_target_masks(_slots(empty, loaded))
    assert masks["unload_actor_mask"][0] == 0.0
    assert masks["unload_actor_mask"][1] == 1.0


def test_cloak_and_uncloak_actor_masks_respect_state_and_upgrades():
    from sc2.ids.upgrade_id import UpgradeId

    ghost = _unit(1, UnitTypeId.GHOST, energy=50.0)
    ghost.is_cloaked = False
    cloaked = _unit(2, UnitTypeId.GHOST, energy=50.0)
    cloaked.is_cloaked = True
    ling = _unit(3, UnitTypeId.ZERGLING)
    ling.is_burrowed = False
    burrowed = _unit(4, UnitTypeId.ZERGLINGBURROWED)
    burrowed.is_burrowed = True

    no_tech = build_squad_target_masks(
        _slots(ghost, cloaked, ling, burrowed), upgrades=frozenset()
    )
    # Without PERSONALCLOAKING / BURROW only the already-burrowed morph can reveal.
    assert not no_tech["cloak_actor_mask"].any()
    assert no_tech["uncloak_actor_mask"][0] == 0.0
    assert no_tech["uncloak_actor_mask"][1] == 1.0  # cloaked ghost can uncloak
    assert no_tech["uncloak_actor_mask"][3] == 1.0  # burrowed ling can unburrow

    tech = frozenset({UpgradeId.PERSONALCLOAKING, UpgradeId.BURROW})
    with_tech = build_squad_target_masks(
        _slots(ghost, cloaked, ling, burrowed), upgrades=tech
    )
    assert with_tech["cloak_actor_mask"][0] == 1.0
    assert with_tech["cloak_actor_mask"][1] == 0.0  # already cloaked
    assert with_tech["cloak_actor_mask"][2] == 1.0  # surface ling with BURROW
    assert with_tech["cloak_actor_mask"][3] == 0.0  # already burrowed
