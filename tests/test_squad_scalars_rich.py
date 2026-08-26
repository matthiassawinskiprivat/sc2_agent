"""Tests for SquadObservationBuilder rich scalar computation."""
import math
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest
from sc2.ids.unit_typeid import UnitTypeId

# Older/partial burnysc2 installs may omit detector constants used by ObservationTransformer.
import sc2.constants as _sc2_constants

if not hasattr(_sc2_constants, "IS_DETECTOR"):
    _sc2_constants.IS_DETECTOR = frozenset(
        {
            UnitTypeId.OBSERVER,
            UnitTypeId.OBSERVERSIEGEMODE,
            UnitTypeId.OVERSEER,
            UnitTypeId.OVERSEERSIEGEMODE,
            UnitTypeId.RAVEN,
            UnitTypeId.MISSILETURRET,
            UnitTypeId.SPORECRAWLER,
        }
    )
if not hasattr(_sc2_constants, "UNIT_PHOTONCANNON"):
    _sc2_constants.UNIT_PHOTONCANNON = UnitTypeId.PHOTONCANNON

from src.agent.squad_constants import MAX_SQUAD_UNITS, SQUAD_UNIT_FEAT_DIM
from src.agent.unit_scalar_constants import UnitScalarConstants as USC
from src.transformers.squad_observation_builder import SquadObservationBuilder


@pytest.fixture
def training_conf():
    return {
        "active_layers": ["allied_unit_biological", "enemy_unit_biological"],
        "active_tactical_layers": ["allied_unit_biological", "enemy_unit_biological"],
        "agent_architecture": "squad",
    }


def _make_unit(
    tag=10,
    x=64.0, y=32.0,
    health=100.0, health_max=100.0,
    shield=0.0, shield_max=0.0,
    weapon_cooldown=0.0,
    weapon_ready=True,
    energy=0.0, energy_max=0.0,
    ground_range=5.0, air_range=0.0,
    ground_dps=10.0, air_dps=0.0,
    is_biological=True, is_mechanical=False,
    is_armored=False, is_massive=False,
    is_flying=False,
    is_attacking=False, is_moving=False,
    is_burrowed=False, is_cloaked=False, is_detected=False,
    type_id=None,
):
    unit = MagicMock()
    unit.tag = tag
    unit.position.x = x
    unit.position.y = y
    unit.health = health
    unit.health_max = health_max
    unit.shield = shield
    unit.shield_max = shield_max
    unit.weapon_cooldown = weapon_cooldown
    unit.weapon_ready = weapon_ready
    unit.energy = energy
    unit.energy_max = energy_max
    unit.ground_range = ground_range
    unit.air_range = air_range
    unit.ground_dps = ground_dps
    unit.air_dps = air_dps
    unit.is_biological = is_biological
    unit.is_mechanical = is_mechanical
    unit.is_armored = is_armored
    unit.is_massive = is_massive
    unit.is_flying = is_flying
    unit.is_attacking = is_attacking
    unit.is_moving = is_moving
    unit.is_burrowed = is_burrowed
    unit.is_cloaked = is_cloaked
    unit.is_detected = is_detected
    unit.is_hallucination = False
    unit.calculate_speed = MagicMock(return_value=3.15)
    unit.distance_to_weapon_ready = 0.0
    unit.type_id = type_id if type_id is not None else MagicMock()
    return unit


def test_scalar_vector_length(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit()
    feats = builder._unit_scalars(
        unit, "attack",
        playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(),
        objective=None,
        enemy_com=None,
        allied_com=None,
    )
    assert feats.shape == (SQUAD_UNIT_FEAT_DIM,)


def test_health_pct_full_health(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(health=100.0, health_max=100.0)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.HEALTH_PCT]] == pytest.approx(1.0)


def test_health_pct_half(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(health=50.0, health_max=100.0)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.HEALTH_PCT]] == pytest.approx(0.5)


def test_shield_zero_for_non_protoss(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(shield=0.0, shield_max=0.0)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.SHIELD_PCT]] == pytest.approx(0.0)


def test_weapon_ready_flag(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(weapon_ready=True, weapon_cooldown=0.0)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.WEAPON_READY]] == pytest.approx(1.0)

    unit2 = _make_unit(weapon_ready=False, weapon_cooldown=12.0)
    feats2 = builder._unit_scalars(
        unit2, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats2[USC.FEATURES[USC.WEAPON_READY]] == pytest.approx(0.0)


def test_role_flags(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit()
    feats_a = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats_a[USC.FEATURES[USC.ROLE_ATTACK]] == pytest.approx(1.0)
    assert feats_a[USC.FEATURES[USC.ROLE_DEFEND]] == pytest.approx(0.0)

    feats_d = builder._unit_scalars(
        unit, "defend", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats_d[USC.FEATURES[USC.ROLE_ATTACK]] == pytest.approx(0.0)
    assert feats_d[USC.FEATURES[USC.ROLE_DEFEND]] == pytest.approx(1.0)


def test_is_melee_for_melee_unit(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(ground_range=0.1, air_range=0.0)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.IS_MELEE]] == pytest.approx(1.0)
    assert feats[USC.FEATURES[USC.CAN_ATTACK_AIR]] == pytest.approx(0.0)


def test_is_melee_for_archon_despite_range(training_conf):
    """Archon range is 3.0 but CTF treats it as melee (force-melee override)."""
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(ground_range=3.0, air_range=3.0, type_id=UnitTypeId.ARCHON)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.IS_MELEE]] == pytest.approx(1.0)


def test_can_attack_air(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(ground_range=5.0, air_range=5.0, air_dps=8.0)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.CAN_ATTACK_AIR]] == pytest.approx(1.0)


def test_air_dps_norm(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(air_dps=20.0)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.AIR_DPS_NORM]] == pytest.approx(float(np.tanh(1.0)))


def test_has_splash_and_onetime_for_baneling(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(type_id=UnitTypeId.BANELING)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.HAS_SPLASH]] == pytest.approx(1.0)
    assert feats[USC.FEATURES[USC.ONETIME_UNIT]] == pytest.approx(1.0)


def test_disruptor_caster_not_splash_or_onetime(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(type_id=UnitTypeId.DISRUPTOR)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.HAS_SPLASH]] == pytest.approx(0.0)
    assert feats[USC.FEATURES[USC.ONETIME_UNIT]] == pytest.approx(0.0)


def test_archon_has_splash_not_onetime(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(type_id=UnitTypeId.ARCHON)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.HAS_SPLASH]] == pytest.approx(1.0)
    assert feats[USC.FEATURES[USC.ONETIME_UNIT]] == pytest.approx(0.0)
    assert feats[USC.FEATURES[USC.IS_FRIENDLY_FIRE]] == pytest.approx(0.0)


def test_broodling_and_locust_are_temporary_not_onetime(training_conf):
    builder = SquadObservationBuilder(training_conf)
    for type_id in (UnitTypeId.BROODLING, UnitTypeId.LOCUSTMP, UnitTypeId.LOCUSTMPFLYING):
        feats = builder._unit_scalars(
            _make_unit(type_id=type_id),
            "attack",
            playable_w=128.0,
            playable_h=128.0,
            upgrades=frozenset(),
            objective=None,
            enemy_com=None,
            allied_com=None,
        )
        assert feats[USC.FEATURES[USC.ONETIME_UNIT]] == pytest.approx(0.0)
        assert feats[USC.FEATURES[USC.IS_TEMPORARY]] == pytest.approx(1.0)


def test_nova_is_onetime_and_temporary(training_conf):
    builder = SquadObservationBuilder(training_conf)
    feats = builder._unit_scalars(
        _make_unit(type_id=UnitTypeId.DISRUPTORPHASED),
        "attack",
        playable_w=128.0,
        playable_h=128.0,
        upgrades=frozenset(),
        objective=None,
        enemy_com=None,
        allied_com=None,
    )
    assert feats[USC.FEATURES[USC.ONETIME_UNIT]] == pytest.approx(1.0)
    assert feats[USC.FEATURES[USC.IS_TEMPORARY]] == pytest.approx(1.0)


def test_hallucination_is_temporary(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(type_id=UnitTypeId.ZEALOT)
    unit.is_hallucination = True
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.IS_TEMPORARY]] == pytest.approx(1.0)


def test_friendly_fire_for_sieged_tank_and_ravager(training_conf):
    builder = SquadObservationBuilder(training_conf)
    tank = _make_unit(type_id=UnitTypeId.SIEGETANKSIEGED)
    ravager = _make_unit(type_id=UnitTypeId.RAVAGER)
    templar = _make_unit(type_id=UnitTypeId.HIGHTEMPLAR)
    for unit in (tank, ravager, templar):
        feats = builder._unit_scalars(
            unit, "attack", playable_w=128.0, playable_h=128.0,
            upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
        )
        assert feats[USC.FEATURES[USC.IS_FRIENDLY_FIRE]] == pytest.approx(1.0)

    baneling = _make_unit(type_id=UnitTypeId.BANELING)
    feats = builder._unit_scalars(
        baneling, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.IS_FRIENDLY_FIRE]] == pytest.approx(0.0)


def test_concealed_when_burrowed(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(is_burrowed=True)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.IS_CONCEALED]] == pytest.approx(1.0)


def test_concealed_when_cloaked(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(is_cloaked=True)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.IS_CONCEALED]] == pytest.approx(1.0)


def test_dist_to_objective(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(x=0.0, y=0.0)
    obj = MagicMock()
    obj.x = 64.0
    obj.y = 0.0
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=obj, enemy_com=None, allied_com=None,
    )
    expected = 64.0 / 128.0
    assert feats[USC.FEATURES[USC.DIST_TO_OBJECTIVE_NORM]] == pytest.approx(expected)


def test_dist_to_enemy_com(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(x=0.0, y=0.0)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None,
        enemy_com=(96.0, 0.0), allied_com=None,
    )
    expected = 96.0 / 128.0
    assert feats[USC.FEATURES[USC.DIST_TO_ENEMY_COM_NORM]] == pytest.approx(expected)


def test_max_health_norm_distinguishes_tankiness(training_conf):
    """Zergling-scale HP should score lower than Battlecruiser-scale HP."""
    builder = SquadObservationBuilder(training_conf)
    zergling = _make_unit(health=35.0, health_max=35.0, shield=0.0, shield_max=0.0)
    bc = _make_unit(health=550.0, health_max=550.0, shield=0.0, shield_max=0.0)
    feats_z = builder._unit_scalars(
        zergling, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    feats_bc = builder._unit_scalars(
        bc, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    idx = USC.FEATURES[USC.MAX_HEALTH_NORM]
    assert feats_z[idx] < feats_bc[idx]
    assert feats_z[idx] == pytest.approx(math.tanh(35.0 / 400.0))
    assert feats_bc[idx] == pytest.approx(math.tanh(550.0 / 400.0))


def test_is_detected_flag(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit(is_cloaked=True, is_detected=True)
    feats = builder._unit_scalars(
        unit, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats[USC.FEATURES[USC.IS_DETECTED]] == pytest.approx(1.0)
    assert feats[USC.FEATURES[USC.IS_CONCEALED]] == pytest.approx(1.0)


def test_static_feat_cache_used(training_conf):
    """_unit_static_feats is called once per type_id (cache hit on second call)."""
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit()
    _ = builder._unit_static_feats(unit, frozenset())
    assert len(builder._static_feat_cache) == 1
    _ = builder._unit_static_feats(unit, frozenset())  # cache hit
    assert len(builder._static_feat_cache) == 1


def test_clear_global_cache_clears_static_cache(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit()
    _ = builder._unit_static_feats(unit, frozenset())
    assert len(builder._static_feat_cache) == 1
    builder.clear_global_cache()
    assert len(builder._static_feat_cache) == 0


def test_active_unit_scalars_subset_config_order(training_conf):
    """Obs vector follows active_unit_scalars config order, not catalog index order."""
    training_conf = {
        **training_conf,
        "active_unit_scalars": [
            USC.ROLE_DEFEND,
            USC.HEALTH_PCT,
            USC.X_NORM,
        ],
    }
    builder = SquadObservationBuilder(training_conf)
    assert builder.active_unit_scalars == [
        USC.ROLE_DEFEND, USC.HEALTH_PCT, USC.X_NORM
    ]
    assert builder.unit_feat_dim == 3

    unit = _make_unit(health=50.0, health_max=100.0, x=64.0, y=0.0)
    feats = builder._unit_scalars(
        unit, "defend", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
    )
    assert feats.shape == (3,)
    assert feats[0] == pytest.approx(1.0)   # role_defend
    assert feats[1] == pytest.approx(0.5)   # health_pct
    assert feats[2] == pytest.approx(0.5)   # x_norm


def test_zero_observation_respects_active_subset(training_conf):
    training_conf = {
        **training_conf,
        "active_unit_scalars": [USC.HEALTH_PCT, USC.WEAPON_READY],
    }
    builder = SquadObservationBuilder(training_conf)
    obs = builder.zero_observation()
    assert obs["unit_scalars"].shape == (MAX_SQUAD_UNITS, 2)


def test_can_aoe_ghost_only_vs_protoss(training_conf):
    builder = SquadObservationBuilder(training_conf)
    ghost = _make_unit(type_id=UnitTypeId.GHOST, energy=200.0)
    feats_p = builder._unit_scalars(
        ghost, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
        enemy_race="protoss",
    )
    feats_z = builder._unit_scalars(
        ghost, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
        enemy_race="zerg",
    )
    assert feats_p[USC.FEATURES[USC.CAN_AOE]] == pytest.approx(1.0)
    assert feats_z[USC.FEATURES[USC.CAN_AOE]] == pytest.approx(0.0)

    templar = _make_unit(type_id=UnitTypeId.HIGHTEMPLAR, energy=200.0)
    feats_ht = builder._unit_scalars(
        templar, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
        enemy_race="terran",
    )
    assert feats_ht[USC.FEATURES[USC.CAN_AOE]] == pytest.approx(1.0)
