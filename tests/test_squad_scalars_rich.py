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


def test_neuraled_scalar_marks_stolen_body_only(training_conf):
    """Control buff is 1 on ally and enemy rows; caster and windup buffs stay 0."""
    from sc2.ids.buff_id import BuffId

    builder = SquadObservationBuilder(training_conf)
    stolen = _make_unit(type_id=UnitTypeId.BATTLECRUISER)
    stolen.buffs = {BuffId.NEURALPARASITE}
    own = _make_unit(type_id=UnitTypeId.INFESTOR)
    own.buffs = {BuffId.NEURALPARASITECHILDREN}
    windup = _make_unit(type_id=UnitTypeId.BATTLECRUISER)
    windup.buffs = {BuffId.NEURALPARASITEWAIT}
    plain = _make_unit(type_id=UnitTypeId.ROACH)

    kwargs = dict(
        playable_w=128.0, playable_h=128.0, upgrades=frozenset(),
        objective=None, enemy_com=None, allied_com=None,
    )
    stolen_ally = builder._unit_scalars(stolen, "attack", **kwargs)
    stolen_enemy = builder._unit_scalars(stolen, "attack", for_enemy=True, **kwargs)
    own_feats = builder._unit_scalars(own, "attack", **kwargs)
    windup_feats = builder._unit_scalars(windup, "attack", **kwargs)
    plain_feats = builder._unit_scalars(plain, "attack", **kwargs)

    idx = USC.FEATURES[USC.IS_NEURALED]
    assert stolen_ally[idx] == pytest.approx(1.0)
    assert stolen_enemy[idx] == pytest.approx(1.0)
    assert own_feats[idx] == pytest.approx(0.0)
    assert windup_feats[idx] == pytest.approx(0.0)
    assert plain_feats[idx] == pytest.approx(0.0)

    half = _make_unit(type_id=UnitTypeId.BATTLECRUISER)
    half.buffs = {BuffId.NEURALPARASITE}
    half.buff_duration_remain = 50
    half.buff_duration_max = 200
    half_feats = builder._unit_scalars(half, "attack", for_enemy=True, **kwargs)
    assert half_feats[idx] == pytest.approx(0.25)

    stim = _make_unit(type_id=UnitTypeId.MARINE)
    stim.buffs = []
    stim.buff_duration_remain = 10
    stim.buff_duration_max = 10
    stim_feats = builder._unit_scalars(stim, "attack", **kwargs)
    assert stim_feats[idx] == pytest.approx(0.0)


def test_both_perspectives_list_a_neuraled_unit_as_an_attack_target(training_conf):
    from sc2.ids.buff_id import BuffId

    infestor = _make_unit(tag=1, type_id=UnitTypeId.INFESTOR)
    infestor.owner_id = 1
    infestor.is_structure = False
    infestor.buffs = []
    stolen = _make_unit(tag=2, type_id=UnitTypeId.COLOSSUS)
    stolen.owner_id = 1
    stolen.is_structure = False
    stolen.buffs = {BuffId.NEURALPARASITE}
    bot = MagicMock()
    bot.all_units = [infestor, stolen]
    bot.units = [infestor, stolen]
    bot.enemy_units = []
    bot.structures = []
    bot.enemy_structures = []

    zerg_targets = SquadObservationBuilder.resolve_enemy_target_tags(bot, 1)
    other_targets = SquadObservationBuilder.resolve_enemy_target_tags(bot, 2)
    assert zerg_targets == [2]
    assert 2 in other_targets
    assert 1 in other_targets


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


def test_clear_global_cache_keeps_static_cache(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit()
    _ = builder._unit_static_feats(unit, frozenset())
    assert len(builder._static_feat_cache) == 1
    builder.clear_global_cache()
    assert len(builder._static_feat_cache) == 1


def test_clear_static_feat_cache(training_conf):
    builder = SquadObservationBuilder(training_conf)
    unit = _make_unit()
    _ = builder._unit_static_feats(unit, frozenset())
    builder.clear_static_feat_cache()
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


def test_can_aoe_ghost_vs_protoss_and_zerg(training_conf):
    training_conf = {**training_conf, "action_heads": ["aoe_local"]}
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
    feats_t = builder._unit_scalars(
        ghost, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
        enemy_race="terran",
    )
    assert feats_p[USC.FEATURES[USC.CAN_AOE]] == pytest.approx(1.0)
    assert feats_z[USC.FEATURES[USC.CAN_AOE]] == pytest.approx(1.0)
    assert feats_t[USC.FEATURES[USC.CAN_AOE]] == pytest.approx(0.0)

    from sc2.ids.upgrade_id import UpgradeId

    templar = _make_unit(type_id=UnitTypeId.HIGHTEMPLAR, energy=200.0)
    feats_ht = builder._unit_scalars(
        templar, "attack", playable_w=128.0, playable_h=128.0,
        upgrades=frozenset({UpgradeId.PSISTORMTECH}),
        objective=None, enemy_com=None, allied_com=None,
        enemy_race="terran",
    )
    assert feats_ht[USC.FEATURES[USC.CAN_AOE]] == pytest.approx(1.0)


def test_readiness_scalars_gated_by_action_heads(training_conf):
    """Without heal/aoe/snipe heads, readiness lookups stay zero."""
    from src.utils.unit_feature_helpers import unit_can_heal

    ghost = _make_unit(type_id=UnitTypeId.GHOST, energy=200.0)
    builder = SquadObservationBuilder(training_conf)
    with patch(
        "src.transformers.squad_observation_builder.unit_can_heal",
        wraps=unit_can_heal,
    ) as heal:
        feats = builder._unit_scalars(
            ghost, "attack", playable_w=128.0, playable_h=128.0,
            upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
            enemy_race="protoss",
        )
    heal.assert_not_called()
    assert feats[USC.FEATURES[USC.CAN_AOE]] == pytest.approx(0.0)
    assert feats[USC.FEATURES[USC.CAN_SNIPE]] == pytest.approx(0.0)
    assert feats[USC.FEATURES[USC.CAN_HEAL]] == pytest.approx(0.0)


def test_enemy_rows_skip_readiness_lookups(training_conf):
    training_conf = {
        **training_conf,
        "action_heads": ["heal_unit", "aoe_local", "snipe"],
    }
    builder = SquadObservationBuilder(training_conf)
    ghost = _make_unit(type_id=UnitTypeId.GHOST, energy=200.0)
    with patch(
        "src.transformers.squad_observation_builder.unit_can_snipe",
        return_value=True,
    ) as snipe:
        ally = builder._unit_scalars(
            ghost, "attack", playable_w=128.0, playable_h=128.0,
            upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
        )
        enemy = builder._unit_scalars(
            ghost, "attack", playable_w=128.0, playable_h=128.0,
            upgrades=frozenset(), objective=None, enemy_com=None, allied_com=None,
            for_enemy=True,
        )
    assert snipe.call_count == 1
    assert ally[USC.FEATURES[USC.CAN_SNIPE]] == pytest.approx(1.0)
    assert enemy[USC.FEATURES[USC.CAN_SNIPE]] == pytest.approx(0.0)


def test_unit_scalars_accepts_precomputed_readiness(training_conf):
    training_conf = {
        **training_conf,
        "action_heads": ["heal_unit", "aoe_local", "snipe"],
    }
    builder = SquadObservationBuilder(training_conf)
    ghost = _make_unit(type_id=UnitTypeId.GHOST, energy=200.0)
    with patch(
        "src.transformers.squad_observation_builder.unit_can_snipe",
        return_value=True,
    ) as snipe:
        feats = builder._unit_scalars(
            ghost,
            "attack",
            playable_w=128.0,
            playable_h=128.0,
            upgrades=frozenset(),
            objective=None,
            enemy_com=None,
            allied_com=None,
            readiness={
                "can_heal": 0.0,
                "can_repair": 0.0,
                "can_aoe": 1.0,
                "can_snipe": 1.0,
                "can_load": 0.0,
                "can_unload": 0.0,
                "can_cloak": 0.0,
                "is_immobile": 0.0,
            },
        )
    snipe.assert_not_called()
    assert feats[USC.FEATURES[USC.CAN_SNIPE]] == pytest.approx(1.0)
    assert feats[USC.FEATURES[USC.CAN_AOE]] == pytest.approx(1.0)


def test_build_reuses_legality_readiness_without_second_unit_can_pass(training_conf):
    """Ally scalars must not re-call unit_can_* after build_squad_legality."""
    training_conf = {
        **training_conf,
        "action_heads": ["heal_unit", "aoe_local", "snipe", "move_local"],
        "active_unit_scalars": [
            USC.CAN_HEAL,
            USC.CAN_AOE,
            USC.CAN_SNIPE,
            USC.CAN_LOAD,
            USC.CAN_CLOAK,
            USC.IS_IMMOBILE,
        ],
    }
    builder = SquadObservationBuilder(training_conf)
    ghost = _make_unit(tag=10, type_id=UnitTypeId.GHOST, energy=200.0)
    ghost.is_structure = False
    bot = MagicMock()
    bot.player_id = 1
    bot.units = MagicMock()
    bot.units.exists = True
    bot.units.first = ghost
    bot.enemy_units = MagicMock()
    bot.structures = MagicMock()
    bot.enemy_structures = MagicMock()
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot.state.upgrades = frozenset()
    bot.game_info = MagicMock()
    bot.game_data = MagicMock()
    bot.unit_target_map = {}
    bot.unit_command_map = {10: "attack"}
    bot.enemy_pylon_pos = {}
    bot.all_units.find_by_tag = lambda tag: ghost if tag == 10 else None

    fake_map = {
        "map": np.zeros((2, 128, 128), dtype=np.float32),
        "command": np.zeros(10, dtype=np.float32),
    }
    fake_tactical = np.zeros((2, 32, 32), dtype=np.float32)

    with patch(
        "src.agent.squad_target_masks.unit_can_snipe", return_value=True
    ) as snipe_mask, patch(
        "src.transformers.squad_observation_builder.unit_can_snipe", return_value=True
    ) as snipe_scalar, patch(
        "src.transformers.squad_observation_builder.perspective_unit_groups",
        return_value=([ghost], [], [], []),
    ), patch.object(
        builder._transformer, "transform", return_value=fake_map
    ), patch.object(
        builder._transformer, "transform_tactical", return_value=fake_tactical
    ):
        obs = builder.build(
            bot,
            perspective_player_id=1,
            squad_unit_tags=[10],
            pool_high_level_command="attack",
            playable_size=(128.0, 128.0),
            anchor_unit_tag=10,
            enemy_race="protoss",
        )

    assert snipe_mask.call_count == 1
    snipe_scalar.assert_not_called()
    assert obs["snipe_actor_mask"][0] == pytest.approx(1.0)
    # active list: CAN_HEAL, CAN_AOE, CAN_SNIPE, ...
    assert obs["unit_scalars"][0, 2] == pytest.approx(1.0)
