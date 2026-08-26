"""Tests for splash_data catalog helpers."""
from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId

from src.utils.splash_data import (
    SPLASH_BY_UNIT,
    FRIENDLY_FIRE_UNIT_TYPES,
    ONETIME_UNIT_TYPES,
    TEMPORARY_LIFETIME_UNIT_TYPES,
    splash_spec,
    unit_has_friendly_fire,
    unit_has_splash,
    unit_is_onetime,
    unit_is_temporary,
)


def _unit(type_id, *, is_hallucination=False):
    u = MagicMock()
    u.type_id = type_id
    u.is_hallucination = is_hallucination
    return u


def test_catalog_includes_core_splash_units():
    for name in (
        "BANELING",
        "DISRUPTORPHASED",
        "SIEGETANKSIEGED",
        "ARCHON",
        "HELLION",
        "HELLIONTANK",
        "WIDOWMINEBURROWED",
        "COLOSSUS",
        "ULTRALISK",
        "LURKERMPBURROWED",
        "LURKERBURROWED",
        "LIBERATOR",
        "THOR",
    ):
        assert name in SPLASH_BY_UNIT
    # Morphs without splash weapons must not paint splash / HAS_SPLASH.
    assert "LURKERMP" not in SPLASH_BY_UNIT
    assert "WIDOWMINE" not in SPLASH_BY_UNIT
    assert "THORAP" not in SPLASH_BY_UNIT
    assert "LIBERATORAG" not in SPLASH_BY_UNIT
    assert unit_has_splash(_unit(UnitTypeId.LURKERMP)) is False
    assert unit_has_splash(_unit(UnitTypeId.LURKERMPBURROWED)) is True
    assert unit_has_splash(_unit(UnitTypeId.WIDOWMINE)) is False
    assert unit_has_splash(_unit(UnitTypeId.WIDOWMINEBURROWED)) is True
    assert unit_has_splash(_unit(UnitTypeId.LIBERATOR)) is True
    assert unit_has_splash(_unit(UnitTypeId.LIBERATORAG)) is False
    assert unit_has_splash(_unit(UnitTypeId.THOR)) is True
    assert unit_has_splash(_unit(UnitTypeId.THORAP)) is False


def test_baneling_and_nova_are_onetime():
    assert ONETIME_UNIT_TYPES == frozenset({"BANELING", "DISRUPTORPHASED"})
    assert unit_is_onetime(_unit(UnitTypeId.BANELING)) is True
    assert unit_is_onetime(_unit(UnitTypeId.DISRUPTORPHASED)) is True
    assert unit_is_onetime(_unit(UnitTypeId.BROODLING)) is False
    assert unit_is_onetime(_unit(UnitTypeId.LOCUSTMP)) is False


def test_disruptor_caster_is_not_splash_or_onetime():
    disruptor = _unit(UnitTypeId.DISRUPTOR)
    assert unit_has_splash(disruptor) is False
    assert unit_is_onetime(disruptor) is False
    assert splash_spec(disruptor) is None


def test_sieged_tank_has_min_range_donut():
    radius, min_range, damage = splash_spec(_unit(UnitTypeId.SIEGETANKSIEGED))
    assert radius == 13.0
    assert min_range == 2.0
    assert damage == 40.0


def test_liberator_fighter_splash_sieged_ag_is_not_splash():
    from src.utils.splash_data import (
        LIBERATOR_FIGHTER_SPLASH_DAMAGE,
        LIBERATOR_FIGHTER_SPLASH_RADIUS_GU,
    )

    r_f, min_f, dmg_f = splash_spec(_unit(UnitTypeId.LIBERATOR))
    assert r_f == LIBERATOR_FIGHTER_SPLASH_RADIUS_GU
    assert min_f == 0.0
    assert dmg_f == LIBERATOR_FIGHTER_SPLASH_DAMAGE
    assert splash_spec(_unit(UnitTypeId.LIBERATORAG)) is None
    assert unit_has_friendly_fire(_unit(UnitTypeId.LIBERATORAG)) is False


def test_thor_explosive_payload_splash_not_high_impact():
    from src.utils.splash_data import (
        THOR_JAVELIN_SPLASH_DAMAGE,
        THOR_JAVELIN_SPLASH_RADIUS_GU,
    )

    radius, min_range, damage = splash_spec(_unit(UnitTypeId.THOR))
    assert radius == THOR_JAVELIN_SPLASH_RADIUS_GU
    assert min_range == 0.0
    assert damage == THOR_JAVELIN_SPLASH_DAMAGE
    assert splash_spec(_unit(UnitTypeId.THORAP)) is None


def test_marine_has_no_splash():
    marine = _unit(UnitTypeId.MARINE)
    assert unit_has_splash(marine) is False
    assert unit_is_onetime(marine) is False
    assert splash_spec(marine) is None


def test_friendly_fire_catalog():
    assert FRIENDLY_FIRE_UNIT_TYPES == frozenset(
        {
            "SIEGETANKSIEGED",
            "WIDOWMINEBURROWED",
            "DISRUPTORPHASED",
            "RAVAGER",
            "HIGHTEMPLAR",
        }
    )
    assert unit_has_friendly_fire(_unit(UnitTypeId.SIEGETANKSIEGED)) is True
    assert unit_has_friendly_fire(_unit(UnitTypeId.WIDOWMINEBURROWED)) is True
    assert unit_has_friendly_fire(_unit(UnitTypeId.DISRUPTORPHASED)) is True
    assert unit_has_friendly_fire(_unit(UnitTypeId.RAVAGER)) is True
    assert unit_has_friendly_fire(_unit(UnitTypeId.HIGHTEMPLAR)) is True
    assert unit_has_friendly_fire(_unit(UnitTypeId.BANELING)) is False
    assert unit_has_friendly_fire(_unit(UnitTypeId.ARCHON)) is False
    assert unit_has_friendly_fire(_unit(UnitTypeId.DISRUPTOR)) is False
    assert unit_has_friendly_fire(_unit(UnitTypeId.MARINE)) is False


def test_temporary_lifetime_catalog():
    for name in (
        "DISRUPTORPHASED",
        "BROODLING",
        "LOCUSTMP",
        "LOCUSTMPFLYING",
        "ADEPTPHASESHIFT",
        "AUTOTURRET",
        "POINTDEFENSEDRONE",
        "INFESTORTERRAN",
        "CHANGELING",
        "MULE",
    ):
        assert name in TEMPORARY_LIFETIME_UNIT_TYPES

    assert unit_is_temporary(_unit(UnitTypeId.BROODLING)) is True
    assert unit_is_temporary(_unit(UnitTypeId.LOCUSTMP)) is True
    assert unit_is_temporary(_unit(UnitTypeId.ADEPTPHASESHIFT)) is True
    assert unit_is_temporary(_unit(UnitTypeId.AUTOTURRET)) is True
    assert unit_is_temporary(_unit(UnitTypeId.DISRUPTORPHASED)) is True
    assert unit_is_temporary(_unit(UnitTypeId.BANELING)) is False
    assert unit_is_temporary(_unit(UnitTypeId.MARINE)) is False
    assert unit_is_temporary(_unit(UnitTypeId.INTERCEPTOR)) is False
    # Hallucinations expire even when the base type is permanent.
    assert unit_is_temporary(_unit(UnitTypeId.ZEALOT, is_hallucination=True)) is True
