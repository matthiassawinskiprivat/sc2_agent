"""Tests for UnitScalarConstants index mapping."""
import pytest
from src.agent.unit_scalar_constants import UnitScalarConstants as USC


def test_get_dim_equals_max_index_plus_one():
    max_idx = max(USC.FEATURES.values())
    assert USC.get_dim() == max_idx + 1


def test_indices_are_unique():
    indices = list(USC.FEATURES.values())
    assert len(indices) == len(set(indices)), "Duplicate indices in FEATURES"


def test_indices_form_contiguous_range():
    indices = sorted(USC.FEATURES.values())
    assert indices == list(range(len(indices))), "FEATURES indices must be 0-based and contiguous"


def test_all_string_constants_present_in_features():
    """Every class-level string constant must have an entry in FEATURES."""
    string_constants = {
        v for k, v in vars(USC).items()
        if not k.startswith("_") and isinstance(v, str) and k != "__doc__"
    }
    for name in string_constants:
        assert name in USC.FEATURES, f"Constant {name!r} not in FEATURES"


def test_dim_matches_squad_unit_feat_dim():
    from src.agent.squad_constants import SQUAD_UNIT_FEAT_DIM
    assert SQUAD_UNIT_FEAT_DIM == USC.get_dim()


def test_expected_feature_names_present():
    required = [
        USC.HEALTH_PCT, USC.SHIELD_PCT, USC.COMBINED_VITALITY,
        USC.WEAPON_COOLDOWN_NORM, USC.WEAPON_READY,
        USC.MOVEMENT_SPEED_NORM, USC.DIST_TO_WEAPON_READY_NORM, USC.ENERGY_PCT,
        USC.GROUND_RANGE_NORM, USC.AIR_RANGE_NORM, USC.GROUND_DPS_NORM,
        USC.CAN_ATTACK_AIR, USC.IS_MELEE, USC.MAX_HEALTH_NORM,
        USC.IS_BIOLOGICAL, USC.IS_MECHANICAL, USC.IS_ARMORED,
        USC.IS_MASSIVE, USC.IS_FLYING, USC.HAS_SHIELD,
        USC.X_NORM, USC.Y_NORM,
        USC.DIST_TO_OBJECTIVE_NORM, USC.DIST_TO_ENEMY_COM_NORM, USC.DIST_TO_ALLIED_COM_NORM,
        USC.IS_ATTACKING, USC.IS_MOVING, USC.IS_CONCEALED, USC.IS_DETECTED,
        USC.ROLE_ATTACK, USC.ROLE_DEFEND,
        USC.AIR_DPS_NORM, USC.HAS_SPLASH, USC.ONETIME_UNIT, USC.IS_FRIENDLY_FIRE,
        USC.IS_TEMPORARY, USC.CAN_AOE,
    ]
    for name in required:
        assert name in USC.FEATURES


def test_append_new_feature_does_not_shift_existing():
    """Verify that the existing indices are stable — simulates append-only contract."""
    snapshot = dict(USC.FEATURES)
    for name, idx in snapshot.items():
        assert USC.FEATURES[name] == idx, f"Index for {name!r} must never change"


def test_default_active_features_matches_catalog_order():
    names = USC.default_active_features()
    assert names == [n for n, _ in sorted(USC.FEATURES.items(), key=lambda kv: kv[1])]
    assert len(names) == USC.get_dim()


def test_resolve_active_features_empty_falls_back_to_catalog():
    assert USC.resolve_active_features(None) == USC.default_active_features()
    assert USC.resolve_active_features([]) == USC.default_active_features()


def test_resolve_active_features_subset_preserves_config_order():
    resolved = USC.resolve_active_features(
        [USC.ROLE_DEFEND, USC.HEALTH_PCT, USC.X_NORM]
    )
    assert resolved == [USC.ROLE_DEFEND, USC.HEALTH_PCT, USC.X_NORM]


def test_resolve_active_features_dedup_and_casefold():
    resolved = USC.resolve_active_features(["Health_Pct", "health_pct", "X_NORM"])
    assert resolved == [USC.HEALTH_PCT, USC.X_NORM]


def test_resolve_active_features_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown unit scalar feature"):
        USC.resolve_active_features(["not_a_real_feature"])
