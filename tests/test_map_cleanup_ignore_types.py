"""MAP_CLEANUP_IGNORE_TYPE_IDS keeps static map props out of reset cleanup logic."""

import pytest
from unittest.mock import MagicMock

from sc2.data import Alliance
from sc2.ids.unit_typeid import UnitTypeId

from src.sc2_bot import MAP_CLEANUP_IGNORE_TYPE_IDS, unit_is_subject_to_map_cleanup


def test_collapsible_rock_tower_in_ignore_set_when_enum_exists():
    tower = getattr(UnitTypeId, "COLLAPSIBLEROCKTOWER", None)
    if tower is not None:
        assert tower in MAP_CLEANUP_IGNORE_TYPE_IDS


def test_unit_is_subject_to_map_cleanup_skips_collapsible_rock():
    tower = getattr(UnitTypeId, "COLLAPSIBLEROCKTOWER", None)
    if tower is None:
        pytest.skip("COLLAPSIBLEROCKTOWER not in UnitTypeId")
    u = MagicMock()
    u.alliance = Alliance.Self
    u.type_id = tower
    assert unit_is_subject_to_map_cleanup(u) is False


def test_unit_is_subject_to_map_cleanup_skips_collapsible_purifier_tower_diagonal():
    tower = getattr(UnitTypeId, "COLLAPSIBLEPURIFIERTOWERDIAGONAL", None)
    if tower is None:
        pytest.skip("COLLAPSIBLEPURIFIERTOWERDIAGONAL not in UnitTypeId")
    u = MagicMock()
    u.alliance = Alliance.Self
    u.type_id = tower
    assert unit_is_subject_to_map_cleanup(u) is False


def test_unit_is_subject_to_map_cleanup_skips_collapsible_terran_tower_diagonal():
    tower = getattr(UnitTypeId, "COLLAPSIBLETERRANTOWERDIAGONAL", None)
    if tower is None:
        pytest.skip("COLLAPSIBLETERRANTOWERDIAGONAL not in UnitTypeId")
    u = MagicMock()
    u.alliance = Alliance.Self
    u.type_id = tower
    assert unit_is_subject_to_map_cleanup(u) is False


def test_unit_is_subject_to_map_cleanup_skips_collapsible_rock_tower_diagonal():
    tower = getattr(UnitTypeId, "COLLAPSIBLEROCKTOWERDIAGONAL", None)
    if tower is None:
        pytest.skip("COLLAPSIBLEROCKTOWERDIAGONAL not in UnitTypeId")
    u = MagicMock()
    u.alliance = Alliance.Self
    u.type_id = tower
    assert unit_is_subject_to_map_cleanup(u) is False


def test_unit_is_subject_to_map_cleanup_includes_non_neutral_marine():
    u = MagicMock()
    u.alliance = Alliance.Self
    u.type_id = UnitTypeId.MARINE
    u.is_mineral_field = False
    u.is_vespene_geyser = False
    assert unit_is_subject_to_map_cleanup(u) is True


def test_unit_is_subject_to_map_cleanup_skips_neutral():
    u = MagicMock()
    u.alliance = Alliance.Neutral
    u.type_id = UnitTypeId.MARINE
    u.is_mineral_field = False
    u.is_vespene_geyser = False
    assert unit_is_subject_to_map_cleanup(u) is False


def test_unit_is_subject_to_map_cleanup_skips_raw_int_neutral():
    """Proto often exposes alliance as int 3; Enum inequality used to miss this."""
    u = MagicMock()
    u.alliance = 3
    u.type_id = UnitTypeId.MINERALFIELD
    u.is_mineral_field = False
    u.is_vespene_geyser = False
    assert unit_is_subject_to_map_cleanup(u) is False


def test_unit_is_subject_to_map_cleanup_skips_mineral_field_flag():
    u = MagicMock()
    u.alliance = Alliance.Self  # mislabeled; flag still wins
    u.type_id = UnitTypeId.MINERALFIELD
    u.is_mineral_field = True
    u.is_vespene_geyser = False
    assert unit_is_subject_to_map_cleanup(u) is False


def test_unit_is_subject_to_map_cleanup_skips_neutral_geyser_flag():
    u = MagicMock()
    u.alliance = Alliance.Neutral
    u.type_id = UnitTypeId.VESPENEGEYSER
    u.is_mineral_field = False
    u.is_vespene_geyser = True
    assert unit_is_subject_to_map_cleanup(u) is False


def test_unit_is_subject_to_map_cleanup_includes_self_extractor():
    """Regression: burnysc2 Extractors report is_vespene_geyser=True but must be killed."""
    u = MagicMock()
    u.alliance = Alliance.Self
    u.type_id = UnitTypeId.EXTRACTOR
    u.is_mineral_field = False
    u.is_vespene_geyser = True
    assert unit_is_subject_to_map_cleanup(u) is True


def test_unit_is_subject_to_map_cleanup_includes_enemy_refinery():
    u = MagicMock()
    u.alliance = Alliance.Enemy
    u.type_id = UnitTypeId.REFINERY
    u.is_mineral_field = False
    u.is_vespene_geyser = True
    assert unit_is_subject_to_map_cleanup(u) is True
