"""Regression: weaponless curriculum units must survive unit_data.json generation."""

from src.tools.unit_data_inclusion import (
    WHITELIST,
    apply_unit_field_overrides,
    should_include_unit,
)


def test_swarmhostmp_included_without_weapons():
    # Swarm Host has no own weapon (locusts do); without whitelist it was dropped.
    assert "SWARMHOSTMP" in WHITELIST
    assert should_include_unit("SWARMHOSTMP", has_weapons=False) is True


def test_disruptor_included_without_weapons():
    assert "DISRUPTOR" in WHITELIST
    assert should_include_unit("DISRUPTOR", has_weapons=False) is True


def test_weaponless_unknown_unit_excluded():
    assert should_include_unit("SOMEFAKEUNIT", has_weapons=False) is False


def test_armed_unit_included_without_whitelist():
    assert should_include_unit("SOMEFAKEUNIT", has_weapons=True) is True


def test_infestor_still_whitelisted():
    assert should_include_unit("INFESTOR", has_weapons=False) is True


def test_swarmhostmp_overrides_fix_sight_and_radius():
    unit_db = {
        "SWARMHOSTMP": {
            "id": 494,
            "name": "SWARMHOSTMP",
            "cost": {"minerals": 0, "vespene": 0, "supply": 0.0},
            "radius": 0.0,
            "sight": 9.0,
            "detect": 0.0,
            "weapons": [],
        }
    }
    apply_unit_field_overrides(unit_db)
    entry = unit_db["SWARMHOSTMP"]
    assert entry["sight"] == 10.0
    assert entry["radius"] == 0.8125
    assert entry["cost"] == {"minerals": 100, "vespene": 75, "supply": 3.0}
    assert entry["weapons"] == []


def test_disruptor_overrides_fix_cost_and_weapons():
    unit_db = {
        "DISRUPTOR": {
            "id": 694,
            "name": "DISRUPTOR",
            "cost": {"minerals": 0, "vespene": 0, "supply": 0.0},
            "radius": 0.0,
            "sight": 9.0,
            "detect": 0.0,
            "weapons": [],
        }
    }
    apply_unit_field_overrides(unit_db)
    entry = unit_db["DISRUPTOR"]
    assert entry["radius"] == 0.75
    assert entry["cost"] == {"minerals": 150, "vespene": 150, "supply": 3.0}
    assert entry["weapons"] == []
