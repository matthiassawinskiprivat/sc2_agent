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


def test_widowmine_included_without_weapons():
    # Unburrowed Widow Mine has no weapons; combat is on the burrowed form.
    assert "WIDOWMINE" in WHITELIST
    assert should_include_unit("WIDOWMINE", has_weapons=False) is True


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


def test_widowmine_overrides_fix_cost_and_sight():
    unit_db = {
        "WIDOWMINE": {
            "id": 498,
            "name": "WIDOWMINE",
            "cost": {"minerals": 0, "vespene": 0, "supply": 0.0},
            "radius": 0.0,
            "sight": 9.0,
            "detect": 0.0,
            "weapons": [],
        }
    }
    apply_unit_field_overrides(unit_db)
    entry = unit_db["WIDOWMINE"]
    assert entry["radius"] == 0.5
    assert entry["sight"] == 7.0
    assert entry["cost"] == {"minerals": 75, "vespene": 25, "supply": 2.0}
    assert entry["weapons"] == []


def test_raven_and_overseer_whitelisted_with_detect():
    assert "RAVEN" in WHITELIST
    assert "OVERSEER" in WHITELIST
    assert should_include_unit("RAVEN", has_weapons=False) is True
    assert should_include_unit("OVERSEER", has_weapons=False) is True
    unit_db = {
        "RAVEN": {
            "id": 56,
            "name": "RAVEN",
            "cost": {"minerals": 0, "vespene": 0, "supply": 0.0},
            "radius": 0.0,
            "sight": 9.0,
            "detect": 0.0,
            "weapons": [],
        },
        "OVERSEER": {
            "id": 129,
            "name": "OVERSEER",
            "cost": {"minerals": 0, "vespene": 0, "supply": 0.0},
            "radius": 0.0,
            "sight": 9.0,
            "detect": 0.0,
            "weapons": [],
        },
    }
    apply_unit_field_overrides(unit_db)
    assert unit_db["RAVEN"]["detect"] == 11.0
    assert unit_db["OVERSEER"]["detect"] == 11.0
    assert unit_db["RAVEN"]["cost"]["vespene"] == 200


def test_baneling_override_forces_melee_weapon_range():
    unit_db = {
        "BANELING": {
            "id": 9,
            "name": "BANELING",
            "cost": {"minerals": 50, "vespene": 25, "supply": 0.5},
            "radius": 0.375,
            "sight": 9.0,
            "detect": 0.0,
            "weapons": [
                {
                    "type": 1,
                    "damage": 20.0,
                    "attacks": 1,
                    "period": 1.0,
                    "range": 6.0,
                    "upgrade_bonus": 2,
                    "bonuses": [{"type": 1, "bonus": 15.0}],
                }
            ],
        }
    }
    apply_unit_field_overrides(unit_db)
    assert unit_db["BANELING"]["weapons"][0]["range"] == 0.25
    assert unit_db["BANELING"]["weapons"][0]["damage"] == 20.0

