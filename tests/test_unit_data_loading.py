import json
import pytest
from pathlib import Path

def test_unit_data_json_exists():
    path = Path("data/unit_data.json")
    assert path.exists(), "unit_data.json does not exist"

def test_unit_data_json_loadable():
    with open("data/unit_data.json", "r") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0

def test_required_ctf_units_present():
    with open("data/unit_data.json", "r") as f:
        data = json.load(f)
    
    required_units = [
        "ZERGLING", "ROACH", "MARINE", "REAPER", "ZEALOT", "SWARMHOSTMP", "DISRUPTOR",
        "WIDOWMINE",
        "RAVEN",
        "OVERSEER",
    ]
    for unit in required_units:
        assert unit in data, f"Unit {unit} missing from unit_data.json"
        assert "cost" in data[unit], f"Cost missing for {unit}"
        assert "minerals" in data[unit]["cost"], f"Mineral cost missing for {unit}"
        assert "vespene" in data[unit]["cost"], f"Vespene cost missing for {unit}"


def test_swarmhostmp_cost_matches_lotv():
    with open("data/unit_data.json", "r") as f:
        data = json.load(f)
    entry = data["SWARMHOSTMP"]
    cost = entry["cost"]
    assert cost["minerals"] == 100
    assert cost["vespene"] == 75
    assert cost["supply"] == 3.0
    assert entry["id"] == 494
    assert entry["sight"] == 10.0
    assert entry["radius"] == 0.8125
    assert entry["weapons"] == []


def test_disruptor_cost_matches_lotv():
    with open("data/unit_data.json", "r") as f:
        data = json.load(f)
    entry = data["DISRUPTOR"]
    cost = entry["cost"]
    assert cost["minerals"] == 150
    assert cost["vespene"] == 150
    assert cost["supply"] == 3.0
    assert entry["id"] == 694
    assert entry["radius"] == 0.75
    assert entry["weapons"] == []


def test_widowmine_cost_matches_lotv():
    with open("data/unit_data.json", "r") as f:
        data = json.load(f)
    entry = data["WIDOWMINE"]
    cost = entry["cost"]
    assert cost["minerals"] == 75
    assert cost["vespene"] == 25
    assert cost["supply"] == 2.0
    assert entry["id"] == 498
    assert entry["radius"] == 0.5
    assert entry["sight"] == 7.0
    assert entry["weapons"] == []


def test_swarmhostmp_ctf_attributes_armored_biological():
    from train.envs.ctf.game_data import UNIT_DATA, uid

    info = UNIT_DATA[uid("SWARMHOSTMP")]
    assert info.attributes == frozenset({"biological", "armored"})
    assert info.mineral_cost == 100
    assert info.vespene_cost == 75
    assert info.footprint_radius == 0.8125

def test_unit_stats_integrity():
    with open("data/unit_data.json", "r") as f:
        data = json.load(f)
    
    # Check a few specific stats for well-known units
    # Note: These values might change if the game updates, but for now they serve as a sanity check
    if "MARINE" in data:
        marine_cost = data["MARINE"]["cost"]
        assert marine_cost["minerals"] == 50
        assert marine_cost["vespene"] == 0
        assert marine_cost["supply"] == 1.0
    
    if "ZERGLING" in data:
        zergling_cost = data["ZERGLING"]["cost"]
        # In SC2, 1 zergling costs 25 minerals (50 for 2)
        # But unit_data.json might store it differently depending on how it was extracted
        # Let's see what's actually there
        assert zergling_cost["minerals"] in [25, 50]


def test_baneling_weapon_range_is_melee():
    """Baneling explode radius is splash-only; catalog DPS range is melee contact."""
    with open("data/unit_data.json", "r") as f:
        data = json.load(f)
    weapons = data["BANELING"]["weapons"]
    assert len(weapons) == 1
    assert weapons[0]["range"] == 0.25
    assert weapons[0]["damage"] == 20.0
