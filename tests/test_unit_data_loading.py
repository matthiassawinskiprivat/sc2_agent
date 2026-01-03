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
    
    required_units = ["ZERGLING", "ROACH", "MARINE", "REAPER", "ZEALOT"]
    for unit in required_units:
        assert unit in data, f"Unit {unit} missing from unit_data.json"
        assert "cost" in data[unit], f"Cost missing for {unit}"
        assert "minerals" in data[unit]["cost"], f"Mineral cost missing for {unit}"
        assert "vespene" in data[unit]["cost"], f"Vespene cost missing for {unit}"

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
