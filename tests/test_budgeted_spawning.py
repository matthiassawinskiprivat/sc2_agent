import pytest
import random
from unittest.mock import MagicMock

# Mock unit data for testing
MOCK_UNIT_DATA = {
    "ZERGLING": {"cost": {"minerals": 25, "vespene": 0, "supply": 0.5}},
    "ROACH": {"cost": {"minerals": 75, "vespene": 25, "supply": 2.0}},
    "MARINE": {"cost": {"minerals": 50, "vespene": 0, "supply": 1.0}},
    "REAPER": {"cost": {"minerals": 50, "vespene": 50, "supply": 1.0}},
    "ZEALOT": {"cost": {"minerals": 100, "vespene": 0, "supply": 2.0}}
}

def calculate_spawns(minerals, vespene, unit_pool, unit_data):
    spawns = []
    m, v = minerals, vespene
    
    def can_afford(u):
        cost = unit_data[u]["cost"]
        return cost["minerals"] <= m and cost["vespene"] <= v

    affordable = [u for u in unit_pool if can_afford(u)]
    while affordable:
        u = random.choice(affordable)
        spawns.append(u)
        m -= unit_data[u]["cost"]["minerals"]
        v -= unit_data[u]["cost"]["vespene"]
        affordable = [u for u in unit_pool if can_afford(u)]
    
    return spawns, m, v

def test_calculate_spawns_basic():
    unit_pool = ["MARINE"]
    spawns, rem_m, rem_v = calculate_spawns(150, 0, unit_pool, MOCK_UNIT_DATA)
    assert len(spawns) == 3
    assert all(u == "MARINE" for u in spawns)
    assert rem_m == 0

def test_calculate_spawns_mixed_budget():
    unit_pool = ["ZERGLING", "ROACH"]
    # 200 minerals, 25 vespene
    # Should spawn 1 roach (75m, 25v) and then as many zerglings as possible
    # 200 - 75 = 125 minerals left. 125 / 25 = 5 zerglings.
    # Total: 1 Roach, 5 Zerglings.
    # Note: random.choice might pick zerglings first.
    # Total cost will always be 200m, 25v if it fills up.
    
    # We run multiple times to check stability
    for _ in range(10):
        spawns, rem_m, rem_v = calculate_spawns(200, 25, unit_pool, MOCK_UNIT_DATA)
        total_m = sum(MOCK_UNIT_DATA[u]["cost"]["minerals"] for u in spawns)
        total_v = sum(MOCK_UNIT_DATA[u]["cost"]["vespene"] for u in spawns)
        assert total_m <= 200
        assert total_v <= 25
        # Ensure no more can be spawned
        assert not any(MOCK_UNIT_DATA[u]["cost"]["minerals"] <= rem_m and 
                       MOCK_UNIT_DATA[u]["cost"]["vespene"] <= rem_v for u in unit_pool)

def test_calculate_spawns_no_vespene():
    unit_pool = ["ROACH", "REAPER"]
    # 500 minerals, 0 vespene. 
    # Both ROACH and REAPER require vespene. Should spawn nothing.
    spawns, rem_m, rem_v = calculate_spawns(500, 0, unit_pool, MOCK_UNIT_DATA)
    assert len(spawns) == 0
    assert rem_m == 500
    assert rem_v == 0

def test_calculate_spawns_expensive_pool():
    unit_pool = ["ZEALOT"]
    # 50 minerals. Zealot costs 100.
    spawns, rem_m, rem_v = calculate_spawns(50, 0, unit_pool, MOCK_UNIT_DATA)
    assert len(spawns) == 0


def test_ctf_level_01_attacker_can_afford_reaper():
    """Regression: Reaper is 50/50; gas=0 used to spawn an empty attacker army."""
    import json
    from pathlib import Path

    import yaml

    from train.envs.ctf.budgeted_spawns import calculate_budgeted_spawns

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config_universal" / "ctf_level_01.yaml").read_text())
    unit_data = json.loads((root / "data" / "unit_data.json").read_text())
    attack = next(p for p in cfg["env"]["unit_pools"] if p.get("high_level_command") == "attack")
    spawns = calculate_budgeted_spawns(
        attack["unit_pool"],
        unit_data,
        attack["mineral_budget"],
        attack["vespene_budget"],
    )
    assert spawns, "attacker pool must spawn at least one unit"
    assert all(u == "REAPER" for u in spawns)
