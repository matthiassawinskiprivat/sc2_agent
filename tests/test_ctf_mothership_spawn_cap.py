"""Protoss Mothership must appear at most once per budgeted CTF spawn."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

_HELPER = Path(__file__).resolve().parents[1] / "train" / "envs" / "ctf" / "budgeted_spawns.py"
_spec = importlib.util.spec_from_file_location("budgeted_spawns", _HELPER)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
calculate_budgeted_spawns = _mod.calculate_budgeted_spawns


UNIT_DATA = {
    "ZEALOT": {"cost": {"minerals": 100, "vespene": 0}},
    "STALKER": {"cost": {"minerals": 125, "vespene": 50}},
    "MOTHERSHIP": {"cost": {"minerals": 400, "vespene": 400}},
    "CARRIER": {"cost": {"minerals": 350, "vespene": 250}},
}


def test_mothership_capped_at_one_even_with_huge_budget():
    rng = random.Random(0)
    spawns = calculate_budgeted_spawns(
        ["MOTHERSHIP", "ZEALOT"],
        UNIT_DATA,
        minerals=5000,
        vespene=5000,
        rng=rng,
    )
    assert spawns.count("MOTHERSHIP") == 1
    assert spawns.count("ZEALOT") >= 1


def test_mothership_cap_still_spends_remaining_budget_on_others():
    rng = random.Random(1)
    spawns = calculate_budgeted_spawns(
        ["MOTHERSHIP", "ZEALOT"],
        UNIT_DATA,
        minerals=900,
        vespene=400,
        rng=rng,
    )
    assert spawns.count("MOTHERSHIP") <= 1
    spent_m = sum(UNIT_DATA[u]["cost"]["minerals"] for u in spawns)
    spent_v = sum(UNIT_DATA[u]["cost"]["vespene"] for u in spawns)
    assert spent_m <= 900
    assert spent_v <= 400


def test_no_mothership_when_not_in_pool():
    spawns = calculate_budgeted_spawns(
        ["ZEALOT", "STALKER"],
        UNIT_DATA,
        minerals=2000,
        vespene=500,
        rng=random.Random(2),
    )
    assert "MOTHERSHIP" not in spawns
