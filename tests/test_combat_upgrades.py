"""Combat upgrade sampler and cost helpers."""
import random

from src.utils.combat_upgrades import (
    all_combat_upgrade_names,
    combat_tracks_for_race,
    cumulative_upgrade_cost,
    sample_pool_upgrades,
    upgrade_ids_for_tiers,
)


def test_cumulative_cost_tiers():
    assert cumulative_upgrade_cost(0) == (0, 0)
    assert cumulative_upgrade_cost(1) == (100, 100)
    assert cumulative_upgrade_cost(2) == (275, 275)
    assert cumulative_upgrade_cost(3) == (525, 525)


def test_sample_respects_budget():
    rng = random.Random(0)
    budget_m, budget_v = 200, 200
    chosen, rem_m, rem_v = sample_pool_upgrades("zerg", budget_m, budget_v, rng=rng)
    total_m = total_v = 0
    for tier in chosen.values():
        cm, cv = cumulative_upgrade_cost(tier)
        total_m += cm
        total_v += cv
    assert total_m <= budget_m and total_v <= budget_v
    assert rem_m == budget_m - total_m
    assert rem_v == budget_v - total_v
    assert set(chosen) == {t.name for t in combat_tracks_for_race("zerg")}


def test_upgrade_ids_only_combat():
    chosen = {t.name: 1 for t in combat_tracks_for_race("protoss")}
    ids = upgrade_ids_for_tiers(chosen)
    names = {getattr(u, "name", str(u)) for u in ids}
    assert names <= all_combat_upgrade_names()
    assert "CHARGE" not in names
    assert "STIMPACK" not in names


def test_zero_budget_all_zero_tiers():
    chosen, rem_m, rem_v = sample_pool_upgrades("terran", 0, 0, rng=random.Random(1))
    assert all(t == 0 for t in chosen.values())
    assert rem_m == 0 and rem_v == 0
