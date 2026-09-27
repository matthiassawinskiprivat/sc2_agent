"""P1-only ``debug_upgrade``: conflict levels are split with a reversed sibling."""
from pathlib import Path

import yaml

from src.utils.config_manager import ConfigManager
from train.envs.ctf.pool_assignment import get_level_races, static_spawn_player_id

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

# Researched cast abilities that only unlock for SC2 player 1.
_ABILITY_TECH_UNITS = {
    "INFESTOR": "NEURALPARASITE",
    "INFESTORBURROWED": "NEURALPARASITE",
    "BATTLECRUISER": "YAMATOCANNON",
    "HIGHTEMPLAR": "PSISTORMTECH",
}
_SNIPE_RESEARCH_UNITS = {
    "INFESTOR": "NEURALPARASITE",
    "INFESTORBURROWED": "NEURALPARASITE",
    "BATTLECRUISER": "YAMATOCANNON",
}


def _load(level: str) -> dict:
    return yaml.safe_load((_CONFIG_DIR / f"{level}.yaml").read_text(encoding="utf-8"))


def _races_and_tech(pools: list) -> tuple[list[str], dict[str, set[str]]]:
    races = get_level_races(pools)
    tech: dict[str, set[str]] = {r: set() for r in races}
    for pool in pools:
        race = str(pool.get("race", "")).lower().strip()
        for unit in pool.get("unit_pool") or []:
            key = str(unit).upper()
            if key in _ABILITY_TECH_UNITS:
                tech[race].add(key)
    return races, tech


def _needs_reversed_sibling(name: str, cfg: dict) -> bool:
    pools = (cfg.get("env") or {}).get("unit_pools") or []
    races, tech = _races_and_tech(pools)
    if len(races) < 2:
        return False
    if sum(1 for r in races if tech[r]) >= 2:
        return True
    if name.startswith("snipe_"):
        p2_units = set()
        for pool in pools:
            if static_spawn_player_id(pool, pools) == 2:
                p2_units.update(str(u).upper() for u in pool.get("unit_pool") or [])
        if p2_units & set(_SNIPE_RESEARCH_UNITS):
            return True
    return False


def test_ability_tech_conflict_levels_have_reversed_sibling_with_flipped_p1():
    """From snipe onward, each conflict level is followed by ``*_reversed``.

    Pre-snipe AoE / heal-revisit stages stay unsplit so a pipeline restart can
    CONTINUE_TRAINING past already-completed AoE without hitting empty reversed
    siblings inserted mid-curriculum.
    """
    cm = ConfigManager(str(_CONFIG_DIR))
    snipe_start = cm.levels.index("snipe_ghost_marauder_vs_roach_hydra")
    conflict_parents = []
    for name in cm.levels:
        if name.endswith("_reversed"):
            continue
        if cm.levels.index(name) < snipe_start:
            continue
        cfg = _load(name)
        if not _needs_reversed_sibling(name, cfg):
            continue
        conflict_parents.append(name)
        rev = f"{name}_reversed"
        assert rev in cm.levels, name
        assert cm.levels.index(rev) == cm.levels.index(name) + 1, name

        parent_pools = cfg["env"]["unit_pools"]
        rev_pools = _load(rev)["env"]["unit_pools"]
        parent_races = get_level_races(parent_pools)
        rev_races = get_level_races(rev_pools)
        assert len(parent_races) == 2
        assert rev_races[0] == parent_races[1]
        assert rev_races[1] == parent_races[0]

        parent_steps = int(cfg["training"]["training_steps"])
        rev_steps = int(_load(rev)["training"]["training_steps"])
        assert parent_steps == rev_steps
        # Split halves a 250k budget; allow other base budgets but require equality.
        assert parent_steps <= 250_000

    assert "snipe_ghost_marauder_vs_roach_hydra" in conflict_parents
    assert "snipe_ghost_marauder_vs_roach_hydra_reversed" in cm.levels
    assert "aoe_zergling_infestor_vs_stalker_templar_reversed" not in cm.levels
    assert "scenario_queen_vs_battlecruiser_reversed" not in cm.levels
    assert len(conflict_parents) >= 18


def test_reversed_siblings_do_not_redeclare_cumulative_obs_or_heads():
    """Reversed YAMLs only flip P1; they must not re-append layers/heads."""
    cm = ConfigManager(str(_CONFIG_DIR))
    for name in cm.levels:
        if not name.endswith("_reversed"):
            continue
        env = _load(name)["env"]
        assert not env.get("active_layers"), name
        assert not env.get("active_tactical_layers"), name
        assert not env.get("active_unit_scalars"), name
        assert not env.get("action_heads"), name
        assert not env.get("categorical_heads"), name
