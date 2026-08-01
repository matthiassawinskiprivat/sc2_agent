"""Tests for CTF pool assignment and multi-race curriculum configs."""

from pathlib import Path
import importlib.util
import sys
import types

import yaml

from src.utils.config_manager import ConfigManager

# Load pool_assignment without importing train.envs (pulls SC2 bot stack).
_pool_path = Path(__file__).resolve().parents[1] / "train" / "envs" / "ctf" / "pool_assignment.py"
_train = types.ModuleType("train")
_train_envs = types.ModuleType("train.envs")
_train_ctf = types.ModuleType("train.envs.ctf")
sys.modules.setdefault("train", _train)
sys.modules.setdefault("train.envs", _train_envs)
sys.modules.setdefault("train.envs.ctf", _train_ctf)
_spec = importlib.util.spec_from_file_location("train.envs.ctf.pool_assignment", _pool_path)
_pool_mod = importlib.util.module_from_spec(_spec)
sys.modules["train.envs.ctf.pool_assignment"] = _pool_mod
_spec.loader.exec_module(_pool_mod)
get_trainee_race = _pool_mod.get_trainee_race
is_pool_trainee = _pool_mod.is_pool_trainee
pool_player_id = _pool_mod.pool_player_id

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

_CTF_BOOTSTRAP_LEVELS = frozenset({"ctf_level_01", "ctf_level_01_zergling_vs_marine"})

_RACE_UNITS = {
    "terran": {
        "MARINE", "MARAUDER", "GHOST", "HELLION", "HELLIONTANK", "SIEGETANK", "VIKINGFIGHTER", "BANSHEE", "THOR",
        "CYCLONE", "REAPER", "BATTLECRUISER",
    },
    "zerg": {
        "ZERGLING", "ROACH", "HYDRALISK", "MUTALISK", "ULTRALISK", "RAVAGER", "BANELING",
        "CORRUPTOR", "SWARMHOSTMP", "BROODLORD",
    },
    "protoss": {
        "ZEALOT", "STALKER", "ADEPT", "IMMORTAL", "PHOENIX", "VOIDRAY", "COLOSSUS", "ARCHON",
        "DISRUPTOR", "TEMPEST", "CARRIER",
    },
}

# Levels 06+ grow beyond the early ≤3 unit-type curriculum pools.
_CTF_LARGE_POOL_MIN_NUMBER = 6


def _load_env(level_name: str) -> dict:
    data = yaml.safe_load((_CONFIG_DIR / f"{level_name}.yaml").read_text(encoding="utf-8"))
    return data["env"]


def _load_unit_pools(level_name: str):
    return _load_env(level_name)["unit_pools"]


def test_pool_assignment_alternates_trainee_race_each_episode():
    pools = [
        {"race": "terran", "high_level_command": "attack"},
        {"race": "zerg", "high_level_command": "defend"},
    ]
    assert get_trainee_race(pools, 1, alternate=True) == "terran"
    assert get_trainee_race(pools, 2, alternate=True) == "zerg"
    assert get_trainee_race(pools, 3, alternate=True) == "terran"
    assert pool_player_id(pools[0], pools, 1, alternate=True) == 1
    assert pool_player_id(pools[1], pools, 1, alternate=True) == 2
    assert pool_player_id(pools[0], pools, 2, alternate=True) == 2
    assert pool_player_id(pools[1], pools, 2, alternate=True) == 1


def test_pool_assignment_legacy_agent_flag_when_not_alternating():
    pools = [
        {"agent": True, "high_level_command": "attack"},
        {"high_level_command": "defend"},
    ]
    assert pool_player_id(pools[0], pools, 1, alternate=False) == 1
    assert pool_player_id(pools[1], pools, 1, alternate=False) == 2
    assert is_pool_trainee(pools[0], pools, 1, alternate=False)


def test_ctf_levels_enable_alternate_trainee_race_except_bootstrap():
    cm = ConfigManager(str(_CONFIG_DIR))
    ctf_levels = [lvl for lvl in cm.levels if lvl.startswith("ctf_level_")]
    for level in ctf_levels:
        env = _load_env(level)
        if level in _CTF_BOOTSTRAP_LEVELS:
            assert env.get("alternate_trainee_race") is False, level
            continue
        assert env.get("alternate_trainee_race") is True, level


def test_ctf_levels_include_race_from_level_12():
    """Race obs encoding starts after ctf_level_11 (query_linear 138→144)."""
    cm = ConfigManager(str(_CONFIG_DIR))
    ctf_levels = [lvl for lvl in cm.levels if lvl.startswith("ctf_level_")]
    for level in ctf_levels:
        env = _load_env(level)
        # Extract numeric suffix when present (ctf_level_12 → 12); skip named variants.
        parts = level.replace("ctf_level_", "").split("_", 1)
        try:
            num = int(parts[0])
        except ValueError:
            assert not env.get("include_race"), level
            continue
        if num >= 12:
            assert env.get("include_race") is True, level
        else:
            assert not env.get("include_race"), level


def test_ctf_levels_define_race_per_pool_static_agent_only_on_bootstrap():
    cm = ConfigManager(str(_CONFIG_DIR))
    ctf_levels = [lvl for lvl in cm.levels if lvl.startswith("ctf_level_")]
    for level in ctf_levels:
        env = _load_env(level)
        for pool in _load_unit_pools(level):
            assert "race" in pool, f"{level} missing race on pool"
            if level in _CTF_BOOTSTRAP_LEVELS and not env.get("alternate_trainee_race", False):
                if pool.get("agent"):
                    assert pool.get("high_level_command") == "attack"
                continue
            assert "agent" not in pool, f"{level} still uses static agent flag"


def test_ctf_level_01_zergling_attack_vs_marine_defend():
    env = _load_env("ctf_level_01_zergling_vs_marine")
    assert env.get("alternate_trainee_race") is False
    pools = env["unit_pools"]
    assert len(pools) == 2
    assert pools[0]["agent"] is True
    assert pools[0]["race"] == "zerg"
    assert pools[0]["high_level_command"] == "attack"
    assert pools[0]["unit_pool"] == ["ZERGLING"]
    assert pools[0]["mineral_budget"] == 200
    assert pools[1]["race"] == "terran"
    assert pools[1]["high_level_command"] == "defend"
    assert pools[1]["unit_pool"] == ["MARINE"]
    assert pools[1]["mineral_budget"] == 150
    assert pools[0]["region"] == pools[1]["region"] == "A"
    assert env["squad_selection"]["min_selected_units"] == 8


def test_ctf_pipeline_inserts_zergling_vs_marine_after_level_01():
    cm = ConfigManager(str(_CONFIG_DIR))

    idx_01 = cm.levels.index("ctf_level_01")
    idx_zvm = cm.levels.index("ctf_level_01_zergling_vs_marine")
    idx_melee_1 = cm.levels.index("ctf_level_02_melee_zealot_vs_roach")

    assert idx_01 < idx_zvm < idx_melee_1


def test_ctf_level_01_reaper_attack_vs_zergling_defend():
    env = _load_env("ctf_level_01")
    assert env.get("alternate_trainee_race") is False
    pools = env["unit_pools"]
    assert len(pools) == 2
    assert pools[0]["agent"] is True
    assert pools[0]["race"] == "terran"
    assert pools[0]["high_level_command"] == "attack"
    assert pools[0]["unit_pool"] == ["REAPER"]
    assert pools[1]["race"] == "zerg"
    assert pools[1]["high_level_command"] == "defend"
    assert pools[1]["unit_pool"] == ["ZERGLING"]
    assert pools[0]["region"] == pools[1]["region"] == "A"


def test_ctf_unit_pools_match_declared_race():
    cm = ConfigManager(str(_CONFIG_DIR))
    ctf_levels = [lvl for lvl in cm.levels if lvl.startswith("ctf_level_")]
    for level in ctf_levels:
        for pool in _load_unit_pools(level):
            race = pool["race"]
            units = set(pool.get("unit_pool", []))
            assert units <= _RACE_UNITS[race], f"{level} {race} pool has {units - _RACE_UNITS[race]}"


def test_ctf_levels_use_at_most_three_unit_types_per_pool_before_large_pools():
    import re

    cm = ConfigManager(str(_CONFIG_DIR))
    ctf_levels = [lvl for lvl in cm.levels if lvl.startswith("ctf_level_")]
    for level in ctf_levels:
        m = re.match(r"ctf_level_(\d+)", level)
        if m and int(m.group(1)) >= _CTF_LARGE_POOL_MIN_NUMBER:
            continue
        for pool in _load_unit_pools(level):
            assert len(pool.get("unit_pool", [])) <= 3, level


def test_ctf_level_12_adds_remaining_air_dps_layers():
    cm = ConfigManager(str(_CONFIG_DIR))
    prior = set(cm.get_cumulative_active_layers("ctf_level_11"))
    at_12 = cm.get_cumulative_active_layers("ctf_level_12")
    added = [layer for layer in at_12 if layer not in prior]
    assert added == [
        "allied_a_dps_light",
        "enemy_a_dps_light",
        "allied_a_dps_armor",
        "enemy_a_dps_armor",
        "allied_a_dps_massive",
        "enemy_a_dps_massive",
    ]


def test_ctf_levels_12_14_budgets_and_matchups():
    """12–14 continue 09–11 style: PvT / ZvP / ZvT with raised budgets."""
    expected = {
        "ctf_level_12": ("protoss", "terran"),
        "ctf_level_13": ("zerg", "protoss"),
        "ctf_level_14": ("zerg", "terran"),
    }
    for level, (attack_race, defend_race) in expected.items():
        pools = _load_unit_pools(level)
        assert len(pools) == 4, level
        a_attack = next(p for p in pools if p["region"] == "A" and p["high_level_command"] == "attack")
        a_defend = next(p for p in pools if p["region"] == "A" and p["high_level_command"] == "defend")
        assert a_attack["race"] == attack_race
        assert a_defend["race"] == defend_race
        assert a_attack["mineral_budget"] == 2000
        assert a_attack["vespene_budget"] == 2000
        assert a_defend["mineral_budget"] == 1500
        assert a_defend["vespene_budget"] == 1500
        if attack_race == "protoss" or defend_race == "protoss":
            protoss_pool = a_attack if attack_race == "protoss" else a_defend
            assert "DISRUPTOR" in protoss_pool["unit_pool"]


def test_ctf_levels_15_17_mirror_12_14_matchups_and_budgets():
    """15–17 reuse 12–14 full-army PvT / ZvP / ZvT pools and budgets.

    Zerg pools on 16–17 also include BANELING (splash curriculum).
    """
    expected = {
        "ctf_level_15": ("protoss", "terran"),
        "ctf_level_16": ("zerg", "protoss"),
        "ctf_level_17": ("zerg", "terran"),
    }
    mirrors = {
        "ctf_level_15": "ctf_level_12",
        "ctf_level_16": "ctf_level_13",
        "ctf_level_17": "ctf_level_14",
    }
    for level, (attack_race, defend_race) in expected.items():
        pools = _load_unit_pools(level)
        assert len(pools) == 4, level
        a_attack = next(p for p in pools if p["region"] == "A" and p["high_level_command"] == "attack")
        a_defend = next(p for p in pools if p["region"] == "A" and p["high_level_command"] == "defend")
        assert a_attack["race"] == attack_race
        assert a_defend["race"] == defend_race
        assert a_attack["mineral_budget"] == 2000
        assert a_defend["mineral_budget"] == 1500
        mirrored = _load_unit_pools(mirrors[level])
        for pool, mirror_pool in zip(pools, mirrored):
            if pool["race"] == "zerg":
                assert "BANELING" in pool["unit_pool"], level
                expected_pool = list(mirror_pool["unit_pool"])
                if "BANELING" not in expected_pool:
                    if "ZERGLING" in expected_pool:
                        idx = expected_pool.index("ZERGLING") + 1
                        expected_pool.insert(idx, "BANELING")
                    else:
                        expected_pool.insert(0, "BANELING")
                assert pool["unit_pool"] == expected_pool, level
            else:
                assert pool["unit_pool"] == mirror_pool["unit_pool"], level


def test_ctf_pipeline_inserts_mirror_matchups_after_level_17():
    cm = ConfigManager(str(_CONFIG_DIR))
    idx_17 = cm.levels.index("ctf_level_17")
    idx_t = cm.levels.index("ctf_level_17_mirror_terran")
    idx_z = cm.levels.index("ctf_level_17_mirror_zerg")
    idx_p = cm.levels.index("ctf_level_17_mirror_protoss")
    idx_18 = cm.levels.index("ctf_level_18")
    assert idx_17 < idx_t < idx_z < idx_p < idx_18


def test_ctf_level_17_mirror_matchups_reuse_level_17_config_and_prior_pools():
    """Mirror TvT / ZvZ / PvP after 17: level-17 budgets/rewards, pools from 14–16."""
    source_pools = {
        "terran": next(
            p["unit_pool"]
            for p in _load_unit_pools("ctf_level_14")
            if p["race"] == "terran"
        ),
        "zerg": next(
            p["unit_pool"]
            for p in _load_unit_pools("ctf_level_16")
            if p["race"] == "zerg"
        ),
        "protoss": next(
            p["unit_pool"]
            for p in _load_unit_pools("ctf_level_15")
            if p["race"] == "protoss"
        ),
    }
    env_17 = _load_env("ctf_level_17")
    for race in ("terran", "zerg", "protoss"):
        level = f"ctf_level_17_mirror_{race}"
        env = _load_env(level)
        assert env["rewards"] == env_17["rewards"], level
        assert env["squad_selection"] == env_17["squad_selection"], level
        assert env.get("alternate_trainee_race") is True, level
        assert env.get("include_race") is True, level
        pools = env["unit_pools"]
        assert len(pools) == 4, level
        for pool in pools:
            assert pool["race"] == race, level
            assert pool["unit_pool"] == source_pools[race], level
        a_attack = next(p for p in pools if p["region"] == "A" and p["high_level_command"] == "attack")
        a_defend = next(p for p in pools if p["region"] == "A" and p["high_level_command"] == "defend")
        assert a_attack["mineral_budget"] == 2000, level
        assert a_attack["vespene_budget"] == 2000, level
        assert a_defend["mineral_budget"] == 1500, level
        assert a_defend["vespene_budget"] == 1500, level


def test_ctf_level_15_adds_splash_layers_and_scalars():
    cm = ConfigManager(str(_CONFIG_DIR))
    prior_layers = set(cm.get_cumulative_active_layers("ctf_level_14"))
    at_15 = cm.get_cumulative_active_layers("ctf_level_15")
    added_layers = [layer for layer in at_15 if layer not in prior_layers]
    assert added_layers == ["allied_splash", "enemy_splash"]

    prior_tac = set(cm.get_cumulative_active_tactical_layers("ctf_level_14"))
    at_15_tac = cm.get_cumulative_active_tactical_layers("ctf_level_15")
    added_tac = [layer for layer in at_15_tac if layer not in prior_tac]
    assert added_tac == ["allied_splash", "enemy_splash"]

    prior_scalars = set(cm.get_cumulative_active_unit_scalars("ctf_level_14"))
    at_15_scalars = cm.get_cumulative_active_unit_scalars("ctf_level_15")
    added_scalars = [s for s in at_15_scalars if s not in prior_scalars]
    assert added_scalars == [
        "air_dps_norm",
        "has_splash",
        "onetime_unit",
        "is_friendly_fire",
        "is_temporary",
    ]

    # 16/17 inherit without re-listing
    assert "allied_splash" in cm.get_cumulative_active_layers("ctf_level_16")
    assert "onetime_unit" in cm.get_cumulative_active_unit_scalars("ctf_level_17")
    assert "is_friendly_fire" in cm.get_cumulative_active_unit_scalars("ctf_level_17")
    assert "is_temporary" in cm.get_cumulative_active_unit_scalars("ctf_level_17")
    env_16 = _load_env("ctf_level_16")
    assert "active_layers" not in env_16 or not env_16.get("active_layers")
    assert "active_unit_scalars" not in env_16 or not env_16.get("active_unit_scalars")


def test_ctf_levels_cover_all_three_races():
    cm = ConfigManager(str(_CONFIG_DIR))
    ctf_levels = [lvl for lvl in cm.levels if lvl.startswith("ctf_level_")]
    races_seen = set()
    for level in ctf_levels:
        for pool in _load_unit_pools(level):
            races_seen.add(pool["race"])
    assert races_seen == {"terran", "zerg", "protoss"}


def test_ctf_pipeline_inserts_two_levels_between_01_and_02():
    cm = ConfigManager(str(_CONFIG_DIR))

    idx_01 = cm.levels.index("ctf_level_01")
    idx_02 = cm.levels.index("ctf_level_02")
    idx_03 = cm.levels.index("ctf_level_03")

    inserted_1 = "ctf_level_02_melee_zealot_vs_roach"
    inserted_2 = "ctf_level_02_melee_helliontank_vs_hydralisk"

    assert inserted_1 in cm.levels
    assert inserted_2 in cm.levels

    idx_i1 = cm.levels.index(inserted_1)
    idx_i2 = cm.levels.index(inserted_2)

    assert idx_01 < idx_i1 < idx_i2 < idx_02
    # Training should still proceed into what used to be ctf_level_03.
    assert idx_02 < idx_03


def test_ctf_level_01_zergling_vs_marine_inherits_layers_without_adding():
    cm = ConfigManager(str(_CONFIG_DIR))
    at_01 = set(cm.get_cumulative_active_layers("ctf_level_01"))
    at_zvm = cm.get_cumulative_active_layers("ctf_level_01_zergling_vs_marine")
    assert at_zvm == sorted(at_01)


def test_ctf_level_01_adds_only_bio_light_layers():
    cm = ConfigManager(str(_CONFIG_DIR))
    prior = set(cm.get_cumulative_active_layers("level_attack_5"))
    at_01 = cm.get_cumulative_active_layers("ctf_level_01")
    added = [layer for layer in at_01 if layer not in prior]
    assert added == [
        "allied_unit_biological",
        "enemy_unit_biological",
        "allied_g_dps_light",
        "enemy_g_dps_light",
    ]
