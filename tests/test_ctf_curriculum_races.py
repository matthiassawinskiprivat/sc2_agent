"""Tests for CTF pool assignment and multi-race curriculum configs."""

from pathlib import Path

import yaml

from src.utils.config_manager import ConfigManager
from train.envs.ctf.pool_assignment import (
    get_trainee_race,
    is_pool_trainee,
    pool_player_id,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

_CTF_BOOTSTRAP_LEVELS = frozenset({"ctf_level_01", "ctf_level_01_zergling_vs_marine"})

_RACE_UNITS = {
    "terran": {
        "MARINE", "MARAUDER", "HELLION", "HELLIONTANK", "SIEGETANK", "VIKINGFIGHTER", "BANSHEE", "THOR",
    },
    "zerg": {
        "ZERGLING", "ROACH", "HYDRALISK", "MUTALISK", "ULTRALISK", "RAVAGER", "BANELING",
    },
    "protoss": {
        "ZEALOT", "STALKER", "ADEPT", "IMMORTAL", "PHOENIX", "VOIDRAY", "COLOSSUS",
    },
}


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


def test_ctf_levels_use_at_most_three_unit_types_per_pool():
    cm = ConfigManager(str(_CONFIG_DIR))
    ctf_levels = [lvl for lvl in cm.levels if lvl.startswith("ctf_level_")]
    for level in ctf_levels:
        for pool in _load_unit_pools(level):
            assert len(pool.get("unit_pool", [])) <= 3, level


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
