"""Tests for CTF pool assignment and multi-race curriculum configs."""

from pathlib import Path
import importlib.util

import yaml

from src.utils.config_manager import ConfigManager

# Load pool_assignment under a unique module name. Do NOT install stub
# ``train`` / ``train.envs`` packages into sys.modules — that poisons later
# imports (``train.envs.ctf.game_data``, CaptureTheFlagSingleEnv, …) for the
# rest of the pytest process.
_pool_path = Path(__file__).resolve().parents[1] / "train" / "envs" / "ctf" / "pool_assignment.py"
_spec = importlib.util.spec_from_file_location("_ctf_pool_assignment_under_test", _pool_path)
_pool_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pool_mod)
get_trainee_race = _pool_mod.get_trainee_race
is_pool_trainee = _pool_mod.is_pool_trainee
pool_player_id = _pool_mod.pool_player_id

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

# Phase-1 bootstrap / intro levels keep static ``agent:`` flags and
# ``alternate_trainee_race: false`` so spawn owners stay simple while the
# pointer / heal actions are introduced.
_CTF_BOOTSTRAP_LEVELS = frozenset(
    {
        "ctf_level_01",
        "ctf_level_01_zergling_vs_marine",
        "ctf_level_02_melee_zealot_vs_roach",
        "ctf_level_02_melee_helliontank_vs_hydralisk",
    }
)

_RACE_UNITS = {
    "terran": {
        "MARINE", "MARAUDER", "GHOST", "HELLION", "HELLIONTANK", "SIEGETANK", "VIKINGFIGHTER", "BANSHEE", "THOR",
        "THORAP", "CYCLONE", "REAPER", "BATTLECRUISER", "SCV", "MEDIVAC", "LIBERATOR", "WIDOWMINE",
    },
    "zerg": {
        "ZERGLING", "ROACH", "HYDRALISK", "MUTALISK", "ULTRALISK", "RAVAGER", "BANELING",
        "CORRUPTOR", "SWARMHOSTMP", "BROODLORD", "QUEEN", "INFESTOR", "LURKERMP", "VIPER",
        "OVERLORD", "OVERLORDTRANSPORT", "OVERSEER",
    },
    "protoss": {
        "ZEALOT", "STALKER", "ADEPT", "IMMORTAL", "PHOENIX", "VOIDRAY", "COLOSSUS", "ARCHON",
        "DISRUPTOR", "TEMPEST", "CARRIER", "HIGHTEMPLAR", "DARKTEMPLAR", "SENTRY", "ORACLE",
        "MOTHERSHIP", "WARPPRISM", "OBSERVER", "PROBE", "SHIELDBATTERY",
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
    assert pool_player_id(pools[0], pools, 2, alternate=True) == 1
    assert pool_player_id(pools[1], pools, 2, alternate=True) == 2
    assert is_pool_trainee(pools[0], pools, 1, alternate=True)
    assert not is_pool_trainee(pools[1], pools, 1, alternate=True)
    assert not is_pool_trainee(pools[0], pools, 2, alternate=True)
    assert is_pool_trainee(pools[1], pools, 2, alternate=True)


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
    # Same-race mirrors keep a fixed trainee race.
    mirror_levels = {lvl for lvl in ctf_levels if "_mirror_" in lvl}
    exempt = _CTF_BOOTSTRAP_LEVELS | mirror_levels
    for level in ctf_levels:
        env = _load_env(level)
        if level in exempt:
            assert env.get("alternate_trainee_race") is False, level
            continue
        assert env.get("alternate_trainee_race") is True, level


def test_ctf_include_race_off_only_on_level_01():
    """Only ctf_level_01 disables race so the universal→squad graft stays 138-wide.

    Every other CTF / scenario / AoE level in the pipeline must keep race on
    (explicit ``include_race: true`` or the code default).
    """
    from src.agent.squad_constants import include_race_enabled

    cm = ConfigManager(str(_CONFIG_DIR))
    for level in cm.levels:
        if not (
            level.startswith("ctf_")
            or level.startswith("scenario_")
            or level.startswith("aoe_")
        ):
            continue
        env = _load_env(level)
        enabled = include_race_enabled({"env": env})
        if level == "ctf_level_01":
            assert enabled is False, level
            assert env.get("include_race") is False, level
        else:
            assert enabled is True, level


def test_ctf_levels_define_race_per_pool():
    cm = ConfigManager(str(_CONFIG_DIR))
    ctf_levels = [lvl for lvl in cm.levels if lvl.startswith("ctf_level_")]
    for level in ctf_levels:
        pools = _load_unit_pools(level)
        for pool in pools:
            assert "race" in pool, f"{level} missing race on pool"
        if level in _CTF_BOOTSTRAP_LEVELS:
            assert any(p.get("agent") for p in pools), level
            assert _load_env(level).get("alternate_trainee_race") is False, level



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
    assert env["squad_selection"]["min_selected_units"] == 3


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


def test_thor_pools_include_both_payload_modes():
    """Explosive Payload (THOR, Javelin splash) and High Impact (THORAP) stay paired.

    Heal intro scenarios are exempt: they teach SCV repair on a single Thor mode,
    not the dual-payload combat curriculum.
    """
    exempt = {
        "ctf_heal_thor_scv_vs_thor_scv.yaml",
        "ctf_heal_queen_vs_battlecruiser_scv.yaml",
    }
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        if path.name in exempt:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for i, pool in enumerate((data.get("env") or {}).get("unit_pools") or []):
            units = set(pool.get("unit_pool") or [])
            assert ("THOR" in units) == ("THORAP" in units), (
                f"{path.name} pool {i} must list both THOR and THORAP"
            )


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
    """Phase-2: mirrors sit after level 17, before the upgrades block (27+).

    Heal intros moved to phase 1 (before level 18), so they are no longer between
    the mirrors and ctf_level_18.
    """
    cm = ConfigManager(str(_CONFIG_DIR))
    idx_17 = cm.levels.index("ctf_level_17")
    idx_t = cm.levels.index("ctf_level_17_mirror_terran")
    idx_z = cm.levels.index("ctf_level_17_mirror_zerg")
    idx_p = cm.levels.index("ctf_level_17_mirror_protoss")
    idx_heal = cm.levels.index("ctf_heal_queen_vs_queen")
    idx_18 = cm.levels.index("ctf_level_18")
    idx_27 = cm.levels.index("ctf_level_27")
    assert idx_heal < idx_18 < idx_17
    assert idx_17 < idx_t < idx_z < idx_p < idx_27


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
        # Same-race mirrors keep a fixed trainee race (no alternate).
        assert env.get("alternate_trainee_race") is False, level
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

    # Per-unit scalar catalog is declared once at ctf_level_01; level 15 does
    # not append splash-related scalars again.
    prior_scalars = cm.get_cumulative_active_unit_scalars("ctf_level_14")
    at_15_scalars = cm.get_cumulative_active_unit_scalars("ctf_level_15")
    assert at_15_scalars == prior_scalars
    for name in (
        "air_dps_norm",
        "has_splash",
        "onetime_unit",
        "is_friendly_fire",
        "is_temporary",
    ):
        assert name in at_15_scalars

    assert "allied_splash" in cm.get_cumulative_active_layers("ctf_level_16")
    assert "onetime_unit" in cm.get_cumulative_active_unit_scalars("ctf_level_17")
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
    """Melee intros sit in phase 1; detailed ctf_level_02 opens phase 2."""
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

    assert idx_01 < idx_i1 < idx_i2
    # Phase-1 heal / cloak / aoe / heal-revisit / snipe sit between the melee
    # intros and the phase-2 detailed-layer block that starts at ctf_level_02.
    # Transport is deferred to the pipeline end.
    assert idx_i2 < idx_02 < idx_03


def test_ctf_level_01_zergling_vs_marine_inherits_layers_without_adding():
    cm = ConfigManager(str(_CONFIG_DIR))
    at_01 = cm.get_cumulative_active_layers("ctf_level_01")
    at_zvm = cm.get_cumulative_active_layers("ctf_level_01_zergling_vs_marine")
    assert at_zvm == at_01


def test_ctf_level_01_adds_no_map_layers_phase1_defers_detail():
    """Phase-1 ctf_level_01 keeps the grafted attack-5 channel set unchanged.

    Biological / light / mechanical overlays land on ctf_level_02 when phase 2 begins.
    """
    cm = ConfigManager(str(_CONFIG_DIR))
    prior = cm.get_cumulative_active_layers("level_attack_5")
    at_01 = cm.get_cumulative_active_layers("ctf_level_01")
    assert at_01 == prior

    # End of phase 1 (last snipe intro; transport channels attach later at ctf_24).
    prior_phase2 = set(
        cm.get_cumulative_active_layers("snipe_battlecruiser_marine_vs_corruptor_hydra")
    )
    at_02 = cm.get_cumulative_active_layers("ctf_level_02")
    added = [layer for layer in at_02 if layer not in prior_phase2]
    assert added == [
        "allied_unit_biological",
        "enemy_unit_biological",
        "allied_g_dps_light",
        "enemy_g_dps_light",
        "allied_unit_mechanical",
    ]
