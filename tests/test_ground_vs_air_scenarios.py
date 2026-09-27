"""Ground-vs-air scenario levels share existing scenario config; only unit pools differ."""
from pathlib import Path

import yaml

from src.utils.config_manager import ConfigManager

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"
_UNIT_DATA = Path(__file__).resolve().parents[1] / "data" / "unit_data.json"
_REFERENCE_SCENARIO = "scenario_colossus_zealot_vs_swarmhost_hydra"

_GROUND_VS_AIR_LEVELS = (
    "scenario_hydra_vs_tempest",
    "scenario_marine_vs_mutalisk",
    "scenario_stalker_vs_mutalisk",
    "scenario_queen_vs_banshee",
    "scenario_thor_vs_voidray",
)

_GROUND_AA_UNITS = {
    "HYDRALISK",
    "MARINE",
    "STALKER",
    "QUEEN",
    "THOR",
    "THORAP",
}
_SPELLCASTER_UNITS = {"HIGHTEMPLAR", "GHOST", "INFESTOR"}

_AIR_UNITS = {
    "TEMPEST",
    "MUTALISK",
    "BANSHEE",
    "VOIDRAY",
}

_RACE_FROM_UNIT_DATA = {
    "Race.Terran": "terran",
    "Race.Zerg": "zerg",
    "Race.Protoss": "protoss",
}


def _load_level(level_name: str) -> dict:
    return yaml.safe_load((_CONFIG_DIR / f"{level_name}.yaml").read_text(encoding="utf-8"))


def _strip_unit_pools(cfg: dict) -> dict:
    env = dict(cfg["env"])
    env.pop("unit_pools", None)
    return {k: v for k, v in cfg.items() if k != "env"} | {"env": env}


def _compositions(pools: list[dict]) -> tuple[frozenset[str], frozenset[str]]:
    first_race = pools[0]["race"]
    a = frozenset()
    b = frozenset()
    for pool in pools:
        units = frozenset(pool["unit_pool"])
        if pool["race"] == first_race:
            a |= units
        else:
            b |= units
    return a, b


def test_ground_vs_air_scenarios_are_in_pipeline_after_existing_scenarios():
    cm = ConfigManager(str(_CONFIG_DIR))
    idx_prev = cm.levels.index("scenario_ravager_roach_vs_siegetank_marine")
    for level in _GROUND_VS_AIR_LEVELS:
        idx = cm.levels.index(level)
        assert idx > idx_prev, level
        rev = f"{level}_reversed"
        if rev in cm.levels:
            assert cm.levels.index(rev) == idx + 1, rev


def test_ground_vs_air_scenarios_match_reference_config_except_unit_pools():
    """Structural parity with the reference; steps / env_mode may diverge."""
    reference = _load_level(_REFERENCE_SCENARIO)
    for level in _GROUND_VS_AIR_LEVELS:
        got = _load_level(level)
        assert got["env"]["env_class"] == reference["env"]["env_class"], level
        assert got["env"]["maps_dir"] == reference["env"]["maps_dir"], level
        assert got["training"]["num_envs"] == reference["training"]["num_envs"], level


def test_ground_vs_air_scenarios_use_two_races_and_four_pools():
    for level in _GROUND_VS_AIR_LEVELS:
        pools = _load_level(level)["env"]["unit_pools"]
        assert len(pools) == 4, level
        races = {p["race"] for p in pools}
        assert len(races) == 2, level
        assert {p["high_level_command"] for p in pools} == {"attack", "defend"}
        assert {p["region"] for p in pools} == {"A", "B"}
        for race in races:
            race_pools = [p for p in pools if p["race"] == race]
            assert {frozenset(p["unit_pool"]) for p in race_pools} == {
                frozenset(race_pools[0]["unit_pool"])
            }, level


def test_ground_vs_air_scenarios_pair_ground_aa_with_air():
    for level in _GROUND_VS_AIR_LEVELS:
        ground, air = _compositions(_load_level(level)["env"]["unit_pools"])
        ground = ground - _SPELLCASTER_UNITS
        air = air - _SPELLCASTER_UNITS
        sides = {ground, air}
        assert any(side <= _GROUND_AA_UNITS for side in sides), level
        assert any(side <= _AIR_UNITS for side in sides), level
        assert not (ground <= _GROUND_AA_UNITS and air <= _GROUND_AA_UNITS), level
        assert not (ground <= _AIR_UNITS and air <= _AIR_UNITS), level


def test_ground_vs_air_unit_pools_match_unit_data_race():
    import json

    unit_db = json.loads(_UNIT_DATA.read_text(encoding="utf-8"))
    for level in _GROUND_VS_AIR_LEVELS:
        for pool in _load_level(level)["env"]["unit_pools"]:
            for unit_name in pool["unit_pool"]:
                assert unit_name in unit_db, f"{level} unknown unit {unit_name}"
                declared = _RACE_FROM_UNIT_DATA[unit_db[unit_name]["race"]]
                assert declared == pool["race"], f"{level} {unit_name}"
