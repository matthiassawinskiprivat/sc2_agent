"""Air-vs-air scenario levels share existing scenario config; only unit pools differ."""
from pathlib import Path

import yaml

from src.utils.config_manager import ConfigManager

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"
_UNIT_DATA = Path(__file__).resolve().parents[1] / "data" / "unit_data.json"
_REFERENCE_SCENARIO = "scenario_colossus_zealot_vs_swarmhost_hydra"

_AIR_VS_AIR_LEVELS = (
    "scenario_corruptor_mutalisk_vs_tempest_carrier",
    "scenario_banshee_viking_vs_voidray_carrier",
    "scenario_battlecruiser_viking_vs_corruptor_mutalisk",
    "scenario_tempest_voidray_vs_corruptor_mutalisk",
    "scenario_battlecruiser_liberator_vs_corruptor_mutalisk",
)

_AIR_UNITS = {
    "CORRUPTOR",
    "MUTALISK",
    "TEMPEST",
    "CARRIER",
    "LIBERATOR",
    "VIKINGFIGHTER",
    "VOIDRAY",
    "BATTLECRUISER",
    "BANSHEE",
}

# Weapon target: 1 = ground, 2 = air, 3 = any. Carriers fight via interceptors.
# Corruptors have air-only weapons; buildings are Caustic Spray (actuation).
_GROUND_WEAPON_TYPES = {1, 3}
_GROUND_VIA_SUMMON = {"CARRIER"}
_GROUND_VIA_CAUSTIC_SPRAY = {"CORRUPTOR"}

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


def test_air_vs_air_scenarios_are_in_pipeline_before_ctf_level_24():
    cm = ConfigManager(str(_CONFIG_DIR))
    idx_prev = cm.levels.index("scenario_thor_vs_voidray")
    idx_next = cm.levels.index("ctf_level_24")
    for offset, level in enumerate(_AIR_VS_AIR_LEVELS, start=1):
        assert cm.levels.index(level) == idx_prev + offset, level
    assert cm.levels.index(_AIR_VS_AIR_LEVELS[-1]) < idx_next


def test_air_vs_air_scenarios_match_reference_config_except_unit_pools():
    reference = _strip_unit_pools(_load_level(_REFERENCE_SCENARIO))
    for level in _AIR_VS_AIR_LEVELS:
        got = _strip_unit_pools(_load_level(level))
        assert got == reference, level


def test_air_vs_air_scenarios_use_two_races_and_four_pools():
    for level in _AIR_VS_AIR_LEVELS:
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
            units = frozenset(race_pools[0]["unit_pool"])
            assert units <= _AIR_UNITS, level
            assert len(units) == 2, level


def test_air_vs_air_unit_pools_match_unit_data_race():
    import json

    unit_db = json.loads(_UNIT_DATA.read_text(encoding="utf-8"))
    for level in _AIR_VS_AIR_LEVELS:
        for pool in _load_level(level)["env"]["unit_pools"]:
            for unit_name in pool["unit_pool"]:
                assert unit_name in unit_db, f"{level} unknown unit {unit_name}"
                declared = _RACE_FROM_UNIT_DATA[unit_db[unit_name]["race"]]
                assert declared == pool["race"], f"{level} {unit_name}"


def _can_attack_ground(unit_names, unit_db) -> bool:
    """True if the spawned pool can damage a pylon (ground structure)."""
    for name in unit_names:
        if name in _GROUND_VIA_SUMMON or name in _GROUND_VIA_CAUSTIC_SPRAY:
            return True
        for weapon in unit_db.get(name, {}).get("weapons") or []:
            if int(weapon.get("type", 0)) in _GROUND_WEAPON_TYPES:
                return True
    return False


def test_corruptor_counts_as_ground_attack_via_caustic_spray():
    import json

    unit_db = json.loads(_UNIT_DATA.read_text(encoding="utf-8"))
    assert _can_attack_ground(["CORRUPTOR"], unit_db) is True
    weapons = unit_db["CORRUPTOR"]["weapons"]
    assert all(int(w.get("type", 0)) == 2 for w in weapons)


def test_liberator_fighter_plus_viking_fighter_cannot_hit_pylons():
    import json

    unit_db = json.loads(_UNIT_DATA.read_text(encoding="utf-8"))
    assert _can_attack_ground(["LIBERATOR", "VIKINGFIGHTER"], unit_db) is False


def test_air_vs_air_each_race_can_attack_ground_pylons():
    import json

    unit_db = json.loads(_UNIT_DATA.read_text(encoding="utf-8"))
    for level in _AIR_VS_AIR_LEVELS:
        by_race = {}
        for pool in _load_level(level)["env"]["unit_pools"]:
            by_race.setdefault(pool["race"], set()).update(pool["unit_pool"])
        for race, units in by_race.items():
            assert _can_attack_ground(units, unit_db), (
                f"{level} {race} pool {sorted(units)} has no ground attack "
                "(cannot destroy pylons)"
            )
