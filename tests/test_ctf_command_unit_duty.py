"""Duty-cycle: after each command head, T/Z/P pools keep an enabling unit."""
from pathlib import Path

import yaml

from src.utils.config_manager import ConfigManager

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

_HEAL = "ctf_heal_queen_vs_queen"
_AOE = "aoe_high_templar_zealot_vs_marine_marauder"
_SNIPE = "snipe_ghost_marauder_vs_roach_hydra"
_FLYING = "scenario_queen_vs_battlecruiser"

_BIO = {"MARINE", "MARAUDER", "REAPER", "GHOST"}
_MECH = {
    "HELLION",
    "HELLIONTANK",
    "CYCLONE",
    "SIEGETANK",
    "THOR",
    "THORAP",
    "WIDOWMINE",
    "BANSHEE",
    "VIKINGFIGHTER",
    "VIKINGASSAULT",
    "LIBERATOR",
    "LIBERATORAG",
    "BATTLECRUISER",
    "RAVEN",
}
_AIR_FOR_VIPER = {
    "BATTLECRUISER",
    "CORRUPTOR",
    "MUTALISK",
    "VIPER",
    "BROODLORD",
    "BANSHEE",
    "VIKINGFIGHTER",
    "VIKINGASSAULT",
    "LIBERATOR",
    "LIBERATORAG",
    "RAVEN",
    "ORACLE",
    "PHOENIX",
    "VOIDRAY",
    "CARRIER",
    "TEMPEST",
    "MOTHERSHIP",
}


def _pools(level: str) -> list[dict]:
    return list(
        (yaml.safe_load((_CONFIG_DIR / f"{level}.yaml").read_text(encoding="utf-8"))
         .get("env")
         or {})
        .get("unit_pools")
        or []
    )


def test_healers_present_after_heal_intro_when_race_can_heal():
    cm = ConfigManager(str(_CONFIG_DIR))
    heal_i = cm.levels.index(_HEAL)
    fly_i = cm.levels.index(_FLYING)
    for i, level in enumerate(cm.levels):
        if i < heal_i:
            continue
        if not level.startswith(("ctf_", "scenario_", "aoe_", "snipe_")):
            continue
        for pool in _pools(level):
            race = pool["race"]
            units = set(pool.get("unit_pool") or [])
            if race == "zerg":
                assert "QUEEN" in units, level
            elif race == "terran":
                if units & _MECH:
                    assert "SCV" in units, level
                if i >= fly_i and units & _BIO:
                    assert "MEDIVAC" in units, level


def test_aoe_casters_present_after_aoe_intro():
    cm = ConfigManager(str(_CONFIG_DIR))
    aoe_i = cm.levels.index(_AOE)
    for i, level in enumerate(cm.levels):
        if i < aoe_i:
            continue
        if not level.startswith(("ctf_", "scenario_", "aoe_", "snipe_")):
            continue
        for pool in _pools(level):
            race = pool["race"]
            units = set(pool.get("unit_pool") or [])
            if race == "zerg":
                assert "INFESTOR" in units, level
                if "ROACH" in units:
                    assert "RAVAGER" in units, level
            elif race == "terran":
                assert "GHOST" in units, level
            elif race == "protoss":
                assert "HIGHTEMPLAR" in units, level


def test_viper_in_air_matchups_after_flying_and_aoe():
    cm = ConfigManager(str(_CONFIG_DIR))
    fly_i = cm.levels.index(_FLYING)
    for i, level in enumerate(cm.levels):
        if i < fly_i:
            continue
        if not level.startswith(("ctf_", "scenario_", "aoe_", "snipe_")):
            continue
        pools = _pools(level)
        all_units = {u for p in pools for u in (p.get("unit_pool") or [])}
        combat_air = all_units & _AIR_FOR_VIPER
        if not (combat_air - {"VIPER"}):
            # Ground matchup: Viper must not hitch a ride on Medivac-only air.
            for pool in pools:
                if pool["race"] == "zerg":
                    assert "VIPER" not in set(pool.get("unit_pool") or []), level
            continue
        for pool in pools:
            if pool["race"] == "zerg":
                assert "VIPER" in set(pool.get("unit_pool") or []), level


def test_medivac_not_before_flying_gate():
    cm = ConfigManager(str(_CONFIG_DIR))
    fly_i = cm.levels.index(_FLYING)
    for i, level in enumerate(cm.levels):
        if i >= fly_i:
            break
        if not level.startswith(("ctf_", "scenario_", "aoe_", "snipe_")):
            continue
        units = {u for p in _pools(level) for u in (p.get("unit_pool") or [])}
        assert "MEDIVAC" not in units, level
        assert "VIPER" not in units, level
