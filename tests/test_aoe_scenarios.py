"""AoE spellcaster scenario levels: same scenario config plus aoe_local and spellcaster."""
from pathlib import Path

import yaml

from src.utils.config_manager import ConfigManager

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"
_UNIT_DATA = Path(__file__).resolve().parents[1] / "data" / "unit_data.json"
_REFERENCE_SCENARIO = "scenario_colossus_zealot_vs_swarmhost_hydra"

_AOE_LEVELS = (
    "aoe_high_templar_zealot_vs_marine_marauder",
    "aoe_infestor_roach_vs_marine_marauder",
    "aoe_zergling_infestor_vs_stalker_templar",
    "aoe_infestor_ling_vs_infestor_ling",
    "aoe_zergling_roach_vs_high_templar_immortal",
    "aoe_ghost_marauder_vs_high_templar_sentry",
    "aoe_marine_ghost_vs_darktemplar_adept",
    "aoe_viper_hydra_vs_voidray_phoenix",
    "aoe_viper_corruptor_vs_mutalisk",
    "aoe_ravager_roach_vs_immortal_colossus",
)

_AOE_CASTERS = {
    "HIGHTEMPLAR",
    "INFESTOR",
    "GHOST",
    "VIPER",
    "RAVAGER",
}

_RACE_FROM_UNIT_DATA = {
    "Race.Terran": "terran",
    "Race.Zerg": "zerg",
    "Race.Protoss": "protoss",
}


def _load_level(level_name: str) -> dict:
    return yaml.safe_load((_CONFIG_DIR / f"{level_name}.yaml").read_text(encoding="utf-8"))


def _strip_pools_and_aoe(cfg: dict) -> dict:
    env = dict(cfg["env"])
    env.pop("unit_pools", None)
    env.pop("action_heads", None)
    env.pop("env_mode", None)
    env.pop("active_layers", None)
    env.pop("active_tactical_layers", None)
    env.pop("active_unit_scalars", None)
    return {k: v for k, v in cfg.items() if k != "env"} | {"env": env}


def test_aoe_scenarios_are_in_pipeline_after_air_vs_air_before_ctf_24():
    cm = ConfigManager(str(_CONFIG_DIR))
    idx_prev = cm.levels.index("scenario_battlecruiser_liberator_vs_corruptor_mutalisk")
    idx_next = cm.levels.index("ctf_level_24")
    for offset, level in enumerate(_AOE_LEVELS, start=1):
        assert cm.levels.index(level) == idx_prev + offset, level
    assert cm.levels.index(_AOE_LEVELS[-1]) == idx_next - 1


def test_aoe_scenarios_match_reference_except_pools_mode_and_heads():
    reference = _strip_pools_and_aoe(_load_level(_REFERENCE_SCENARIO))
    for level in _AOE_LEVELS:
        got = _strip_pools_and_aoe(_load_level(level))
        assert got == reference, level


def test_aoe_scenarios_use_spellcaster_mode_and_aoe_local():
    for level in _AOE_LEVELS:
        env = _load_level(level)["env"]
        modes = {m.strip() for m in str(env["env_mode"]).split(",")}
        assert modes == {"heal", "cloak", "upgrades", "spellcaster"}, level
        assert env.get("action_heads") == ["aoe_local"], level


def test_aoe_scenarios_use_two_races_four_pools_and_a_caster():
    for level in _AOE_LEVELS:
        pools = _load_level(level)["env"]["unit_pools"]
        assert len(pools) == 4, level
        races = {p["race"] for p in pools}
        assert len(races) in (1, 2), level
        assert {p["high_level_command"] for p in pools} == {"attack", "defend"}
        assert {p["region"] for p in pools} == {"A", "B"}
        all_units = {u for p in pools for u in p["unit_pool"]}
        assert all_units & _AOE_CASTERS, level
        if len(races) == 1:
            trainee = [p for p in pools if p.get("agent")]
            opponent = [p for p in pools if not p.get("agent")]
            assert len(trainee) == 2 and len(opponent) == 2, level


def test_aoe_unit_pools_match_unit_data_race():
    import json

    unit_db = json.loads(_UNIT_DATA.read_text(encoding="utf-8"))
    for level in _AOE_LEVELS:
        for pool in _load_level(level)["env"]["unit_pools"]:
            for unit_name in pool["unit_pool"]:
                assert unit_name in unit_db, f"{level} unknown unit {unit_name}"
                declared = _RACE_FROM_UNIT_DATA[unit_db[unit_name]["race"]]
                assert declared == pool["race"], f"{level} {unit_name}"


def test_aoe_local_accumulates_from_first_aoe_level_not_earlier_scenarios():
    cm = ConfigManager(str(_CONFIG_DIR))
    assert "aoe_local" not in cm.get_cumulative_action_heads(
        "scenario_battlecruiser_liberator_vs_corruptor_mutalisk"
    )
    first = _AOE_LEVELS[0]
    heads = cm.get_cumulative_action_heads(first)
    assert "aoe_local" in heads
    assert heads.index("attack_local") < heads.index("aoe_local")
    for later in (_AOE_LEVELS[-1], "ctf_level_24", "ctf_level_25", "ctf_level_26"):
        assert "aoe_local" in cm.get_cumulative_action_heads(later), later


def test_first_aoe_level_adds_energy_layers_and_can_aoe():
    cm = ConfigManager(str(_CONFIG_DIR))
    first = _AOE_LEVELS[0]
    env = _load_level(first)["env"]
    assert env.get("active_layers") == ["allied_energy", "enemy_energy"]
    assert env.get("active_tactical_layers") == ["allied_energy", "enemy_energy"]
    assert env.get("active_unit_scalars") == ["can_aoe"]

    prior = "scenario_battlecruiser_liberator_vs_corruptor_mutalisk"
    assert "allied_energy" not in cm.get_cumulative_active_layers(prior)
    assert "can_aoe" not in cm.get_cumulative_active_unit_scalars(prior)

    layers = cm.get_cumulative_active_layers(first)
    tactical = cm.get_cumulative_active_tactical_layers(first)
    scalars = cm.get_cumulative_active_unit_scalars(first)
    for name in ("allied_energy", "enemy_energy"):
        assert name in layers
        assert name in tactical
    assert "can_aoe" in scalars

    for later in (_AOE_LEVELS[-1], "ctf_level_24", "ctf_level_25", "ctf_level_26"):
        assert "allied_energy" in cm.get_cumulative_active_layers(later), later
        assert "enemy_energy" in cm.get_cumulative_active_tactical_layers(later), later
        assert "can_aoe" in cm.get_cumulative_active_unit_scalars(later), later


def test_later_aoe_levels_do_not_relist_energy_or_can_aoe():
    for level in _AOE_LEVELS[1:]:
        env = _load_level(level)["env"]
        assert not env.get("active_layers"), level
        assert not env.get("active_tactical_layers"), level
        assert not env.get("active_unit_scalars"), level
