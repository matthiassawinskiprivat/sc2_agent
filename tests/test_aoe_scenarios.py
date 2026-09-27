"""AoE spellcaster scenario levels: same scenario config plus aoe_local and spellcaster."""
from pathlib import Path

import yaml

from src.utils.config_manager import ConfigManager

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"
_UNIT_DATA = Path(__file__).resolve().parents[1] / "data" / "unit_data.json"
_REFERENCE_SCENARIO = "scenario_colossus_zealot_vs_swarmhost_hydra"

_PHASE1_AOE = (
    "aoe_high_templar_zealot_vs_marine_marauder",
    "aoe_infestor_roach_vs_marine_marauder",
    "aoe_zergling_infestor_vs_stalker_templar",
    "aoe_infestor_ling_vs_infestor_ling",
    "aoe_zergling_roach_vs_high_templar_immortal",
)

_PHASE2_AOE = (
    "aoe_ghost_marauder_vs_high_templar_sentry",
    "aoe_marine_ghost_vs_darktemplar_adept",
    "aoe_viper_hydra_vs_voidray_phoenix",
    "aoe_viper_corruptor_vs_mutalisk",
    "aoe_ravager_roach_vs_immortal_colossus",
)

_AOE_LEVELS = _PHASE1_AOE + _PHASE2_AOE


def _expand_with_reversed(cm: ConfigManager, levels: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for level in levels:
        out.append(level)
        rev = f"{level}_reversed"
        if rev in cm.levels:
            out.append(rev)
    return out


def test_aoe_scenarios_are_split_across_curriculum_phases():
    """Phase-1 AoE intros sit before the heal revisit; phase-2 revisits precede transport."""
    cm = ConfigManager(str(_CONFIG_DIR))
    idx_cloak = cm.levels.index("ctf_level_23")
    idx_heal_revisit = cm.levels.index("scenario_queen_vs_battlecruiser")
    phase1_block = _expand_with_reversed(cm, _PHASE1_AOE)
    for offset, level in enumerate(phase1_block, start=1):
        assert cm.levels.index(level) == idx_cloak + offset, level
    assert cm.levels.index(phase1_block[-1]) == idx_heal_revisit - 1

    # Liberator may have a `_reversed` sibling before phase-2 AoE starts.
    idx_liberator = cm.levels.index("scenario_battlecruiser_liberator_vs_corruptor_mutalisk")
    phase2_start = idx_liberator + 1
    if "scenario_battlecruiser_liberator_vs_corruptor_mutalisk_reversed" in cm.levels:
        phase2_start += 1
    phase2_block = _expand_with_reversed(cm, _PHASE2_AOE)
    for offset, level in enumerate(phase2_block):
        assert cm.levels.index(level) == phase2_start + offset, level
    # Pipeline ends with transport after phase-2 AoE revisits.
    assert cm.levels.index(phase2_block[-1]) == cm.levels.index("ctf_level_24") - 1
    assert cm.levels[-1] == "ctf_level_26"

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


def test_aoe_scenarios_match_reference_except_pools_mode_and_heads():
    """Training hyperparams may diverge; keep structural env_class / maps_dir parity."""
    reference = _load_level(_REFERENCE_SCENARIO)
    for level in _AOE_LEVELS:
        got = _load_level(level)
        assert got["env"]["env_class"] == reference["env"]["env_class"], level
        assert got["env"]["maps_dir"] == reference["env"]["maps_dir"], level
        assert got["training"]["num_envs"] == reference["training"]["num_envs"], level


def test_aoe_scenarios_use_spellcaster_mode_and_aoe_local():
    """Phase-1 AoE: heal+cloak+spellcaster. Phase-2: also upgrades (drop is pipeline-end)."""
    phase1_modes = {"heal", "cloak", "spellcaster"}
    phase2_modes = {"heal", "cloak", "spellcaster", "upgrades"}
    for level in _PHASE1_AOE:
        env = _load_level(level)["env"]
        modes = {m.strip() for m in str(env["env_mode"]).split(",") if m.strip()}
        assert modes == phase1_modes, level
        assert env.get("action_heads") == ["aoe_local"], level
    for level in _PHASE2_AOE:
        env = _load_level(level)["env"]
        modes = {m.strip() for m in str(env["env_mode"]).split(",") if m.strip()}
        assert modes == phase2_modes, level
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
    # Phase-1 action curriculum: cloak practice precedes the first AoE intro.
    assert "aoe_local" not in cm.get_cumulative_action_heads("ctf_level_23")
    first = _PHASE1_AOE[0]
    heads = cm.get_cumulative_action_heads(first)
    assert "aoe_local" in heads
    assert heads.index("attack_local") < heads.index("aoe_local")
    for later in (_PHASE1_AOE[-1], "ctf_level_24", "ctf_level_25", "ctf_level_26"):
        assert "aoe_local" in cm.get_cumulative_action_heads(later), later
    # Phase-2 air scenarios sit after phase-1 AoE, so they inherit aoe_local.
    assert "aoe_local" in cm.get_cumulative_action_heads(
        "scenario_battlecruiser_liberator_vs_corruptor_mutalisk"
    )


def test_first_aoe_level_adds_energy_layers_and_can_aoe():
    cm = ConfigManager(str(_CONFIG_DIR))
    first = _PHASE1_AOE[0]
    env = _load_level(first)["env"]
    assert env.get("active_layers") == ["allied_energy", "enemy_energy"]
    assert env.get("active_tactical_layers") == ["allied_energy", "enemy_energy"]
    # can_aoe lives in the ctf_level_01 scalar catalog; AoE levels do not re-list it.
    assert not env.get("active_unit_scalars")

    prior = "ctf_level_23"
    assert "allied_energy" not in cm.get_cumulative_active_layers(prior)
    assert "can_aoe" in cm.get_cumulative_active_unit_scalars(prior)

    layers = cm.get_cumulative_active_layers(first)
    tactical = cm.get_cumulative_active_tactical_layers(first)
    scalars = cm.get_cumulative_active_unit_scalars(first)
    for name in ("allied_energy", "enemy_energy"):
        assert name in layers
        assert name in tactical
    assert "can_aoe" in scalars

    for later in (_PHASE1_AOE[-1], "ctf_level_24", "ctf_level_25", "ctf_level_26"):
        assert "allied_energy" in cm.get_cumulative_active_layers(later), later
        assert "enemy_energy" in cm.get_cumulative_active_tactical_layers(later), later
        assert "can_aoe" in cm.get_cumulative_active_unit_scalars(later), later


def test_later_aoe_levels_do_not_relist_energy_or_can_aoe():
    for level in _PHASE1_AOE[1:] + _PHASE2_AOE:
        env = _load_level(level)["env"]
        assert not env.get("active_layers"), level
        assert not env.get("active_tactical_layers"), level
        assert not env.get("active_unit_scalars"), level
