"""Cumulative drop/transport config from ctf_level_24 onward."""
from src.utils.config_manager import ConfigManager


def test_ctf_level_24_adds_load_unload_transport():
    cm = ConfigManager("config_universal")
    heads = cm.get_cumulative_action_heads("ctf_level_24")
    assert "load_unit" in heads
    assert "unload_local" in heads

    layers = cm.get_cumulative_active_layers("ctf_level_24")
    assert "transport_slots" in layers
    assert "passenger_density" in layers
    # Flying / air-DPS-base attach at scenario_queen_vs_battlecruiser (pre-transport).
    assert "allied_unit_flying" in layers
    assert "enemy_unit_flying" in layers
    assert "allied_a_dps_base" in layers
    assert "enemy_a_dps_base" in layers

    tactical = cm.get_cumulative_active_tactical_layers("ctf_level_24")
    assert "transport_slots" in tactical
    assert "passenger_density" in tactical
    assert "allied_unit_flying" in tactical
    assert "allied_a_dps_base" in tactical

    scalars = cm.get_cumulative_active_unit_scalars("ctf_level_24")
    for name in (
        "is_transport",
        "cargo_used_norm",
        "cargo_free_norm",
        "cargo_size_norm",
        "can_load",
        "can_unload",
    ):
        assert name in scalars


def test_flying_channels_attach_at_queen_vs_battlecruiser_heal_revisit():
    cm = ConfigManager("config_universal")
    prior = set(cm.get_cumulative_active_layers("aoe_zergling_roach_vs_high_templar_immortal"))
    at = cm.get_cumulative_active_layers("scenario_queen_vs_battlecruiser")
    added = [layer for layer in at if layer not in prior]
    assert added == [
        "allied_unit_flying",
        "enemy_unit_flying",
        "allied_a_dps_base",
        "enemy_a_dps_base",
    ]
    assert cm.get_level_config("scenario_queen_vs_battlecruiser")["env"].get(
        "damaged_units"
    ) is True


def test_ctf_level_25_26_inherit_drop_config():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_25", "ctf_level_26"):
        heads = cm.get_cumulative_action_heads(level)
        assert "load_unit" in heads
        assert "unload_local" in heads
        layers = cm.get_cumulative_active_layers(level)
        assert "transport_slots" in layers
        scalars = cm.get_cumulative_active_unit_scalars(level)
        assert "cargo_size_norm" in scalars


def test_env_modes_heal_cloak_drop_retrofit():
    cm = ConfigManager("config_universal")
    assert cm.get_level_config("ctf_heal_queen_vs_queen")["env"]["env_mode"] == "heal"
    assert cm.get_level_config("ctf_heal_thor_scv_vs_thor_scv")["env"]["env_mode"] == "heal"
    assert (
        cm.get_level_config("ctf_heal_queen_vs_battlecruiser_scv")["env"]["env_mode"]
        == "heal,cloak,spellcaster"
    )
    assert cm.get_level_config("ctf_level_18")["env"]["env_mode"] == "heal"
    assert cm.get_level_config("ctf_level_21")["env"]["env_mode"] == "heal,cloak"
    # Drop is deferred to pipeline-end transport intros.
    assert (
        cm.get_level_config("ctf_level_24")["env"]["env_mode"]
        == "heal,cloak,spellcaster,upgrades,drop"
    )


def test_env_modes_accumulate_along_pipeline():
    """Once a mode is introduced, every later CTF/scenario/AoE/snipe level keeps it."""
    cm = ConfigManager("config_universal")
    # Drop is last (transport deferred to pipeline end); upgrades precede it.
    intro = {
        "heal": "ctf_heal_queen_vs_queen",
        "cloak": "ctf_level_21",
        "spellcaster": "aoe_high_templar_zealot_vs_marine_marauder",
        "upgrades": "ctf_level_27",
        "drop": "ctf_level_24",
    }
    order = ("heal", "cloak", "spellcaster", "upgrades", "drop")
    intro_idx = {m: cm.levels.index(lvl) for m, lvl in intro.items()}
    for i, level in enumerate(cm.levels):
        if not (
            level.startswith("ctf_")
            or level.startswith("scenario_")
            or level.startswith("aoe_")
            or level.startswith("snipe_")
        ):
            continue
        raw = cm.get_level_config(level)["env"].get("env_mode") or ""
        modes = {p.strip() for p in str(raw).split(",") if p.strip()}
        expected = {m for m in order if i >= intro_idx[m]}
        assert modes == expected, f"{level}: got {sorted(modes)} want {sorted(expected)}"


def test_transport_levels_are_at_pipeline_end():
    cm = ConfigManager("config_universal")
    assert cm.levels[-1] == "ctf_level_26"
    idx_24 = cm.levels.index("ctf_level_24")
    idx_25 = cm.levels.index("ctf_level_25")
    idx_26 = cm.levels.index("ctf_level_26")
    assert idx_24 < idx_25 < idx_26
    # Ability-tech conflict on 25 inserts ``ctf_level_25_reversed`` before 26.
    transport_tail = cm.levels[idx_24:]
    assert transport_tail[0] == "ctf_level_24"
    assert "ctf_level_25" in transport_tail
    assert transport_tail[-1] == "ctf_level_26"
    if "ctf_level_25_reversed" in cm.levels:
        assert cm.levels.index("ctf_level_25_reversed") == idx_25 + 1


def test_drop_pools_include_transports():
    cm = ConfigManager("config_universal")
    p24 = [
        u
        for pool in cm.get_level_config("ctf_level_24")["env"]["unit_pools"]
        for u in pool["unit_pool"]
    ]
    assert "WARPPRISM" in p24
    assert "MEDIVAC" in p24
    assert "OVERLORDTRANSPORT" not in p24  # P↔T only

    p25 = [
        u
        for pool in cm.get_level_config("ctf_level_25")["env"]["unit_pools"]
        for u in pool["unit_pool"]
    ]
    assert "OVERLORDTRANSPORT" in p25
    assert "WARPPRISM" in p25
