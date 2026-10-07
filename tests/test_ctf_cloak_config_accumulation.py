"""Cumulative cloak/burrow config from ctf_level_21 onward."""
from src.utils.config_manager import ConfigManager


def test_ctf_level_21_adds_cloak_heads_layers_scalars():
    cm = ConfigManager("config_universal")
    cats = cm.get_cumulative_categorical_heads("ctf_level_21")
    assert cats == ["cloak_burrow", "uncloak_unburrow"] or (
        "cloak_burrow" in cats and "uncloak_unburrow" in cats
    )

    layers = cm.get_cumulative_active_layers("ctf_level_21")
    assert "allied_unit_cloak" in layers
    assert "enemy_unit_cloak" in layers
    assert "allied_detection" in layers
    assert "enemy_detection" in layers
    # Appended with cloak so detector discs never shift earlier channels.
    cloak_i = layers.index("allied_unit_cloak")
    assert layers[cloak_i : cloak_i + 4] == [
        "allied_unit_cloak",
        "enemy_unit_cloak",
        "allied_detection",
        "enemy_detection",
    ]

    tactical = cm.get_cumulative_active_tactical_layers("ctf_level_21")
    assert "allied_unit_cloak" in tactical
    assert "enemy_unit_cloak" in tactical
    assert "allied_detection" in tactical
    assert "enemy_detection" in tactical

    scalars = cm.get_cumulative_active_unit_scalars("ctf_level_21")
    assert "is_immobile" in scalars
    assert "can_cloak" in scalars

    env = cm.get_level_config("ctf_level_21")["env"]
    assert "loud_command_heads" not in env
    assert "loud_command_mix" not in env
    assert "loud_command_bias" not in env


def test_ctf_level_22_23_inherit_cloak_config():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_22", "ctf_level_23"):
        cats = cm.get_cumulative_categorical_heads(level)
        assert "cloak_burrow" in cats
        assert "uncloak_unburrow" in cats
        layers = cm.get_cumulative_active_layers(level)
        assert "allied_unit_cloak" in layers
        assert "allied_detection" in layers
        assert "enemy_detection" in layers
        tactical = cm.get_cumulative_active_tactical_layers(level)
        assert "allied_detection" in tactical
        scalars = cm.get_cumulative_active_unit_scalars(level)
        assert "is_immobile" in scalars
        assert "can_cloak" in scalars
        assert "loud_command_heads" not in (cm.get_level_config(level)["env"] or {})


def test_cloak_level_pools_exclude_oracle_and_infestor():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_21", "ctf_level_22", "ctf_level_23"):
        cfg = cm.get_level_config(level)
        pools = cfg.get("env", {}).get("unit_pools", [])
        all_units = []
        for pool in pools:
            all_units.extend(pool.get("unit_pool", []))
        assert "ORACLE" not in all_units
        assert "INFESTOR" not in all_units

    # P↔T (21): cloak/burrow extras
    p21 = [
        u
        for pool in ConfigManager("config_universal").get_level_config("ctf_level_21")["env"][
            "unit_pools"
        ]
        for u in pool["unit_pool"]
    ]
    assert "DARKTEMPLAR" in p21 and "MOTHERSHIP" in p21 and "WIDOWMINE" in p21
    assert "LURKERMP" not in p21

    # Z↔P (22)
    p22 = [
        u
        for pool in ConfigManager("config_universal").get_level_config("ctf_level_22")["env"][
            "unit_pools"
        ]
        for u in pool["unit_pool"]
    ]
    assert "LURKERMP" in p22 and "DARKTEMPLAR" in p22 and "MOTHERSHIP" in p22

    # Z↔T (23)
    p23 = [
        u
        for pool in ConfigManager("config_universal").get_level_config("ctf_level_23")["env"][
            "unit_pools"
        ]
        for u in pool["unit_pool"]
    ]
    assert "LURKERMP" in p23 and "WIDOWMINE" in p23
