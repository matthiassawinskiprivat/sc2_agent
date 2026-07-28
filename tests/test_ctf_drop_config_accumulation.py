"""Cumulative drop/transport config from ctf_level_24 onward."""
from src.utils.config_manager import ConfigManager


def test_ctf_level_24_adds_load_unload_transport():
    cm = ConfigManager("config_universal")
    heads = cm.get_cumulative_action_heads("ctf_level_24")
    assert "load_local" in heads
    assert "unload_local" in heads

    layers = cm.get_cumulative_active_layers("ctf_level_24")
    assert "transport_slots" in layers
    assert "passenger_density" in layers

    tactical = cm.get_cumulative_active_tactical_layers("ctf_level_24")
    assert "transport_slots" in tactical
    assert "passenger_density" in tactical

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


def test_ctf_level_25_26_inherit_drop_config():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_25", "ctf_level_26"):
        heads = cm.get_cumulative_action_heads(level)
        assert "load_local" in heads
        assert "unload_local" in heads
        layers = cm.get_cumulative_active_layers(level)
        assert "transport_slots" in layers
        scalars = cm.get_cumulative_active_unit_scalars(level)
        assert "cargo_size_norm" in scalars


def test_env_modes_heal_cloak_drop_retrofit():
    cm = ConfigManager("config_universal")
    assert cm.get_level_config("ctf_level_18")["env"]["env_mode"] == "heal"
    assert cm.get_level_config("ctf_level_21")["env"]["env_mode"] == "heal,cloak"
    assert cm.get_level_config("ctf_level_24")["env"]["env_mode"] == "heal,cloak,drop"


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
