"""Cumulative heal_local / HPS / creep layers from ctf_level_18 onward."""
from src.utils.config_manager import ConfigManager


def test_ctf_level_18_adds_heal_head_and_hps_creep_layers():
    cm = ConfigManager("config_universal")
    heads = cm.get_cumulative_action_heads("ctf_level_18")
    assert "heal_local" in heads
    # Inherited combat heads remain before heal
    assert heads.index("attack_local") < heads.index("heal_local")

    layers = cm.get_cumulative_active_layers("ctf_level_18")
    for name in ("allied_hps_field", "enemy_hps_field", "terrain_creep"):
        assert name in layers

    tactical = cm.get_cumulative_active_tactical_layers("ctf_level_18")
    for name in ("allied_hps_field", "enemy_hps_field", "terrain_creep"):
        assert name in tactical


def test_ctf_level_19_and_20_inherit_heal_without_relisting():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_19", "ctf_level_20"):
        heads = cm.get_cumulative_action_heads(level)
        assert "heal_local" in heads
        layers = cm.get_cumulative_active_layers(level)
        assert "allied_hps_field" in layers
        assert "terrain_creep" in layers
