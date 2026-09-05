"""Cumulative heal_local / HPS / creep layers from ctf_heal_queen_vs_queen onward."""
from src.utils.config_manager import ConfigManager


def test_heal_intro_adds_heal_head_and_hps_creep_layers():
    cm = ConfigManager("config_universal")
    heads = cm.get_cumulative_action_heads("ctf_heal_queen_vs_queen")
    assert "heal_local" in heads
    assert heads.index("attack_local") < heads.index("heal_local")

    layers = cm.get_cumulative_active_layers("ctf_heal_queen_vs_queen")
    for name in ("allied_hps_field", "enemy_hps_field", "terrain_creep"):
        assert name in layers

    tactical = cm.get_cumulative_active_tactical_layers("ctf_heal_queen_vs_queen")
    for name in ("allied_hps_field", "enemy_hps_field", "terrain_creep"):
        assert name in tactical


def test_ctf_level_18_inherits_heal_without_relisting():
    cm = ConfigManager("config_universal")
    env = cm.get_level_config("ctf_level_18")["env"]
    assert not env.get("action_heads")
    assert not env.get("active_layers")
    assert not env.get("active_tactical_layers")
    assert not env.get("active_unit_scalars")

    heads = cm.get_cumulative_action_heads("ctf_level_18")
    assert "heal_local" in heads
    layers = cm.get_cumulative_active_layers("ctf_level_18")
    assert "allied_hps_field" in layers
    assert "terrain_creep" in layers


def test_ctf_level_19_and_20_inherit_heal_without_relisting():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_19", "ctf_level_20"):
        heads = cm.get_cumulative_action_heads(level)
        assert "heal_local" in heads
        layers = cm.get_cumulative_active_layers(level)
        assert "allied_hps_field" in layers
        assert "terrain_creep" in layers


def test_heal_intro_is_queen_vs_queen_no_race_alternate():
    cm = ConfigManager("config_universal")
    env = cm.get_level_config("ctf_heal_queen_vs_queen")["env"]
    pools = env["unit_pools"]
    assert env.get("alternate_trainee_race") is False
    assert env.get("damaged_units") is True
    assert env.get("rewards", {}).get("heal_cast_reward") == 6.0
    assert env.get("heal_click_cells") == 5
    assert env.get("loud_command_heads") == ["heal_local"]
    assert env.get("loud_command_mix") == 0.5
    assert env.get("loud_command_bias") == 0.5
    assert not (cm.get_level_config("ctf_level_18")["env"] or {}).get("loud_command_heads")
    assert len(pools) == 4
    for pool in pools:
        assert pool["race"] == "zerg"
        assert pool["unit_pool"] == ["QUEEN"]
    assert pools[0].get("agent") is True
    assert "agent" not in pools[1]
    assert pools[2].get("agent") is True
    assert "agent" not in pools[3]
    assert "damaged_units" not in (cm.get_level_config("ctf_level_18")["env"] or {})


def test_heal_thor_scv_intro_is_same_race_no_alternate():
    cm = ConfigManager("config_universal")
    env = cm.get_level_config("ctf_heal_thor_scv_vs_thor_scv")["env"]
    pools = env["unit_pools"]
    assert env.get("alternate_trainee_race") is False
    assert env.get("damaged_units") is True
    assert env.get("heal_click_cells") == 4
    assert env.get("loud_command_heads") == ["heal_local"]
    assert env.get("rewards", {}).get("heal_cast_reward") == 5.0
    assert not env.get("action_heads")
    assert not env.get("active_layers")
    assert "heal_local" in cm.get_cumulative_action_heads("ctf_heal_thor_scv_vs_thor_scv")
    assert len(pools) == 4
    for pool in pools:
        assert pool["race"] == "terran"
        assert pool["unit_pool"] == ["THOR", "SCV"]
    assert pools[0].get("agent") is True
    assert "agent" not in pools[1]
    assert pools[2].get("agent") is True
    assert "agent" not in pools[3]


def test_heal_queen_vs_battlecruiser_scv_is_mixed_race_alternate():
    cm = ConfigManager("config_universal")
    env = cm.get_level_config("ctf_heal_queen_vs_battlecruiser_scv")["env"]
    pools = env["unit_pools"]
    assert env.get("alternate_trainee_race") is True
    assert env.get("heal_click_cells") == 3.5
    assert env.get("loud_command_heads") == ["heal_local"]
    assert env.get("rewards", {}).get("heal_cast_reward") == 4.5
    assert not env.get("action_heads")
    assert "heal_local" in cm.get_cumulative_action_heads(
        "ctf_heal_queen_vs_battlecruiser_scv"
    )
    assert len(pools) == 4
    assert pools[0]["race"] == "zerg" and pools[0]["unit_pool"] == ["QUEEN"]
    assert pools[1]["race"] == "terran" and pools[1]["unit_pool"] == [
        "BATTLECRUISER",
        "SCV",
    ]
    assert pools[2]["race"] == "zerg" and pools[2]["unit_pool"] == ["QUEEN"]
    assert pools[3]["race"] == "terran" and pools[3]["unit_pool"] == [
        "BATTLECRUISER",
        "SCV",
    ]


def test_heal_intro_precedes_ctf_level_18():
    cm = ConfigManager("config_universal")
    queen = cm.levels.index("ctf_heal_queen_vs_queen")
    thor = cm.levels.index("ctf_heal_thor_scv_vs_thor_scv")
    qvbc = cm.levels.index("ctf_heal_queen_vs_battlecruiser_scv")
    lvl18 = cm.levels.index("ctf_level_18")
    assert queen < thor < qvbc < lvl18


def test_heal_click_cells_shrinks_then_defaults_to_one():
    """0.5 smaller each heal level; omit the key when the box is the 1×1 default."""
    from src.utils.unit_feature_helpers import heal_click_cells_from_conf

    cm = ConfigManager("config_universal")
    queen = cm.get_level_config("ctf_heal_queen_vs_queen")["env"]
    thor = cm.get_level_config("ctf_heal_thor_scv_vs_thor_scv")["env"]
    qvbc = cm.get_level_config("ctf_heal_queen_vs_battlecruiser_scv")["env"]
    lvl18 = cm.get_level_config("ctf_level_18")["env"]
    lvl19 = cm.get_level_config("ctf_level_19")["env"]
    lvl20 = cm.get_level_config("ctf_level_20")["env"]
    queen_roach = cm.get_level_config("scenario_queen_roach_vs_stalker_disruptor")["env"]
    lvl21 = cm.get_level_config("ctf_level_21")["env"]
    lvl22 = cm.get_level_config("ctf_level_22")["env"]
    assert queen.get("heal_click_cells") == 5
    assert thor.get("heal_click_cells") == 4
    assert qvbc.get("heal_click_cells") == 3.5
    assert lvl18.get("heal_click_cells") == 3
    assert lvl19.get("heal_click_cells") == 2.5
    assert lvl20.get("heal_click_cells") == 2
    assert queen_roach.get("heal_click_cells") == 1.5
    assert "heal_click_cells" not in lvl21
    assert "heal_click_cells" not in lvl22
    assert heal_click_cells_from_conf({"env": queen}) == 5.0
    assert heal_click_cells_from_conf({"env": thor}) == 4.0
    assert heal_click_cells_from_conf({"env": qvbc}) == 3.5
    assert heal_click_cells_from_conf({"env": lvl18}) == 3.0
    assert heal_click_cells_from_conf({"env": lvl19}) == 2.5
    assert heal_click_cells_from_conf({"env": lvl20}) == 2.0
    assert heal_click_cells_from_conf({"env": queen_roach}) == 1.5
    assert heal_click_cells_from_conf({"env": lvl21}) == 1.0
    assert heal_click_cells_from_conf({"env": lvl22}) == 1.0
    assert heal_click_cells_from_conf({}) == 1.0


def test_heal_cast_reward_anneals_to_two_at_cloak():
    """Intro 6.0; 0.5 smaller each heal level after Thor's first drop; 2.0 at cloak."""
    cm = ConfigManager("config_universal")
    expected = (
        ("ctf_heal_queen_vs_queen", 6.0),
        ("ctf_heal_thor_scv_vs_thor_scv", 5.0),
        ("ctf_heal_queen_vs_battlecruiser_scv", 4.5),
        ("ctf_level_18", 4.0),
        ("ctf_level_19", 3.5),
        ("ctf_level_20", 3.0),
        ("scenario_queen_roach_vs_stalker_disruptor", 2.5),
        ("ctf_level_21", 2.0),
    )
    for level, value in expected:
        env = cm.get_level_config(level)["env"]
        assert env.get("rewards", {}).get("heal_cast_reward") == value, level
    assert "heal_cast_reward" not in (
        cm.get_level_config("ctf_level_22")["env"].get("rewards") or {}
    )
