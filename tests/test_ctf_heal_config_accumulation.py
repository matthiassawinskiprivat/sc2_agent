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


def test_damaged_units_only_enabled_on_first_three_heal_levels():
    """``damaged_units`` (the boolean pre-wound-the-whole-army training wheel)
    is only for the three dedicated heal-introduction scenarios. From
    ctf_level_18 onward (and every later heal-mode level/scenario), heal
    targets must come from real combat damage -- otherwise a guaranteed,
    risk-free wounded ally every episode lets the agent farm heal_reward
    forever without ever having to fight."""
    cm = ConfigManager("config_universal")
    for level in (
        "ctf_heal_queen_vs_queen",
        "ctf_heal_thor_scv_vs_thor_scv",
        "ctf_heal_queen_vs_battlecruiser_scv",
    ):
        assert cm.get_level_config(level)["env"].get("damaged_units") is True, level

    for level in (
        "ctf_level_18",
        "ctf_level_19",
        "ctf_level_20",
        "ctf_level_21",
        "ctf_level_22",
        "ctf_level_23",
        "ctf_level_24",
        "ctf_level_25",
        "ctf_level_26",
        "ctf_level_27",
        "ctf_level_28",
        "ctf_level_29",
        "scenario_queen_roach_vs_stalker_disruptor",
        "scenario_lurker_vs_thor",
        "scenario_dt_vs_marauder",
        "scenario_widowmine_vs_zergling",
        "scenario_ghost_marauder_vs_lurker_roach",
        "scenario_queen_vs_banshee",
        "scenario_banshee_viking_vs_voidray_carrier",
        "scenario_battlecruiser_viking_vs_corruptor_mutalisk",
        "scenario_tempest_voidray_vs_corruptor_mutalisk",
    ):
        assert "damaged_units" not in (cm.get_level_config(level)["env"] or {}), level


def test_heal_intro_is_queen_vs_queen_no_race_alternate():
    cm = ConfigManager("config_universal")
    env = cm.get_level_config("ctf_heal_queen_vs_queen")["env"]
    pools = env["unit_pools"]
    assert env.get("alternate_trainee_race") is False
    assert env.get("damaged_units") is True
    assert env.get("rewards", {}).get("heal_cast_reward") == 3.0
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
    assert env.get("rewards", {}).get("heal_cast_reward") == 2.5
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
    assert env.get("rewards", {}).get("heal_cast_reward") == 2.25
    assert not env.get("action_heads")
    assert "heal_local" in cm.get_cumulative_action_heads(
        "ctf_heal_queen_vs_battlecruiser_scv"
    )
    assert len(pools) == 4
    assert pools[0]["race"] == "zerg" and pools[0]["unit_pool"] == ["QUEEN", "CORRUPTOR"]
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


def test_heal_click_cells_shrinks_slowly_then_defaults_to_one():
    """Cast-rate telemetry showed cast rate crashing at the ctf_level_18 army/
    composition jump (12.7% -> 2.5%) largely independent of click-box size, so
    the anneal from ctf_level_18 onward was slowed down and spread across the
    rest of the heal curriculum (0.25 smaller every one-to-three levels)
    instead of cliffing from 1.5 straight to the 1.0 default right after
    scenario_queen_roach_vs_stalker_disruptor. The 1.0 default is still the
    intended final floor -- it's just reached many levels later now, and made
    explicit through scenario_tempest_voidray_vs_corruptor_mutalisk so later
    levels don't silently jump back up/down relative to their neighbors.
    """
    from src.utils.unit_feature_helpers import heal_click_cells_from_conf

    cm = ConfigManager("config_universal")
    names_and_expected = [
        ("ctf_heal_queen_vs_queen", 5.0),
        ("ctf_heal_thor_scv_vs_thor_scv", 4.0),
        ("ctf_heal_queen_vs_battlecruiser_scv", 3.5),
        ("ctf_level_18", 3.25),
        ("ctf_level_19", 3.0),
        ("ctf_level_20", 2.75),
        ("scenario_queen_roach_vs_stalker_disruptor", 2.5),
        ("ctf_level_21", 2.25),
        ("ctf_level_22", 2.25),
        ("ctf_level_23", 2.0),
        ("ctf_level_27", 2.0),
        ("ctf_level_28", 1.75),
        ("ctf_level_29", 1.75),
        ("scenario_lurker_vs_thor", 1.5),
        ("scenario_dt_vs_marauder", 1.5),
        ("scenario_widowmine_vs_zergling", 1.25),
        ("scenario_ghost_marauder_vs_lurker_roach", 1.25),
        ("scenario_queen_vs_banshee", 1.25),
        ("scenario_banshee_viking_vs_voidray_carrier", 1.0),
        ("scenario_battlecruiser_viking_vs_corruptor_mutalisk", 1.0),
        ("scenario_tempest_voidray_vs_corruptor_mutalisk", 1.0),
    ]

    prev_value = None
    for name, expected in names_and_expected:
        env = cm.get_level_config(name)["env"]
        assert env.get("heal_click_cells") == expected, name
        assert heal_click_cells_from_conf({"env": env}) == expected, name
        if prev_value is not None:
            assert expected <= prev_value, f"{name} must not increase heal_click_cells"
        prev_value = expected

    # Levels after the explicit schedule ends (e.g. the liberator/aoe/transport
    # levels) rely on the code default, which must equal the reached floor.
    liberator = cm.get_level_config(
        "scenario_battlecruiser_liberator_vs_corruptor_mutalisk"
    )["env"]
    assert "heal_click_cells" not in liberator
    assert heal_click_cells_from_conf({"env": liberator}) == 1.0
    assert heal_click_cells_from_conf({}) == 1.0


def test_heal_cast_reward_anneals_to_zero_by_cloak():
    """Intro 3.0 -> 2.25 through qvbc (0.25 smaller each level), then dropped
    further starting at ctf_level_18 (2.0 was still too high relative to the
    army/composition jump there) in 0.25 steps down to ctf_level_21, finally
    landing on the code default of 0.0 from ctf_level_22 onward. No more
    manual zero-then-spike outlier at scenario_queen_roach_vs_stalker_disruptor
    -- it now sits on the same declining line as its neighbors."""
    cm = ConfigManager("config_universal")
    expected = (
        ("ctf_heal_queen_vs_queen", 3.0),
        ("ctf_heal_thor_scv_vs_thor_scv", 2.5),
        ("ctf_heal_queen_vs_battlecruiser_scv", 2.25),
        ("ctf_level_18", 1.5),
        ("ctf_level_19", 1.25),
        ("ctf_level_20", 1.0),
        ("scenario_queen_roach_vs_stalker_disruptor", 0.75),
        ("ctf_level_21", 0.5),
    )
    prev_value = None
    for level, value in expected:
        env = cm.get_level_config(level)["env"]
        assert env.get("rewards", {}).get("heal_cast_reward") == value, level
        if prev_value is not None:
            assert value <= prev_value, f"{level} must not increase heal_cast_reward"
        prev_value = value
    assert "heal_cast_reward" not in (
        cm.get_level_config("ctf_level_22")["env"].get("rewards") or {}
    )


def test_heal_cast_reward_and_heal_reward_never_credit_protoss_unit_types():
    """Protoss has no controllable healer (Shield Battery auto-heals and is
    filtered out of squad selection entirely), so heal_cast_reward /
    heal_reward must be structurally unreachable for every Protoss unit type,
    regardless of curriculum reward tuning. _issue_heal_at_point only ever
    populates _heal_cast_tags / _heal_order_tags (the sets that gate both
    rewards in calculate_squad_step_reward) for MEDIVAC, SCV, and QUEEN --
    any other type_id (including every Protoss unit) returns None before
    self.do(...) or tag bookkeeping runs."""
    from unittest.mock import MagicMock

    from sc2.ids.unit_typeid import UnitTypeId

    from src.agent.ctf_bot import CaptureTheFlagBot

    protoss_types = [
        UnitTypeId.ZEALOT,
        UnitTypeId.STALKER,
        UnitTypeId.ADEPT,
        UnitTypeId.ARCHON,
        UnitTypeId.IMMORTAL,
        UnitTypeId.COLOSSUS,
        UnitTypeId.DISRUPTOR,
        UnitTypeId.HIGHTEMPLAR,
        UnitTypeId.DARKTEMPLAR,
        UnitTypeId.SENTRY,
        UnitTypeId.PHOENIX,
        UnitTypeId.VOIDRAY,
        UnitTypeId.CARRIER,
        UnitTypeId.TEMPEST,
        UnitTypeId.ORACLE,
        UnitTypeId.MOTHERSHIP,
        UnitTypeId.PROBE,
        UnitTypeId.OBSERVER,
        UnitTypeId.WARPPRISM,
        UnitTypeId.SHIELDBATTERY,
    ]

    bot = MagicMock(spec=CaptureTheFlagBot)
    bot._heal_cast_tags = set()
    bot._heal_order_tags = set()
    bot._resolve_heal_target_in_cell = MagicMock(return_value=MagicMock(tag=999))

    for type_id in protoss_types:
        unit = MagicMock()
        unit.type_id = type_id
        unit.tag = 111

        result = CaptureTheFlagBot._issue_heal_at_point(bot, unit, target=MagicMock())

        assert result is None, type_id
        bot.do.assert_not_called()
        assert 111 not in bot._heal_cast_tags, type_id
        assert 111 not in bot._heal_order_tags, type_id


def test_mineral_budget_ramps_gradually_from_level_18_through_26():
    """ctf_level_18 used to jump straight from ctf_level_17's 2000/1500
    (attack/defend mineral_budget) to a 2800/2100 plateau held flat for nine
    levels (18 through 26) before jumping again at ctf_level_27 (a separate
    "upgrades" milestone, left untouched here). That flat-then-jump shape
    compounded the ctf_level_18 army-size spike identified from telemetry
    (heal cast rate collapsing 12.7% -> 2.5%). Replaced with a gradual ramp
    that does not need to reach the old 2800/2100 ceiling by ctf_level_26.
    vespene_budget was never part of the original jump in this block (it
    stayed flat at 2000/1500 throughout), so it is intentionally untouched.
    """
    cm = ConfigManager("config_universal")
    expected = (
        ("ctf_level_17", 2000, 1500),
        ("ctf_level_18", 2100, 1600),
        ("ctf_level_19", 2200, 1650),
        ("ctf_level_20", 2300, 1700),
        ("ctf_level_21", 2350, 1750),
        ("ctf_level_22", 2400, 1800),
        ("ctf_level_23", 2450, 1850),
        ("ctf_level_24", 2500, 1900),
        ("ctf_level_25", 2550, 1950),
        ("ctf_level_26", 2600, 2000),
    )
    prev_attack = prev_defend = None
    for level, attack_val, defend_val in expected:
        pools = cm.get_level_config(level)["env"]["unit_pools"]
        attacks = {p["mineral_budget"] for p in pools if p["high_level_command"] == "attack"}
        defends = {p["mineral_budget"] for p in pools if p["high_level_command"] == "defend"}
        vespenes = {p["vespene_budget"] for p in pools}
        assert attacks == {attack_val}, level
        assert defends == {defend_val}, level
        assert vespenes == {2000, 1500}, level
        if prev_attack is not None:
            assert attack_val >= prev_attack, f"{level} must not decrease attack mineral_budget"
            assert defend_val >= prev_defend, f"{level} must not decrease defend mineral_budget"
        prev_attack, prev_defend = attack_val, defend_val

    # ctf_level_26 stays below the old flat plateau (2800/2100) -- the ramp is
    # not required to reach the value ctf_level_18 used to jump straight to.
    assert prev_attack < 2800
    assert prev_defend < 2100

    # ctf_level_27's separate "upgrades" milestone jump is untouched.
    pools_27 = cm.get_level_config("ctf_level_27")["env"]["unit_pools"]
    attacks_27 = {p["mineral_budget"] for p in pools_27 if p["high_level_command"] == "attack"}
    defends_27 = {p["mineral_budget"] for p in pools_27 if p["high_level_command"] == "defend"}
    assert attacks_27 == {3800}
    assert defends_27 == {3100}
