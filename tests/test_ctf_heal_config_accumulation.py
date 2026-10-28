"""Cumulative heal_unit / HPS / creep layers from the heal intro levels onward."""
from src.utils.config_manager import ConfigManager


def test_heal_intro_adds_heal_head_and_hps_creep_layers():
    cm = ConfigManager("config_universal")
    heads = cm.get_cumulative_action_heads("ctf_heal_queen_vs_queen")
    assert "heal_unit" in heads
    # Spatial attack remains earlier in the cumulative list; heal_unit is grafted
    # from attack_local at the universal→squad transition.
    assert "attack_local" in heads
    assert heads.index("attack_local") < heads.index("heal_unit")

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
    assert "heal_unit" in heads
    layers = cm.get_cumulative_active_layers("ctf_level_18")
    assert "allied_hps_field" in layers
    assert "terrain_creep" in layers


def test_ctf_level_19_and_20_inherit_heal_without_relisting():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_19", "ctf_level_20"):
        heads = cm.get_cumulative_action_heads(level)
        assert "heal_unit" in heads
        layers = cm.get_cumulative_active_layers(level)
        assert "allied_hps_field" in layers
        assert "terrain_creep" in layers


def test_damaged_units_only_enabled_on_heal_intro_revisit_and_snipe():
    """``damaged_units`` on heal intros, heal revisit, and the snipe block."""
    cm = ConfigManager("config_universal")
    damaged_on = [
        "ctf_heal_queen_vs_queen",
        "ctf_heal_thor_scv_vs_thor_scv",
        "ctf_heal_queen_vs_battlecruiser_scv",
        "scenario_queen_vs_battlecruiser",
        "snipe_ghost_marauder_vs_roach_hydra",
        "snipe_infestor_roach_vs_battlecruiser_scv",
        "snipe_high_templar_zealot_vs_ghost_medivac",
        "snipe_battlecruiser_marine_vs_corruptor_hydra",
    ]
    for level in list(damaged_on):
        rev = f"{level}_reversed"
        if rev in cm.levels:
            damaged_on.append(rev)
    for level in damaged_on:
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
    # Unit-pointer heal: no click-square width, patient named by tag.
    assert "heal_click_cells" not in env
    assert "loud_command_heads" not in env
    assert "loud_command_mix" not in env
    assert "loud_command_bias" not in env
    assert len(pools) == 4
    for pool in pools:
        assert pool["race"] == "zerg"
        assert "QUEEN" in pool["unit_pool"]
        # Units fielded before this intro stay in the pool.
        assert {"ZERGLING", "ROACH", "HYDRALISK"} <= set(pool["unit_pool"])
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
    assert "heal_click_cells" not in env
    assert "loud_command_heads" not in env
    assert not env.get("action_heads")
    assert not env.get("active_layers")
    assert "heal_unit" in cm.get_cumulative_action_heads("ctf_heal_thor_scv_vs_thor_scv")
    assert len(pools) == 4
    for pool in pools:
        assert pool["race"] == "terran"
        assert "THOR" in pool["unit_pool"]
        assert "SCV" in pool["unit_pool"]
    assert pools[0].get("agent") is True
    assert "agent" not in pools[1]
    assert pools[2].get("agent") is True
    assert "agent" not in pools[3]


def test_heal_queen_vs_battlecruiser_scv_is_mixed_race_alternate():
    cm = ConfigManager("config_universal")
    env = cm.get_level_config("ctf_heal_queen_vs_battlecruiser_scv")["env"]
    pools = env["unit_pools"]
    assert env.get("alternate_trainee_race") is True
    assert "heal_click_cells" not in env
    assert "loud_command_heads" not in env
    assert not env.get("action_heads")
    assert "heal_unit" in cm.get_cumulative_action_heads(
        "ctf_heal_queen_vs_battlecruiser_scv"
    )
    assert len(pools) == 4
    assert pools[0]["race"] == "zerg"
    assert {"QUEEN", "CORRUPTOR", "INFESTOR", "VIPER"} <= set(pools[0]["unit_pool"])
    assert pools[1]["race"] == "terran"
    assert {"BATTLECRUISER", "SCV", "GHOST", "MEDIVAC"} <= set(pools[1]["unit_pool"])
    assert pools[2]["race"] == "zerg"
    assert {"QUEEN", "INFESTOR", "VIPER"} <= set(pools[2]["unit_pool"])
    assert pools[3]["race"] == "terran"
    assert {"BATTLECRUISER", "SCV", "GHOST", "MEDIVAC"} <= set(pools[3]["unit_pool"])


def test_heal_intro_precedes_ctf_level_18():
    """Same-race heal intros sit before the mixed heal practice block."""
    cm = ConfigManager("config_universal")
    queen = cm.levels.index("ctf_heal_queen_vs_queen")
    thor = cm.levels.index("ctf_heal_thor_scv_vs_thor_scv")
    lvl18 = cm.levels.index("ctf_level_18")
    assert queen < thor < lvl18


def test_heal_queen_vs_battlecruiser_follows_flying_layers():
    """Air heal mix waits until flying / air-DPS map channels exist (heal revisit)."""
    cm = ConfigManager("config_universal")
    flying = cm.levels.index("scenario_queen_vs_battlecruiser")
    qvbc = cm.levels.index("ctf_heal_queen_vs_battlecruiser_scv")
    assert flying < qvbc
    layers = cm.get_cumulative_active_layers("ctf_heal_queen_vs_battlecruiser_scv")
    assert "allied_unit_flying" in layers
    assert "allied_a_dps_base" in layers


def test_phase1_pre_flying_pools_have_no_flyers_or_massives():
    """Before flying (heal revisit) / massive (ctf_07) map layers, pools stay ground-sized."""
    cm = ConfigManager("config_universal")
    flying = {
        "BATTLECRUISER",
        "CORRUPTOR",
        "MUTALISK",
        "VIPER",
        "BROODLORD",
        "OVERLORD",
        "OVERLORDTRANSPORT",
        "OVERSEER",
        "BANSHEE",
        "VIKINGFIGHTER",
        "VIKINGASSAULT",
        "LIBERATOR",
        "LIBERATORAG",
        "MEDIVAC",
        "RAVEN",
        "ORACLE",
        "PHOENIX",
        "VOIDRAY",
        "CARRIER",
        "TEMPEST",
        "MOTHERSHIP",
        "OBSERVER",
        "WARPPRISM",
    }
    massive = {
        "THOR",
        "THORAP",
        "ULTRALISK",
        "COLOSSUS",
        "ARCHON",
        "CARRIER",
        "TEMPEST",
        "MOTHERSHIP",
        "BATTLECRUISER",
        "BROODLORD",
    }
    fly_i = cm.levels.index("scenario_queen_vs_battlecruiser")
    mass_i = cm.levels.index("ctf_level_07")
    thor_i = cm.levels.index("ctf_heal_thor_scv_vs_thor_scv")
    # Massive overlay attaches at ctf_07. Thor is fielded at the heal intro and
    # then kept, because a unit added in a ctf_ level stays in later ctf_ pools.
    # Listed levels spawn a Battlecruiser before that overlay for heal/snipe.
    keep_massive = {
        "ctf_heal_thor_scv_vs_thor_scv",
        "scenario_queen_vs_battlecruiser",
        "snipe_infestor_roach_vs_battlecruiser_scv",
        "snipe_infestor_roach_vs_battlecruiser_scv_reversed",
        "snipe_battlecruiser_marine_vs_corruptor_hydra",
        "snipe_battlecruiser_marine_vs_corruptor_hydra_reversed",
    }
    for i, level in enumerate(cm.levels):
        if not (
            level.startswith("ctf_")
            or level.startswith("scenario_")
            or level.startswith("aoe_")
            or level.startswith("snipe_")
        ):
            continue
        if level == "ctf_heal_thor_scv_vs_thor_scv":
            continue
        pools = (cm.get_level_config(level).get("env") or {}).get("unit_pools") or []
        units = {u for pool in pools for u in pool.get("unit_pool") or []}
        if i < fly_i:
            assert not (units & flying), f"{level}: unexpected flyers {units & flying}"
        if i < mass_i and level not in keep_massive:
            allowed = {"THOR", "THORAP"} if i > thor_i else set()
            unexpected = (units & massive) - allowed
            assert not unexpected, f"{level}: unexpected massives {unexpected}"


def test_heal_unit_has_no_click_cells_anywhere():
    """Unit-pointer heal names its patient by tag; click-square width is gone."""
    cm = ConfigManager("config_universal")
    for level in cm.levels:
        if not (
            level.startswith("ctf_")
            or level.startswith("scenario_")
            or level.startswith("aoe_")
            or level.startswith("snipe_")
        ):
            continue
        env = cm.get_level_config(level).get("env") or {}
        assert "heal_click_cells" not in env, level


def test_heal_cast_reward_and_heal_reward_never_credit_protoss_unit_types():
    """Protoss has no controllable healer; heal bookkeeping must stay unreachable."""
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
    bot._HEAL_ABILITY_BY_TYPE = CaptureTheFlagBot._HEAL_ABILITY_BY_TYPE
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

    assert prev_attack < 2800
    assert prev_defend < 2100

    pools_27 = cm.get_level_config("ctf_level_27")["env"]["unit_pools"]
    assert all("mineral_budget" in p for p in pools_27)
