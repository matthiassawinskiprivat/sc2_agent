"""Zerg creep tumor + Protoss shield battery spawn rules for CTF heal curriculum."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from train.envs.ctf.heal_spawn import guaranteed_healer_name


def _make_env(unit_pools, env_mode="heal"):
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.unit_pools_config = unit_pools
    conf = {"env": {}}
    if env_mode is not None:
        conf["env"]["env_mode"] = env_mode
        conf["env_mode"] = env_mode
    env.training_conf = conf
    env.bot = MagicMock()
    env.bot.client = MagicMock()
    env.bot.client.debug_create_unit = AsyncMock()
    env.bot.client.step = AsyncMock()
    env._refresh_bot_state = AsyncMock()
    env._pool_player_id = lambda pool: 1 if pool.get("agent") else 2
    env._load_region_pairs = MagicMock(
        return_value=[
            {
                "a": MagicMock(center=Point2((10.0, 10.0))),
                "d": MagicMock(center=Point2((20.0, 20.0))),
            }
        ]
    )
    env._assign_region_pairs = MagicMock(return_value={"A": 0})
    env._calculate_spawns_for_pool = MagicMock(return_value=["ZERGLING"])
    env._resolve_map = MagicMock(return_value=MagicMock(path="map.SC2Map"))
    env.map_name = "map"
    return env


def _all_create_cmds(env):
    out = []
    for call in env.bot.client.debug_create_unit.await_args_list:
        out.extend(call.args[0])
    return out


def _spawn(env):
    def get_valid_pos(p):
        return p

    def get_offset_pos(base, index, total):
        return Point2((base.x + 1.0 + float(index), base.y + 1.0))

    def to_id(name):
        return getattr(UnitTypeId, name)

    return asyncio.run(env._spawn_from_unit_pools(get_valid_pos, get_offset_pos, to_id))


def test_heal_army_spawn_does_not_plant_tumors():
    """Creep coverage is a separate playable-area grid, not 1–2 tumors at spawn."""
    pools = [
        {
            "high_level_command": "attack",
            "race": "zerg",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZERGLING", "QUEEN"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
        {
            "high_level_command": "defend",
            "race": "terran",
            "region": "A",
            "unit_pool": ["MARINE"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
    ]
    env = _make_env(pools)
    _spawn(env)

    cmds = _all_create_cmds(env)
    tumor_types = {
        UnitTypeId.CREEPTUMOR,
        UnitTypeId.CREEPTUMORBURROWED,
        UnitTypeId.CREEPTUMORQUEEN,
    }
    assert not any(c[0] in tumor_types for c in cmds)


def test_protoss_defend_defers_budget_batteries():
    """Budget-selected batteries are deferred (need pylon power), not in the first batch."""
    pools = [
        {
            "high_level_command": "defend",
            "race": "protoss",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZEALOT", "SHIELDBATTERY"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
        {
            "high_level_command": "attack",
            "race": "terran",
            "region": "A",
            "unit_pool": ["MARINE"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
    ]
    env = _make_env(pools)
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["ZEALOT", "SHIELDBATTERY", "SHIELDBATTERY"]
            if pool["race"] == "protoss"
            else ["MARINE"]
        )
    )
    centers, _, deferred, _detectors = _spawn(env)

    army_cmds = _all_create_cmds(env)
    assert not any(c[0] == UnitTypeId.SHIELDBATTERY for c in army_cmds)
    assert any(c[0] == UnitTypeId.PYLON for c in army_cmds)
    assert any(c[0] == UnitTypeId.ZEALOT for c in army_cmds)
    assert len(deferred) == 2
    assert all(c[0] == UnitTypeId.SHIELDBATTERY for c in deferred)
    assert centers[0][1] == "defend"


def test_protoss_defend_guarantees_battery_when_rng_misses():
    """Heal + defend Protoss with SHIELDBATTERY in pool always gets ≥1 battery."""
    pools = [
        {
            "high_level_command": "defend",
            "race": "protoss",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZEALOT", "SHIELDBATTERY"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
        {
            "high_level_command": "attack",
            "race": "terran",
            "region": "A",
            "unit_pool": ["MARINE"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
    ]
    env = _make_env(pools, env_mode="heal")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["ZEALOT"] if pool["race"] == "protoss" else ["MARINE"]
        )
    )
    _, _, deferred, _detectors = _spawn(env)
    assert len(deferred) == 1
    assert deferred[0][0] == UnitTypeId.SHIELDBATTERY


def test_without_heal_mode_no_battery_guarantee():
    pools = [
        {
            "high_level_command": "defend",
            "race": "protoss",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZEALOT", "SHIELDBATTERY"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
        {
            "high_level_command": "attack",
            "race": "terran",
            "region": "A",
            "unit_pool": ["MARINE"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
    ]
    env = _make_env(pools, env_mode="cloak")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["ZEALOT"] if pool["race"] == "protoss" else ["MARINE"]
        )
    )
    _, _, deferred, detectors = _spawn(env)
    assert deferred == []
    # Cloak mode still defers race detectors even when battery guarantee is off.
    assert any(c[0] == UnitTypeId.OBSERVER for c in detectors)
    assert any(c[0] == UnitTypeId.RAVEN for c in detectors)


def test_guaranteed_healer_name_prefers_medivac_over_scv():
    assert guaranteed_healer_name(["MARINE", "SCV", "MEDIVAC"], ["MARINE"]) == "MEDIVAC"


def test_guaranteed_healer_name_prefers_queen():
    assert guaranteed_healer_name(["ZERGLING", "QUEEN"], ["ZERGLING"]) == "QUEEN"


def test_guaranteed_healer_name_none_when_already_present():
    assert guaranteed_healer_name(["SCV", "MEDIVAC"], ["SCV"]) is None


def test_guaranteed_healer_name_none_when_pool_has_none():
    assert guaranteed_healer_name(["ZEALOT", "STALKER"], ["ZEALOT"]) is None


def test_heal_mode_guarantees_healer_when_rng_misses():
    pools = [
        {
            "high_level_command": "attack",
            "race": "terran",
            "region": "A",
            "agent": True,
            "unit_pool": ["MARINE", "SCV", "MEDIVAC"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
        {
            "high_level_command": "defend",
            "race": "zerg",
            "region": "A",
            "unit_pool": ["ZERGLING"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
    ]
    env = _make_env(pools, env_mode="heal")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["MARINE"] if pool["race"] == "terran" else ["ZERGLING"]
        )
    )
    _spawn(env)
    cmds = _all_create_cmds(env)
    assert any(c[0] == UnitTypeId.MEDIVAC for c in cmds)
    assert any(c[0] == UnitTypeId.MARINE for c in cmds)


def test_heal_mode_keeps_budget_healer_without_extra():
    pools = [
        {
            "high_level_command": "attack",
            "race": "terran",
            "region": "A",
            "agent": True,
            "unit_pool": ["MARINE", "SCV", "MEDIVAC"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
        {
            "high_level_command": "defend",
            "race": "zerg",
            "region": "A",
            "unit_pool": ["ZERGLING"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
    ]
    env = _make_env(pools, env_mode="heal")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["MARINE", "SCV"] if pool["race"] == "terran" else ["ZERGLING"]
        )
    )
    _spawn(env)
    cmds = _all_create_cmds(env)
    medivacs = [c for c in cmds if c[0] == UnitTypeId.MEDIVAC]
    scvs = [c for c in cmds if c[0] == UnitTypeId.SCV]
    assert len(scvs) == 1
    assert medivacs == []


def test_without_heal_mode_no_healer_guarantee():
    pools = [
        {
            "high_level_command": "attack",
            "race": "terran",
            "region": "A",
            "agent": True,
            "unit_pool": ["MARINE", "SCV", "MEDIVAC"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
        {
            "high_level_command": "defend",
            "race": "zerg",
            "region": "A",
            "unit_pool": ["ZERGLING"],
            "mineral_budget": 500,
            "vespene_budget": 0,
        },
    ]
    env = _make_env(pools, env_mode="cloak")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["MARINE"] if pool["race"] == "terran" else ["ZERGLING"]
        )
    )
    _spawn(env)
    cmds = _all_create_cmds(env)
    assert not any(c[0] == UnitTypeId.MEDIVAC for c in cmds)
    assert not any(c[0] == UnitTypeId.SCV for c in cmds)


def test_damaged_units_enabled_defaults_false():
    from train.envs.ctf.heal_spawn import damaged_units_enabled

    assert damaged_units_enabled({}) is False
    assert damaged_units_enabled({"env": {}}) is False
    assert damaged_units_enabled({"env": {"damaged_units": False}}) is False
    assert damaged_units_enabled({"env": {"damaged_units": True}}) is True
    assert damaged_units_enabled({"damaged_units": True}) is True


def test_damaged_unit_targets_includes_thor_and_scv():
    from train.envs.ctf.heal_spawn import damaged_unit_targets

    thor = MagicMock()
    thor.tag = 1
    thor.type_id = UnitTypeId.THOR
    thor.is_structure = False
    thor.health_max = 400.0
    thor.shield_max = 0.0
    scv = MagicMock()
    scv.tag = 2
    scv.type_id = UnitTypeId.SCV
    scv.is_structure = False
    scv.health_max = 45.0
    scv.shield_max = 0.0
    bot = MagicMock()
    bot.all_units = [thor, scv]

    health_groups, shield_groups = damaged_unit_targets(bot, fraction=0.5)
    assert health_groups == {200.0: [1], 22.5: [2]}
    assert shield_groups == {}


def test_apply_damaged_units_halves_mobile_hp_skips_pylon():
    from train.envs.ctf.heal_spawn import apply_damaged_units

    bc = MagicMock()
    bc.tag = 1
    bc.type_id = UnitTypeId.BATTLECRUISER
    bc.is_structure = False
    bc.health_max = 550.0
    bc.shield_max = 0.0
    pylon = MagicMock()
    pylon.tag = 2
    pylon.type_id = UnitTypeId.PYLON
    pylon.is_structure = True
    pylon.health_max = 200.0
    pylon.shield_max = 200.0
    bot = MagicMock()
    bot.all_units = [bc, pylon]
    bot.client.debug_set_unit_value = AsyncMock()

    n = asyncio.run(apply_damaged_units(bot))
    assert n == 1
    bot.client.debug_set_unit_value.assert_awaited_once()
    tags, unit_value, value = bot.client.debug_set_unit_value.await_args.args
    assert tags == [1]
    assert unit_value == 2
    assert value == 275.0


def test_apply_damaged_units_skipped_when_flag_off():
    env = _make_env(
        [
            {
                "high_level_command": "attack",
                "race": "zerg",
                "region": "A",
                "agent": True,
                "unit_pool": ["QUEEN"],
                "mineral_budget": 500,
                "vespene_budget": 0,
            },
            {
                "high_level_command": "defend",
                "race": "terran",
                "region": "A",
                "unit_pool": ["BATTLECRUISER"],
                "mineral_budget": 500,
                "vespene_budget": 0,
            },
        ],
        env_mode="heal",
    )
    env.bot.client.debug_set_unit_value = AsyncMock()
    asyncio.run(env._apply_damaged_units())
    env.bot.client.debug_set_unit_value.assert_not_called()


def test_apply_damaged_units_runs_when_flag_on():
    env = _make_env(
        [
            {
                "high_level_command": "attack",
                "race": "zerg",
                "region": "A",
                "agent": True,
                "unit_pool": ["QUEEN"],
                "mineral_budget": 500,
                "vespene_budget": 0,
            },
            {
                "high_level_command": "defend",
                "race": "terran",
                "region": "A",
                "unit_pool": ["BATTLECRUISER"],
                "mineral_budget": 500,
                "vespene_budget": 0,
            },
        ],
        env_mode="heal",
    )
    env.training_conf["env"]["damaged_units"] = True
    queen = MagicMock()
    queen.tag = 11
    queen.type_id = UnitTypeId.QUEEN
    queen.is_structure = False
    queen.health_max = 175.0
    queen.shield_max = 0.0
    env.bot.all_units = [queen]
    env.bot.client.debug_set_unit_value = AsyncMock()
    asyncio.run(env._apply_damaged_units())
    env.bot.client.debug_set_unit_value.assert_awaited()
    env.bot.client.step.assert_awaited_once()
    env._refresh_bot_state.assert_awaited()
