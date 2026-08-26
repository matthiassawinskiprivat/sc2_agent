"""env_mode=cloak detector spawn and upgrade helper."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


def _make_env(unit_pools, env_mode=None):
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
    env.bot.client.debug_upgrade = AsyncMock()
    env.bot.client.step = AsyncMock()
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
    env._calculate_spawns_for_pool = MagicMock(return_value=["MARINE"])
    env._resolve_map = MagicMock(return_value=MagicMock(path="map.SC2Map"))
    env.map_name = "map"
    env._refresh_bot_state = AsyncMock()
    return env


def _all_create_cmds(env):
    """Flatten all ``debug_create_unit`` batches (now split per player)."""
    out = []
    for call in env.bot.client.debug_create_unit.await_args_list:
        out.extend(call.args[0])
    return out


def _spawn(env):
    def get_valid_pos(p):
        return p

    def get_offset_pos(base, index, total):
        return Point2((base.x + float(index), base.y))

    def to_id(name):
        return getattr(UnitTypeId, name)

    return asyncio.run(env._spawn_from_unit_pools(get_valid_pos, get_offset_pos, to_id))


def test_is_cloak_env_mode():
    env = _make_env([], env_mode="cloak")
    assert env._is_cloak_env_mode() is True
    assert env._has_env_mode("cloak") is True
    env_multi = _make_env([], env_mode="heal,cloak,drop")
    assert env_multi._has_env_mode("heal") is True
    assert env_multi._has_env_mode("cloak") is True
    assert env_multi._has_env_mode("drop") is True
    assert env_multi._is_cloak_env_mode() is True
    env2 = _make_env([], env_mode=None)
    assert env2._is_cloak_env_mode() is False
    assert env2._has_env_mode("heal") is False


def test_heal_mode_spawns_creep_tumors_for_zerg():
    pools = [
        {
            "high_level_command": "attack",
            "race": "zerg",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZERGLING"],
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
    env = _make_env(pools, env_mode="heal")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["ZERGLING"] if pool["race"] == "zerg" else ["MARINE"]
        )
    )

    _spawn(env)
    cmds = _all_create_cmds(env)
    types = [c[0] for c in cmds]
    assert UnitTypeId.CREEPTUMOR in types


def test_without_heal_mode_no_creep_tumors():
    pools = [
        {
            "high_level_command": "attack",
            "race": "zerg",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZERGLING"],
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
    env = _make_env(pools, env_mode="cloak")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["ZERGLING"] if pool["race"] == "zerg" else ["MARINE"]
        )
    )

    _spawn(env)
    # Army batch only (detectors are deferred).
    cmds = _all_create_cmds(env)
    types = [c[0] for c in cmds]
    assert UnitTypeId.CREEPTUMOR not in types


def test_drop_mode_alone_runs_upgrade_ladder():
    env = _make_env([], env_mode="drop")
    asyncio.run(env._apply_drop_env_mode_upgrades())
    assert env.bot.client.debug_upgrade.await_count == 20


def test_drop_with_cloak_skips_duplicate_upgrades():
    env = _make_env([], env_mode="heal,cloak,drop")
    asyncio.run(env._apply_drop_env_mode_upgrades())
    env.bot.client.debug_upgrade.assert_not_called()


def test_cloak_mode_detector_types():
    assert (
        CaptureTheFlagSingleEnv._cloak_mode_detector_type("protoss")
        == UnitTypeId.OBSERVER
    )
    assert CaptureTheFlagSingleEnv._cloak_mode_detector_type("terran") == UnitTypeId.RAVEN
    assert CaptureTheFlagSingleEnv._cloak_mode_detector_type("zerg") == UnitTypeId.OVERSEER
    assert CaptureTheFlagSingleEnv._cloak_mode_detector_type("unknown") is None


def test_cloak_mode_defers_detectors_per_pool_all_races():
    pools = [
        {
            "high_level_command": "attack",
            "race": "protoss",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZEALOT"],
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
        side_effect=lambda pool: {
            "protoss": ["ZEALOT"],
            "terran": ["MARINE"],
            "zerg": ["ZERGLING"],
        }[pool["race"]]
    )

    _, _, _, deferred_detectors = _spawn(env)
    types = [c[0] for c in deferred_detectors]
    assert types.count(UnitTypeId.OBSERVER) == 1
    assert types.count(UnitTypeId.RAVEN) == 1
    assert types.count(UnitTypeId.OVERSEER) == 1

    # Detectors must not be in the army create batch.
    army_cmds = _all_create_cmds(env)
    army_types = [c[0] for c in army_cmds]
    assert UnitTypeId.OBSERVER not in army_types
    assert UnitTypeId.RAVEN not in army_types
    assert UnitTypeId.OVERSEER not in army_types


def test_spawn_cloak_mode_detectors_creates_all_races_directly():
    env = _make_env([], env_mode="cloak")
    cmds = [
        [UnitTypeId.OBSERVER, 1, Point2((1.0, 1.0)), 1],
        [UnitTypeId.RAVEN, 1, Point2((2.0, 2.0)), 2],
        [UnitTypeId.OVERSEER, 1, Point2((3.0, 3.0)), 1],
    ]
    asyncio.run(env._spawn_cloak_mode_detectors(cmds))
    assert env.bot.client.debug_create_unit.await_count == 2
    created = _all_create_cmds(env)
    types = [c[0] for c in created]
    assert UnitTypeId.OBSERVER in types
    assert UnitTypeId.RAVEN in types
    assert UnitTypeId.OVERSEER in types
    assert UnitTypeId.OVERLORD not in types


def test_non_cloak_mode_does_not_spawn_detectors():
    pools = [
        {
            "high_level_command": "attack",
            "race": "zerg",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZERGLING"],
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
    env = _make_env(pools, env_mode=None)
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["ZERGLING"] if pool["race"] == "zerg" else ["MARINE"]
        )
    )

    _, _, _, deferred_detectors = _spawn(env)
    assert deferred_detectors == []
    cmds = _all_create_cmds(env)
    types = [c[0] for c in cmds]
    assert UnitTypeId.OVERSEER not in types
    assert UnitTypeId.RAVEN not in types


def test_apply_cloak_env_mode_upgrades_calls_debug_upgrade():
    env = _make_env([], env_mode="cloak")
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    assert env.bot.client.debug_upgrade.await_count == 20


def test_debug_create_units_by_player_splits_batches():
    """Mixed-owner create lists are split so trailing player cmds are not dropped."""
    env = _make_env([], env_mode=None)
    cmds = [
        [UnitTypeId.DARKTEMPLAR, 1, Point2((1.0, 1.0)), 1],
        [UnitTypeId.DARKTEMPLAR, 1, Point2((1.5, 1.0)), 1],
        [UnitTypeId.MARAUDER, 1, Point2((2.0, 2.0)), 2],
        [UnitTypeId.MARAUDER, 1, Point2((2.5, 2.0)), 2],
    ]
    asyncio.run(env._debug_create_units_by_player(cmds))
    assert env.bot.client.debug_create_unit.await_count == 2
    first = env.bot.client.debug_create_unit.await_args_list[0].args[0]
    second = env.bot.client.debug_create_unit.await_args_list[1].args[0]
    assert all(c[3] == 1 for c in first)
    assert all(c[3] == 2 for c in second)
    assert [c[0] for c in first] == [UnitTypeId.DARKTEMPLAR, UnitTypeId.DARKTEMPLAR]
    assert [c[0] for c in second] == [UnitTypeId.MARAUDER, UnitTypeId.MARAUDER]


def test_trainee_race_log_suffix_includes_assignment():
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.episode_counter = 2
    env._alternate_trainee_race = True
    env.unit_pools_config = [
        {"race": "protoss", "high_level_command": "attack"},
        {"race": "terran", "high_level_command": "defend"},
    ]
    env.player_units = {1: [1, 2, 3], 2: [4, 5]}
    suffix = env._trainee_race_log_suffix()
    assert "trainee=terran" in suffix
    assert "P1=protoss(3)" in suffix
    assert "P2=terran(2)" in suffix


def test_apply_cloak_env_mode_upgrades_noop_without_mode():
    env = _make_env([], env_mode=None)
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    env.bot.client.debug_upgrade.assert_not_called()
