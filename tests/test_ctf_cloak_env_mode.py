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

    def get_valid_pos(p):
        return p

    def get_offset_pos(base, index, total):
        return Point2((base.x + float(index), base.y))

    def to_id(name):
        return getattr(UnitTypeId, name)

    asyncio.run(env._spawn_from_unit_pools(get_valid_pos, get_offset_pos, to_id))
    cmds = env.bot.client.debug_create_unit.await_args.args[0]
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

    def get_valid_pos(p):
        return p

    def get_offset_pos(base, index, total):
        return Point2((base.x + 1.0, base.y + 1.0))

    def to_id(name):
        return getattr(UnitTypeId, name)

    asyncio.run(env._spawn_from_unit_pools(get_valid_pos, get_offset_pos, to_id))
    cmds = env.bot.client.debug_create_unit.await_args.args[0]
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


def test_cloak_mode_spawns_detectors_per_pool():
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
    ]
    env = _make_env(pools, env_mode="cloak")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: (
            ["ZEALOT"] if pool["race"] == "protoss" else ["MARINE"]
        )
    )

    def get_valid_pos(p):
        return p

    def get_offset_pos(base, index, total):
        return Point2((base.x + float(index), base.y))

    def to_id(name):
        return getattr(UnitTypeId, name)

    asyncio.run(env._spawn_from_unit_pools(get_valid_pos, get_offset_pos, to_id))
    cmds = env.bot.client.debug_create_unit.await_args.args[0]
    types = [c[0] for c in cmds]
    assert UnitTypeId.OBSERVER in types
    assert UnitTypeId.RAVEN in types


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

    def get_valid_pos(p):
        return p

    def get_offset_pos(base, index, total):
        return Point2((base.x + 1.0, base.y + 1.0))

    def to_id(name):
        return getattr(UnitTypeId, name)

    asyncio.run(env._spawn_from_unit_pools(get_valid_pos, get_offset_pos, to_id))
    cmds = env.bot.client.debug_create_unit.await_args.args[0]
    types = [c[0] for c in cmds]
    assert UnitTypeId.OVERSEER not in types
    assert UnitTypeId.RAVEN not in types


def test_apply_cloak_env_mode_upgrades_calls_debug_upgrade():
    env = _make_env([], env_mode="cloak")
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    assert env.bot.client.debug_upgrade.await_count == 20


def test_apply_cloak_env_mode_upgrades_noop_without_mode():
    env = _make_env([], env_mode=None)
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    env.bot.client.debug_upgrade.assert_not_called()
