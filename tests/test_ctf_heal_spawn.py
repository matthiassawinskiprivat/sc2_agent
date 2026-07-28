"""Zerg creep tumor spawn rules for CTF heal curriculum."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


def _make_env(unit_pools):
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.unit_pools_config = unit_pools
    env.bot = MagicMock()
    env.bot.client = MagicMock()
    env.bot.client.debug_create_unit = AsyncMock()
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


def test_zerg_attack_spawns_tumor_on_attack_and_defend_positions():
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

    def get_valid_pos(p):
        return p

    def get_offset_pos(base, index, total):
        return Point2((base.x + 1.0, base.y + 1.0))

    def to_id(name):
        return getattr(UnitTypeId, name)

    asyncio.run(env._spawn_from_unit_pools(get_valid_pos, get_offset_pos, to_id))

    cmds = env.bot.client.debug_create_unit.await_args.args[0]
    tumor_cmds = [c for c in cmds if c[0] == UnitTypeId.CREEPTUMOR]
    assert len(tumor_cmds) == 2  # attack region + defender _d
    positions = {(c[2].x, c[2].y) for c in tumor_cmds}
    assert (11.0, 11.0) in positions  # offset from attack center
    assert (21.0, 21.0) in positions  # offset from defend center


def test_zerg_defend_spawns_tumor_on_defend_only():
    pools = [
        {
            "high_level_command": "defend",
            "race": "zerg",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZERGLING", "QUEEN"],
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
    tumor_cmds = [c for c in cmds if c[0] == UnitTypeId.CREEPTUMOR]
    assert len(tumor_cmds) == 1
    assert tumor_cmds[0][2] == Point2((21.0, 21.0))  # defend spawn offset


def test_non_zerg_pool_does_not_force_spawn_battery():
    """Shield batteries come from unit_pool budget, not forced structure spawn."""
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
            ["ZEALOT"] if pool["race"] == "protoss" else ["MARINE"]
        )
    )

    def get_valid_pos(p):
        return p

    def get_offset_pos(base, index, total):
        return Point2((base.x + 1.0, base.y))

    def to_id(name):
        return getattr(UnitTypeId, name)

    asyncio.run(env._spawn_from_unit_pools(get_valid_pos, get_offset_pos, to_id))

    cmds = env.bot.client.debug_create_unit.await_args.args[0]
    assert not any(c[0] == UnitTypeId.SHIELDBATTERY for c in cmds)
    assert any(c[0] == UnitTypeId.PYLON for c in cmds)
