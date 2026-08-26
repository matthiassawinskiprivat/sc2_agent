"""Zerg creep tumor + Protoss shield battery spawn rules for CTF heal curriculum."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


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
    _spawn(env)

    cmds = _all_create_cmds(env)
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
    _spawn(env)

    cmds = _all_create_cmds(env)
    tumor_cmds = [c for c in cmds if c[0] == UnitTypeId.CREEPTUMOR]
    assert len(tumor_cmds) == 1
    assert tumor_cmds[0][2] == Point2((21.0, 21.0))  # defend spawn offset


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
