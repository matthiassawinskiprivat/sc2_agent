"""Heal-mode corridor tumors (burrowed spawn clusters, no map-wide grid)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from train.envs.ctf.heal_creep import (
    apply_heal_map_creep,
    creep_source_create_cmds,
    spawn_tumor_cluster,
    zerg_arena_tumor_positions,
)


def _xy_set(pts):
    return {(round(p.x, 6), round(p.y, 6)) for p in pts}


def _cluster_xy(x, y):
    return {(round(px, 6), round(py, 6)) for px, py in spawn_tumor_cluster((x, y))}


def test_creep_cmds_use_burrowed_tumor_not_morphing_type():
    cmds = creep_source_create_cmds([Point2((4.0, 5.0))], player_id=2)
    assert cmds == [[UnitTypeId.CREEPTUMORBURROWED, 1, Point2((4.0, 5.0)), 2]]


def test_spawn_cluster_is_center_plus_eight_ring():
    pts = spawn_tumor_cluster((10.0, 20.0), radius=5.0)
    assert pts[0] == (10.0, 20.0)
    assert len(pts) == 9
    assert (15.0, 20.0) in pts
    assert (10.0, 25.0) in pts
    assert (5.0, 20.0) in pts
    assert (10.0, 15.0) in pts


def test_zerg_defend_clusters_home_and_keeps_single_midpoint():
    pools = [
        {"race": "zerg", "high_level_command": "defend", "region": "A"},
        {"race": "protoss", "high_level_command": "attack", "region": "A"},
    ]
    centers = [
        (2, "attack", Point2((10.0, 10.0)), "A"),
        (1, "defend", Point2((30.0, 10.0)), "A"),
    ]
    got = _xy_set(zerg_arena_tumor_positions(centers, pools))
    assert got == _cluster_xy(30.0, 10.0) | {(20.0, 10.0)}
    assert (10.0, 10.0) not in got


def test_zerg_attack_clusters_home_and_assigned_defend():
    pools = [
        {"race": "zerg", "high_level_command": "attack", "region": "A"},
        {"race": "protoss", "high_level_command": "defend", "region": "A"},
    ]
    centers = [
        (1, "attack", Point2((10.0, 10.0)), "A"),
        (2, "defend", Point2((30.0, 10.0)), "A"),
    ]
    got = _xy_set(zerg_arena_tumor_positions(centers, pools))
    assert got == _cluster_xy(10.0, 10.0) | _cluster_xy(30.0, 10.0) | {(20.0, 10.0)}


def test_two_arenas_attack_and_defend_do_not_cluster_enemy_attack_spawn():
    """Level-19 shape: Zerg attack on A, Zerg defend on B."""
    pools = [
        {"race": "zerg", "high_level_command": "attack", "region": "A"},
        {"race": "protoss", "high_level_command": "defend", "region": "A"},
        {"race": "zerg", "high_level_command": "defend", "region": "B"},
        {"race": "protoss", "high_level_command": "attack", "region": "B"},
    ]
    centers = [
        (1, "attack", Point2((10.0, 10.0)), "A"),
        (2, "defend", Point2((30.0, 10.0)), "A"),
        (2, "attack", Point2((10.0, 50.0)), "B"),
        (1, "defend", Point2((30.0, 50.0)), "B"),
    ]
    got = _xy_set(zerg_arena_tumor_positions(centers, pools))
    expected = (
        _cluster_xy(10.0, 10.0)
        | _cluster_xy(30.0, 10.0)
        | {(20.0, 10.0)}
        | _cluster_xy(30.0, 50.0)
        | {(20.0, 50.0)}
    )
    assert got == expected
    assert (10.0, 50.0) not in got


def test_mirror_zerg_arena_dedupes_overlapping_spawn_clusters():
    pools = [
        {"race": "zerg", "high_level_command": "attack", "region": "A"},
        {"race": "zerg", "high_level_command": "defend", "region": "A"},
    ]
    centers = [
        (1, "attack", Point2((0.0, 0.0)), "A"),
        (1, "defend", Point2((8.0, 0.0)), "A"),
    ]
    got = _xy_set(zerg_arena_tumor_positions(centers, pools))
    assert got == _cluster_xy(0.0, 0.0) | _cluster_xy(8.0, 0.0) | {(4.0, 0.0)}


def test_no_positions_without_spawn_centers():
    pools = [{"race": "zerg", "high_level_command": "attack", "region": "A"}]
    assert zerg_arena_tumor_positions([], pools) == []


def test_apply_heal_map_creep_plants_only_given_positions():
    bot = MagicMock()
    bot.client.debug_create_unit = AsyncMock()
    pts = [Point2((10.0, 10.0)), Point2((20.0, 10.0))]
    n = asyncio.run(apply_heal_map_creep(bot=bot, player_id=1, positions=pts))
    assert n == 2
    cmds = bot.client.debug_create_unit.await_args.args[0]
    assert [c[0] for c in cmds] == [UnitTypeId.CREEPTUMORBURROWED] * 2
    assert all(c[3] == 1 for c in cmds)


def test_apply_heal_map_creep_skips_empty_positions():
    bot = MagicMock()
    bot.client.debug_create_unit = AsyncMock()
    n = asyncio.run(apply_heal_map_creep(bot=bot, player_id=1, positions=[]))
    assert n == 0
    bot.client.debug_create_unit.assert_not_called()


def _heal_env(pools, env_mode="heal"):
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.unit_pools_config = pools
    conf = {"env": {}}
    if env_mode is not None:
        conf["env"]["env_mode"] = env_mode
        conf["env_mode"] = env_mode
    env.training_conf = conf
    env.bot = MagicMock()
    env.bot.client.debug_create_unit = AsyncMock()
    env._pool_player_id = lambda pool: 1 if pool.get("agent") else 2
    return env


def test_env_heal_attack_zerg_plants_spawn_clusters_and_midpoint():
    pools = [
        {
            "high_level_command": "attack",
            "race": "zerg",
            "region": "A",
            "agent": True,
        },
        {
            "high_level_command": "defend",
            "race": "protoss",
            "region": "A",
        },
    ]
    env = _heal_env(pools, env_mode="heal")
    spawn_centers = [
        (1, "attack", Point2((10.0, 10.0)), "A"),
        (2, "defend", Point2((30.0, 10.0)), "A"),
    ]
    asyncio.run(env._apply_heal_map_creep(spawn_centers))
    cmds = []
    for call in env.bot.client.debug_create_unit.await_args_list:
        cmds.extend(call.args[0])
    got = {(round(c[2].x, 6), round(c[2].y, 6)) for c in cmds}
    assert got == _cluster_xy(10.0, 10.0) | _cluster_xy(30.0, 10.0) | {(20.0, 10.0)}
    assert all(c[0] == UnitTypeId.CREEPTUMORBURROWED for c in cmds)
    assert all(c[3] == 1 for c in cmds)


def test_env_heal_defend_zerg_skips_enemy_attack_spawn():
    pools = [
        {
            "high_level_command": "defend",
            "race": "zerg",
            "region": "A",
            "agent": True,
        },
        {
            "high_level_command": "attack",
            "race": "protoss",
            "region": "A",
        },
    ]
    env = _heal_env(pools, env_mode="heal")
    spawn_centers = [
        (2, "attack", Point2((10.0, 10.0)), "A"),
        (1, "defend", Point2((30.0, 10.0)), "A"),
    ]
    asyncio.run(env._apply_heal_map_creep(spawn_centers))
    cmds = []
    for call in env.bot.client.debug_create_unit.await_args_list:
        cmds.extend(call.args[0])
    got = {(round(c[2].x, 6), round(c[2].y, 6)) for c in cmds}
    assert got == _cluster_xy(30.0, 10.0) | {(20.0, 10.0)}
    assert (10.0, 10.0) not in got


def test_env_heal_skips_without_zerg_pool():
    pools = [
        {"high_level_command": "attack", "race": "protoss", "region": "A", "agent": True},
        {"high_level_command": "defend", "race": "terran", "region": "A"},
    ]
    env = _heal_env(pools, env_mode="heal")
    spawn_centers = [
        (1, "attack", Point2((10.0, 10.0)), "A"),
        (2, "defend", Point2((30.0, 10.0)), "A"),
    ]
    asyncio.run(env._apply_heal_map_creep(spawn_centers))
    env.bot.client.debug_create_unit.assert_not_called()


def test_env_skips_creep_without_heal_mode():
    pools = [
        {"high_level_command": "attack", "race": "zerg", "region": "A", "agent": True},
    ]
    env = _heal_env(pools, env_mode="cloak")
    spawn_centers = [
        (1, "attack", Point2((10.0, 10.0)), "A"),
        (2, "defend", Point2((30.0, 10.0)), "A"),
    ]
    asyncio.run(env._apply_heal_map_creep(spawn_centers))
    env.bot.client.debug_create_unit.assert_not_called()
