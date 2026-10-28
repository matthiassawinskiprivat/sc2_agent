"""Drop mode adds one free transport per squad, outside the pool budget."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from train.envs.ctf.drop_transports import drop_transport_name, spawn_drop_transports


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
    out = []
    for call in env.bot.client.debug_create_unit.await_args_list:
        out.extend(call.args[0])
    return out


def _spawn(env):
    def get_valid_pos(p):
        return p

    def get_offset_pos(base, index, total):
        return Point2((base.x + float(index), base.y + float(total)))

    def to_id(name):
        return getattr(UnitTypeId, name)

    return asyncio.run(env._spawn_from_unit_pools(get_valid_pos, get_offset_pos, to_id))


def _pools():
    return [
        {
            "high_level_command": "attack",
            "race": "protoss",
            "region": "A",
            "agent": True,
            "unit_pool": ["ZEALOT", "WARPPRISM"],
            "mineral_budget": 650,
            "vespene_budget": 400,
        },
        {
            "high_level_command": "defend",
            "race": "terran",
            "region": "A",
            "unit_pool": ["MARINE", "MEDIVAC"],
            "mineral_budget": 500,
            "vespene_budget": 300,
        },
        {
            "high_level_command": "defend",
            "race": "zerg",
            "region": "A",
            "unit_pool": ["ZERGLING", "OVERLORDTRANSPORT"],
            "mineral_budget": 500,
            "vespene_budget": 300,
        },
    ]


def test_drop_transport_name_by_race():
    assert drop_transport_name("terran") == "MEDIVAC"
    assert drop_transport_name("Protoss") == "WARPPRISM"
    assert drop_transport_name("zerg") == "OVERLORDTRANSPORT"
    assert drop_transport_name("unknown") is None


def test_drop_mode_adds_one_free_transport_per_squad():
    """Budgeted army stays whatever the pool roll returned; drop adds one transport."""
    pools = _pools()
    env = _make_env(pools, env_mode="heal,cloak,spellcaster,drop")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: {
            "protoss": ["ZEALOT"],
            "terran": ["MARINE"],
            "zerg": ["ZERGLING"],
        }[pool["race"]]
    )
    budgets_before = [
        (p["mineral_budget"], p["vespene_budget"], list(p["unit_pool"])) for p in pools
    ]

    _, _, _, _, transports = _spawn(env)

    assert env._calculate_spawns_for_pool.call_count == 3
    for pool, call in zip(pools, env._calculate_spawns_for_pool.call_args_list):
        assert call.args[0] is pool
    assert [
        (p["mineral_budget"], p["vespene_budget"], list(p["unit_pool"])) for p in pools
    ] == budgets_before

    types = [c[0] for c in transports]
    assert types.count(UnitTypeId.WARPPRISM) == 1
    assert types.count(UnitTypeId.MEDIVAC) == 1
    assert types.count(UnitTypeId.OVERLORDTRANSPORT) == 1
    assert len(transports) == 3

    # The free transport is its own create. A Medivac can still appear in the
    # army when heal mode guarantees one the budget roll missed.
    army_types = [c[0] for c in _all_create_cmds(env)]
    assert UnitTypeId.WARPPRISM not in army_types
    assert UnitTypeId.OVERLORDTRANSPORT not in army_types
    assert army_types.count(UnitTypeId.MEDIVAC) == 1


def test_drop_transport_is_extra_when_budget_already_rolled_one():
    """A transport rolled from unit_pool does not replace the free drop transport."""
    pools = [
        {
            "high_level_command": "attack",
            "race": "terran",
            "region": "A",
            "agent": True,
            "unit_pool": ["MARINE", "MEDIVAC"],
            "mineral_budget": 650,
            "vespene_budget": 400,
        }
    ]
    env = _make_env(pools, env_mode="drop")
    env._calculate_spawns_for_pool = MagicMock(return_value=["MARINE", "MEDIVAC"])

    _, _, _, _, transports = _spawn(env)

    army_types = [c[0] for c in _all_create_cmds(env)]
    assert army_types.count(UnitTypeId.MEDIVAC) == 1
    assert [c[0] for c in transports] == [UnitTypeId.MEDIVAC]


def test_drop_transport_does_not_make_a_third_medivac():
    """Two Medivacs from the pool already fill the squad cap."""
    pools = [
        {
            "high_level_command": "attack",
            "race": "terran",
            "region": "A",
            "agent": True,
            "unit_pool": ["MARINE", "MEDIVAC"],
            "mineral_budget": 650,
            "vespene_budget": 400,
        }
    ]
    env = _make_env(pools, env_mode="drop")
    env._calculate_spawns_for_pool = MagicMock(
        return_value=["MARINE", "MEDIVAC", "MEDIVAC"]
    )

    _, _, _, _, transports = _spawn(env)

    army_types = [c[0] for c in _all_create_cmds(env)]
    assert army_types.count(UnitTypeId.MEDIVAC) == 2
    assert transports == []


def test_without_drop_mode_no_free_transport():
    env = _make_env(_pools(), env_mode="heal,cloak,spellcaster")
    env._calculate_spawns_for_pool = MagicMock(
        side_effect=lambda pool: {
            "protoss": ["ZEALOT"],
            "terran": ["MARINE"],
            "zerg": ["ZERGLING"],
        }[pool["race"]]
    )

    _, _, _, _, transports = _spawn(env)

    assert transports == []


def test_spawn_drop_transports_creates_per_owner():
    env = _make_env([], env_mode="drop")
    cmds = [
        [UnitTypeId.WARPPRISM, 1, Point2((1.0, 1.0)), 1],
        [UnitTypeId.MEDIVAC, 1, Point2((2.0, 2.0)), 2],
        [UnitTypeId.OVERLORDTRANSPORT, 1, Point2((3.0, 3.0)), 1],
    ]
    asyncio.run(env._spawn_drop_mode_transports(cmds))
    assert env.bot.client.debug_create_unit.await_count == 2
    created = _all_create_cmds(env)
    types = [c[0] for c in created]
    assert UnitTypeId.WARPPRISM in types
    assert UnitTypeId.MEDIVAC in types
    assert UnitTypeId.OVERLORDTRANSPORT in types


def test_spawn_drop_transports_noop_when_empty():
    bot = MagicMock()
    bot.client.debug_create_unit = AsyncMock()
    refresh = AsyncMock()
    asyncio.run(spawn_drop_transports(bot=bot, transport_cmds=[], refresh_state=refresh))
    bot.client.debug_create_unit.assert_not_called()
    refresh.assert_not_awaited()
