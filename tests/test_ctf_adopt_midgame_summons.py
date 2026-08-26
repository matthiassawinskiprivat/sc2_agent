"""Mid-episode summons join / leave their parent's squad pool."""

from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.utils.unit_feature_helpers import (
    adopt_midgame_summons_into_pools,
    prune_dead_from_pool_maps,
)
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


def _unit(*, tag, owner_id, type_id, position, distance_to=None):
    u = MagicMock()
    u.tag = tag
    u.owner_id = owner_id
    u.type_id = type_id
    u.is_structure = False
    u.position = Point2(position)
    if distance_to is not None:
        u.distance_to = MagicMock(side_effect=distance_to)
    else:
        u.distance_to = MagicMock(return_value=1.0)
    return u


def _adopt(child, parents, pool_idx=0, prefer_tag=None):
    """Adopt *child* into a pool that already contains *parents*."""
    unit_pool_index = {int(p.tag): pool_idx for p in parents}
    unit_command_map = {int(p.tag): "attack" for p in parents}
    unit_target_map = {int(prefer_tag or parents[0].tag): Point2((80.0, 80.0))}
    adopt_midgame_summons_into_pools(
        list(parents) + [child],
        unit_pool_index,
        unit_command_map=unit_command_map,
        unit_target_map=unit_target_map,
    )
    return unit_pool_index, unit_command_map, unit_target_map


def test_adopt_nova_ball_inherits_nearest_disruptor_pool():
    disruptor = _unit(tag=10, owner_id=1, type_id=UnitTypeId.DISRUPTOR, position=(10.0, 10.0))
    marine = _unit(tag=11, owner_id=1, type_id=UnitTypeId.MARINE, position=(50.0, 50.0))
    nova = _unit(
        tag=20,
        owner_id=1,
        type_id=UnitTypeId.DISRUPTORPHASED,
        position=(12.0, 12.0),
        distance_to=lambda other: 2.0 if other.tag == 10 else 40.0,
    )

    unit_pool_index, unit_command_map, unit_target_map = _adopt(
        nova, [disruptor, marine], prefer_tag=10
    )

    assert unit_pool_index[20] == 0
    assert unit_command_map[20] == "attack"
    assert unit_target_map[20] == Point2((80.0, 80.0))


def test_adopt_locust_inherits_swarmhost_pool():
    host = _unit(tag=10, owner_id=1, type_id=UnitTypeId.SWARMHOSTMP, position=(10.0, 10.0))
    locust = _unit(
        tag=20,
        owner_id=1,
        type_id=UnitTypeId.LOCUSTMP,
        position=(12.0, 12.0),
        distance_to=lambda other: 1.0,
    )
    unit_pool_index, _, _ = _adopt(locust, [host])
    assert unit_pool_index[20] == 0


def test_adopt_broodling_inherits_broodlord_pool():
    lord = _unit(tag=10, owner_id=1, type_id=UnitTypeId.BROODLORD, position=(10.0, 10.0))
    broodling = _unit(
        tag=20,
        owner_id=1,
        type_id=UnitTypeId.BROODLING,
        position=(12.0, 12.0),
        distance_to=lambda other: 1.0,
    )
    unit_pool_index, _, _ = _adopt(broodling, [lord])
    assert unit_pool_index[20] == 0


def test_adopt_interceptor_is_not_added_to_carrier_pool():
    """Carrier AI owns Interceptors — adopting them floods the 24-slot pointer."""
    carrier = _unit(tag=10, owner_id=1, type_id=UnitTypeId.CARRIER, position=(10.0, 10.0))
    interceptor = _unit(
        tag=20,
        owner_id=1,
        type_id=UnitTypeId.INTERCEPTOR,
        position=(12.0, 12.0),
        distance_to=lambda other: 1.0,
    )
    unit_pool_index, _, _ = _adopt(interceptor, [carrier])
    assert 20 not in unit_pool_index
    assert unit_pool_index == {10: 0}


def test_adopt_adept_shade_inherits_adept_pool():
    adept = _unit(tag=10, owner_id=1, type_id=UnitTypeId.ADEPT, position=(10.0, 10.0))
    shade = _unit(
        tag=20,
        owner_id=1,
        type_id=UnitTypeId.ADEPTPHASESHIFT,
        position=(12.0, 12.0),
        distance_to=lambda other: 1.0,
    )
    unit_pool_index, _, _ = _adopt(shade, [adept])
    assert unit_pool_index[20] == 0


def test_adopt_prefers_parent_type_over_closer_marine():
    disruptor = _unit(tag=10, owner_id=1, type_id=UnitTypeId.DISRUPTOR, position=(30.0, 30.0))
    marine = _unit(tag=11, owner_id=1, type_id=UnitTypeId.MARINE, position=(11.0, 11.0))
    nova = _unit(
        tag=20,
        owner_id=1,
        type_id=UnitTypeId.DISRUPTORPHASED,
        position=(10.0, 10.0),
        distance_to=lambda other: 20.0 if other.tag == 10 else 1.0,
    )

    unit_pool_index = {10: 0, 11: 0}
    unit_command_map: dict = {}
    adopt_midgame_summons_into_pools(
        [disruptor, marine, nova],
        unit_pool_index,
        unit_command_map=unit_command_map,
    )

    assert unit_pool_index[20] == 0
    assert unit_command_map[20] == "attack"


def test_adopt_skips_already_mapped_summon():
    host = _unit(tag=10, owner_id=1, type_id=UnitTypeId.SWARMHOSTMP, position=(10.0, 10.0))
    locust = _unit(tag=20, owner_id=1, type_id=UnitTypeId.LOCUSTMP, position=(12.0, 12.0))

    unit_pool_index = {10: 0, 20: 1}
    adopt_midgame_summons_into_pools([host, locust], unit_pool_index)
    assert unit_pool_index[20] == 1


def test_prune_dead_summon_from_pool_maps():
    unit_pool_index = {10: 0, 20: 0, 11: 0}
    unit_command_map = {10: "attack", 20: "attack", 11: "attack"}
    unit_target_map = {10: Point2((1.0, 1.0)), 20: Point2((2.0, 2.0))}

    pruned = prune_dead_from_pool_maps(
        alive_tags={10, 11},
        unit_pool_index=unit_pool_index,
        unit_command_map=unit_command_map,
        unit_target_map=unit_target_map,
    )

    assert pruned == [20]
    assert 20 not in unit_pool_index
    assert 20 not in unit_command_map
    assert 20 not in unit_target_map
    assert unit_pool_index == {10: 0, 11: 0}


def test_env_prune_clears_dead_summon_and_anchor():
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.unit_pools_config = [
        {"high_level_command": "attack", "agent": True, "region": "A"},
        {"high_level_command": "defend", "region": "A"},
    ]
    env.unit_pool_index = {10: 0, 20: 0}
    env.player_units = {1: [10], 2: []}
    env._squad_anchor_tag_by_pool = {0: 20}
    env.bot = MagicMock()
    env.bot.unit_command_map = {10: "attack", 20: "attack"}
    env.bot.unit_target_map = {20: Point2((5.0, 5.0))}

    env._prune_dead_pool_membership()
    env._rebuild_squad_unit_tags()

    assert 20 not in env.unit_pool_index
    assert 20 not in env.bot.unit_command_map
    assert 20 not in env.bot.unit_target_map
    assert 0 not in env._squad_anchor_tag_by_pool
    assert env.squad_unit_tags[0][1] == [10]
