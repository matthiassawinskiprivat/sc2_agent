"""Unit tests for per-episode CTF trainee race assignment."""

import numpy as np

from src.agent.squad_constants import SQUAD_RACE_DIM, encode_race_vector, include_race_enabled
from train.envs.ctf.pool_assignment import (
    alternate_trainee_race_enabled,
    get_level_races,
    get_trainee_race,
    is_pool_trainee,
    player_races_for_episode,
    pool_player_id,
)


def test_get_level_races_preserves_first_seen_order():
    pools = [
        {"race": "zerg"},
        {"race": "terran"},
        {"race": "zerg"},
        {"race": "protoss"},
    ]
    assert get_level_races(pools) == ["zerg", "terran", "protoss"]


def test_three_race_level_cycles_trainee_race():
    pools = [
        {"race": "terran"},
        {"race": "zerg"},
        {"race": "protoss"},
    ]
    assert get_trainee_race(pools, 1, alternate=True) == "terran"
    assert get_trainee_race(pools, 2, alternate=True) == "zerg"
    assert get_trainee_race(pools, 3, alternate=True) == "protoss"
    assert get_trainee_race(pools, 4, alternate=True) == "terran"


def test_three_race_static_spawn_owners_while_trainee_cycles():
    pools = [
        {"race": "terran"},
        {"race": "zerg"},
        {"race": "protoss"},
    ]
    for episode in (1, 2, 3, 4):
        assert pool_player_id(pools[0], pools, episode, alternate=True) == 1
        assert pool_player_id(pools[1], pools, episode, alternate=True) == 2
        assert pool_player_id(pools[2], pools, episode, alternate=True) == 2
    assert is_pool_trainee(pools[0], pools, 1, alternate=True)
    assert is_pool_trainee(pools[1], pools, 2, alternate=True)
    assert is_pool_trainee(pools[2], pools, 3, alternate=True)


def test_alternate_trainee_race_enabled_reads_nested_env_flag():
    assert alternate_trainee_race_enabled({"env": {"alternate_trainee_race": True}})
    assert not alternate_trainee_race_enabled({"env": {}})
    assert alternate_trainee_race_enabled({"alternate_trainee_race": True})


def test_include_race_enabled_reads_nested_env_flag():
    assert include_race_enabled({"env": {"include_race": True}})
    # Omitted → default True (stable 144-wide race obs after ctf_level_01).
    assert include_race_enabled({"env": {}})
    assert include_race_enabled({})
    assert include_race_enabled({"include_race": True})
    assert not include_race_enabled({"include_race": False})
    assert not include_race_enabled({"env": {"include_race": False}})


def test_encode_race_vector_own_enemy_one_hots():
    vec = encode_race_vector("zerg", "protoss")
    assert vec.shape == (SQUAD_RACE_DIM,)
    assert vec.dtype == np.float32
    # order: terran, zerg, protoss | terran, zerg, protoss
    assert np.allclose(vec, [0, 1, 0, 0, 0, 1])


def test_encode_race_vector_unknown_is_zeros():
    assert np.allclose(encode_race_vector(None, ""), np.zeros(SQUAD_RACE_DIM))


def test_player_races_for_episode_are_static_spawn_owners():
    pools = [
        {"race": "terran", "high_level_command": "attack"},
        {"race": "zerg", "high_level_command": "defend"},
    ]
    assert player_races_for_episode(pools, 1, alternate=True) == ("terran", "zerg")
    assert player_races_for_episode(pools, 2, alternate=True) == ("terran", "zerg")


def test_pool_player_id_without_race_uses_agent_flag():
    pools = [{"agent": True}, {}]
    assert pool_player_id(pools[0], pools, 5, alternate=True) == 1
    assert pool_player_id(pools[1], pools, 5, alternate=True) == 2


def test_pool_player_id_static_while_trainee_flag_flips():
    pools = [
        {"race": "terran", "high_level_command": "attack"},
        {"race": "zerg", "high_level_command": "defend"},
    ]
    for episode in (1, 2, 3):
        assert pool_player_id(pools[0], pools, episode, alternate=True) == 1
        assert pool_player_id(pools[1], pools, episode, alternate=True) == 2
    assert is_pool_trainee(pools[0], pools, 1, alternate=True)
    assert not is_pool_trainee(pools[1], pools, 1, alternate=True)
    assert not is_pool_trainee(pools[0], pools, 2, alternate=True)
    assert is_pool_trainee(pools[1], pools, 2, alternate=True)
