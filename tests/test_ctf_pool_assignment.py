"""Unit tests for per-episode CTF trainee race assignment."""

from train.envs.ctf.pool_assignment import (
    alternate_trainee_race_enabled,
    get_level_races,
    get_trainee_race,
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


def test_alternate_trainee_race_enabled_reads_nested_env_flag():
    assert alternate_trainee_race_enabled({"env": {"alternate_trainee_race": True}})
    assert not alternate_trainee_race_enabled({"env": {}})
    assert alternate_trainee_race_enabled({"alternate_trainee_race": True})


def test_pool_player_id_without_race_uses_agent_flag():
    pools = [{"agent": True}, {}]
    assert pool_player_id(pools[0], pools, 5, alternate=True) == 1
    assert pool_player_id(pools[1], pools, 5, alternate=True) == 2
