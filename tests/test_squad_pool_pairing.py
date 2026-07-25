"""Tests for CTF squad arena pairing helpers."""

import pytest

from train.envs.ctf.squad_pool_pairing import (
    paired_opponent_pool_index,
    trainee_pool_indices,
)


def _pools():
    return [
        {"high_level_command": "attack", "region": "A", "race": "terran"},
        {"high_level_command": "defend", "region": "A", "race": "zerg"},
        {"high_level_command": "defend", "region": "B", "race": "terran"},
        {"high_level_command": "attack", "region": "B", "race": "zerg"},
    ]


def _is_trainee(pool):
    return pool["race"] == "terran"


def test_trainee_pool_indices_terran_trainee():
    indices = trainee_pool_indices(_pools(), _is_trainee)
    assert indices == [0, 2]


def test_paired_opponent_pool_index_same_region():
    pools = _pools()
    assert paired_opponent_pool_index(0, pools, is_pool_trainee=_is_trainee) == 1
    assert paired_opponent_pool_index(2, pools, is_pool_trainee=_is_trainee) == 3


def test_paired_opponent_pool_index_ambiguous_raises():
    pools = [
        {"region": "A", "race": "terran"},
        {"region": "A", "race": "zerg"},
        {"region": "A", "race": "protoss"},
    ]

    def is_trainee(pool):
        return pool["race"] == "terran"

    with pytest.raises(ValueError, match="Ambiguous opponent pairing"):
        paired_opponent_pool_index(0, pools, is_pool_trainee=is_trainee)
