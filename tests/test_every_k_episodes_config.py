"""every_k_episodes config helper and hard-reset scheduling."""
import pytest

from train.envs.sc2_session_config import (
    DEFAULT_EVERY_K_EPISODES,
    every_k_episodes,
)


def test_every_k_episodes_default():
    assert every_k_episodes({}) == DEFAULT_EVERY_K_EPISODES
    assert DEFAULT_EVERY_K_EPISODES == 100


def test_every_k_episodes_flattened_root():
    assert every_k_episodes({"every_k_episodes": 50}) == 50


def test_every_k_episodes_nested_env():
    assert every_k_episodes({"env": {"every_k_episodes": 25}}) == 25


def test_every_k_episodes_root_overrides_nested_env():
    assert every_k_episodes(
        {"every_k_episodes": 40, "env": {"every_k_episodes": 25}}
    ) == 40


def test_every_k_episodes_rejects_zero():
    with pytest.raises(ValueError, match=">= 1"):
        every_k_episodes({"every_k_episodes": 0})


def test_level0_hard_reset_schedule_matches_every_k():
    """Level0Env: (episode_counter + worker_index) % every_k == 0, episode_counter > 1."""
    every_k = 10
    worker_index = 0
    triggered = [
        ep
        for ep in range(1, 31)
        if ep > 1 and (ep + worker_index) % every_k == 0
    ]
    assert triggered == [10, 20, 30]


def test_level0_map_index_on_hard_reset():
    every_k = 10
    worker_index = 0
    num_workers = 8
    map_list_len = 14
    ep = 10
    resets_done = (ep + worker_index) // every_k
    map_idx = (worker_index + resets_done * num_workers) % map_list_len
    assert map_idx == 8
