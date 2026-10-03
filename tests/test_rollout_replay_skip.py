"""Refuse a PPO weight step when the flat replay of saved decisions misses."""

from src.agent.maskable_recurrent_ppo import (
    ROLLOUT_REPLAY_ABS_ERR_SKIP,
    rollout_replay_should_skip,
)


def test_healthy_replay_is_kept():
    assert ROLLOUT_REPLAY_ABS_ERR_SKIP == 0.05
    assert rollout_replay_should_skip(1e-4) is False
    assert rollout_replay_should_skip(0.05) is False


def test_disconnected_replay_is_skipped():
    assert rollout_replay_should_skip(0.051) is True
    assert rollout_replay_should_skip(3.6) is True
    assert rollout_replay_should_skip(98.0) is True


def test_non_finite_replay_does_not_skip():
    assert rollout_replay_should_skip(float("nan")) is False
    assert rollout_replay_should_skip(float("inf")) is False
