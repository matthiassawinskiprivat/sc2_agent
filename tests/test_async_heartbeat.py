"""Heartbeat / empty_cache opt-in helpers for async collector/learner."""
from train.async_train_universal import (
    DEFAULT_COLLECTOR_STEP_TIMEOUT_SEC,
    collector_fill_stalled,
    cuda_empty_cache_requested,
    heartbeat_due,
    resolve_collector_step_timeout,
)


def test_heartbeat_due_respects_interval():
    assert heartbeat_due(0.0, 29.9, 30.0) is False
    assert heartbeat_due(0.0, 30.0, 30.0) is True
    assert heartbeat_due(10.0, 40.0, 30.0) is True


def test_cuda_empty_cache_is_opt_in_only():
    assert cuda_empty_cache_requested(None, False) is False
    assert cuda_empty_cache_requested("", False) is False
    assert cuda_empty_cache_requested("1", False) is True
    assert cuda_empty_cache_requested("true", False) is True
    assert cuda_empty_cache_requested("", True) is True


def test_resolve_collector_step_timeout_default_and_floor():
    assert resolve_collector_step_timeout({}) == DEFAULT_COLLECTOR_STEP_TIMEOUT_SEC
    assert resolve_collector_step_timeout({"collector_step_timeout": 45}) == 45.0
    assert resolve_collector_step_timeout({"collector_step_timeout": 10}) == 30.0


def test_collector_fill_stalled_ignores_loop_ticks_without_rows():
    assert collector_fill_stalled(0.0, 119.9, 120.0) is False
    assert collector_fill_stalled(0.0, 120.0, 120.0) is True
