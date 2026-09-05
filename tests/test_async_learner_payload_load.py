"""Regression: learner must close np.load NpzFile to avoid Windows handle leaks and gradual slowdown."""
import os
import tempfile

import numpy as np


def test_np_load_context_manager_closes_archive():
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez(path, actions=np.zeros((4, 2, 1), dtype=np.float32))
        with np.load(path) as payload:
            assert "actions" in payload.files
            _ = np.array(payload["actions"])
        with open(path, "rb") as f:
            assert len(f.read()) > 0
    finally:
        os.remove(path)


def test_custom_metrics_deque_bounded():
    from collections import deque

    custom_metrics: dict = {}
    for i in range(250):
        key = "k"
        if key not in custom_metrics:
            custom_metrics[key] = deque(maxlen=100)
        custom_metrics[key].append(float(i))
    assert len(custom_metrics["k"]) == 100
    assert float(np.mean(custom_metrics["k"])) > 0.0


def test_heal_casts_published_as_rollout_sum_not_short_window_mean():
    """A rare heal is invisible as mean(last 100 steps) but visible as a rollout sum."""
    from train.collector_custom_metrics import (
        publish_custom_metrics,
        record_info_scalars,
    )

    custom_metrics: dict = {}
    rollout_sums: dict = {}
    for _ in range(99):
        record_info_scalars(
            custom_metrics,
            rollout_sums,
            {
                "heal_casts": 0,
                "heal_attempts": 1,
                "heal_casts_episode": 0,
                "squad_physics_step": True,
            },
        )
    record_info_scalars(
        custom_metrics,
        rollout_sums,
        {
            "heal_casts": 1,
            "heal_attempts": 1,
            "heal_casts_episode": 1,
            "squad_physics_step": True,
        },
    )
    signal: dict = {}
    publish_custom_metrics(signal, custom_metrics, rollout_sums)
    assert signal["env/heal_casts"] == 1.0
    assert signal["env/heal_attempts"] == 100.0
    assert signal["env/heal_casts_episode"] == 0.01
    assert "env/squad_physics_step" not in signal
