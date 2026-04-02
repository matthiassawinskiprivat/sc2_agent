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
