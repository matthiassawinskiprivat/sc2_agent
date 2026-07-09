"""Tests for async learner/collector zip staging (Windows file-lock avoidance)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from train.async_train_universal import (
    CollectorWeightSync,
    copy_file_retry,
    maybe_poll_collector_weights,
    stage_learner_zip_for_collector,
)


def test_copy_file_retry_writes_identical_file(tmp_path: Path):
    src = tmp_path / "a.bin"
    dst = tmp_path / "b.bin"
    src.write_bytes(b"hello weights")
    copy_file_retry(str(src), str(dst))
    assert dst.read_bytes() == b"hello weights"


def test_maybe_poll_collector_weights_respects_interval(monkeypatch):
    import train.async_train_universal as mod
    from unittest.mock import MagicMock

    calls = {"n": 0}
    sync = MagicMock()
    sync.try_reload.side_effect = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1)

    state = {"last_poll": 0.0}
    times = iter([1.0, 1.05, 1.11, 1.15])

    monkeypatch.setattr(mod.time, "perf_counter", lambda: next(times))

    maybe_poll_collector_weights(sync, None, None, state)
    maybe_poll_collector_weights(sync, None, None, state)
    assert calls["n"] == 1
    maybe_poll_collector_weights(sync, None, None, state)
    assert calls["n"] == 2


def test_collector_weight_sync_skips_unchanged_mtime(tmp_path: Path, monkeypatch):
    import train.async_train_universal as mod

    monkeypatch.setattr(mod, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "COLLECTOR_WEIGHTS_STAGING_ZIP", str(tmp_path / "staging.zip"))

    learner_zip = tmp_path / "latest_learner_model.zip"
    learner_zip.write_bytes(b"PK\x03\x04v1")

    model = type("M", (), {"device": "cpu"})()
    vec_env = object()
    sync = CollectorWeightSync()

    calls = {"n": 0}

    def fake_set_parameters(*_a, **_k):
        calls["n"] += 1

    monkeypatch.setattr(mod, "set_parameters_with_retry", fake_set_parameters)
    monkeypatch.setattr(mod, "stage_learner_zip_for_collector", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "push_opponent_checkpoint_to_envs", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.utils.model_utils.inspect_checkpoint_policy_weights",
        lambda *_a, **_k: (True, "ok", []),
    )

    assert sync.try_reload(model, vec_env) is True
    assert calls["n"] == 1
    assert sync.try_reload(model, vec_env) is False
    assert calls["n"] == 1

    learner_zip.write_bytes(b"PK\x03\x04v2")
    import os
    import time

    time.sleep(0.05)
    os.utime(learner_zip, None)
    assert sync.try_reload(model, vec_env) is True
    assert calls["n"] == 2


def test_stage_learner_zip_for_collector_atomic_shape(tmp_path: Path):
    src = tmp_path / "latest.zip"
    staging = tmp_path / "staging.zip"
    src.write_bytes(b"PK\x03\x04fake")
    stage_learner_zip_for_collector(str(src), str(staging))
    assert staging.read_bytes() == b"PK\x03\x04fake"
    assert not os.path.exists(str(staging) + ".part")
