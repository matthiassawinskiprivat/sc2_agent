"""Async TensorBoard log directory layout."""
import os

from train.async_train_universal import resolve_async_log_dir


def test_resolve_async_log_dir_default_flat():
    path = resolve_async_log_dir("level_attack_0", {})
    assert path == os.path.join("logs_universal", "level_attack_0")


def test_resolve_async_log_dir_legacy_subdir():
    path = resolve_async_log_dir(
        "level_attack_0", {"tensorboard_subdir": "async"}
    )
    assert path == os.path.join("logs_universal", "level_attack_0", "async")
