"""Tests for async learner/collector zip staging (Windows file-lock avoidance)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from train.async_train_universal import copy_file_retry, stage_learner_zip_for_collector


def test_copy_file_retry_writes_identical_file(tmp_path: Path):
    src = tmp_path / "a.bin"
    dst = tmp_path / "b.bin"
    src.write_bytes(b"hello weights")
    copy_file_retry(str(src), str(dst))
    assert dst.read_bytes() == b"hello weights"


def test_stage_learner_zip_for_collector_atomic_shape(tmp_path: Path):
    src = tmp_path / "latest.zip"
    staging = tmp_path / "staging.zip"
    src.write_bytes(b"PK\x03\x04fake")
    stage_learner_zip_for_collector(str(src), str(staging))
    assert staging.read_bytes() == b"PK\x03\x04fake"
    assert not os.path.exists(str(staging) + ".part")
